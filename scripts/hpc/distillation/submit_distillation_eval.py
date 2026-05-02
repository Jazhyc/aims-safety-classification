#!/usr/bin/env python3
"""
Select best distillation adapters from W&B and submit eval jobs.

Two modes (--mode):
  ood-val  (default)
      For every (teacher × student × condition) combination, pick the adapter
      with the highest W&B val_harm_f1 across the LR sweep and submit one OOD
      validation job for it. OOD datasets are ToxicChat (train) and Aegis
      (validation). Already-evaluated adapters are skipped unless --force is
      given.

  test
      Read OOD val results from disk, pick the single adapter with the highest
      average F1 across all (teacher × student × condition) combinations, and
      submit one full test-set eval job for it. Requires OOD val to have been
      run first.

--cleanup removes every adapter in models/distillation-sweep/ that was not
selected as best in test mode. Combine with --dry-run to preview deletions.

Usage:
    python scripts/hpc/submit_distillation_eval.py --dry-run
    python scripts/hpc/submit_distillation_eval.py                    # OOD val
    python scripts/hpc/submit_distillation_eval.py --force
    python scripts/hpc/submit_distillation_eval.py --mode test        # test eval
    python scripts/hpc/submit_distillation_eval.py --mode test --dry-run
    python scripts/hpc/submit_distillation_eval.py --mode test --cleanup --dry-run
    python scripts/hpc/submit_distillation_eval.py --mode test --cleanup
    python scripts/hpc/submit_distillation_eval.py --traces-version v7

    # Top up already-evaluated combinations with a new dataset only:
    python scripts/hpc/submit_distillation_eval.py --mode test \\
        --eval-config eval_distillation_openai_mod \\
        --check-datasets openai_moderation
"""

import argparse
import json
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import wandb
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))
from slurm_utils import create_logs_dir, submit_sbatch, print_header

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DISTILLATION_SWEEP_DIR = PROJECT_ROOT / "models" / "distillation-sweep"

TEST_EVAL_TEMPLATE = "scripts/hpc/distillation/distillation_eval_template.sh"
OOD_VAL_TEMPLATE  = "scripts/hpc/distillation/distillation_ood_eval_template.sh"

OOD_VAL_BASE_DIR  = PROJECT_ROOT / "data" / "safety_experiment" / "ood_validation" / "distillation"
OOD_DATASET_DIRS  = ["toxic-chat", "aegis"]

# ── Sweep dimensions (must match submit_distillation_sweep.py) ────────────────

TEACHER_MODELS = [
    ("cyankiwi/Mistral-Small-4-119B-2603-AWQ-4bit",  "cyankiwi-Mistral-Small-4-119B-2603-AWQ-4bit"),
    ("Qwen/Qwen3-32B",                               "Qwen-Qwen3-32B"),
    ("openai/gpt-oss-120b",                           "openai-gpt-oss-120b"),
    ("RedHatAI/gemma-3-27b-it-quantized.w4a16",       "RedHatAI-gemma-3-27b-it-quantized.w4a16"),
]

STUDENT_MODELS = [
    ("meta-llama/Llama-3.1-8B-Instruct",              "llama-3.1-8b"),
    ("google/gemma-3-12b-it",                          "gemma-3-12b"),
    ("mistralai/Ministral-3-14B-Instruct-2512-BF16",   "ministral-3-14b-instruct"),
    ("Qwen/Qwen3-8B",                                  "qwen3-8b"),
]

LEARNING_RATES = [1e-5, 2e-5, 5e-5, 1e-4]
CONDITIONS = ["no_intent", "synthetic_intent", "human_intent"]
DEFAULT_TRACES_VERSION = "v7"

CONDITION_TO_EVAL = {
    "no_intent":        "finetuned_reasoning_classification",
    "synthetic_intent": "finetuned_reasoning_synthetic_intent",
    "human_intent":     "finetuned_reasoning_human_intent",
}

EVAL_DATASET_DIRS = [
    "annotated-intents",
    "wildguardmix",
    "xstest",
    "toxic-chat",
    "aegis",
    "openai-moderation",
]


# ── Completion checks ─────────────────────────────────────────────────────────

def _is_done(output_dir: str, check_datasets: list[str]) -> bool:
    base = PROJECT_ROOT / output_dir
    return all(
        any((base / ds_dir).glob("*.jsonl"))
        for ds_dir in check_datasets
    )


def compute_ood_val_f1(output_dir: Path) -> float | None:
    """Average harm F1 across both OOD datasets from local JSONL files."""
    f1_scores = []
    for ds_dir in OOD_DATASET_DIRS:
        ds_path = output_dir / ds_dir
        if not ds_path.exists():
            continue
        records = []
        for jsonl_file in ds_path.glob("*.jsonl"):
            with open(jsonl_file) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            records.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
        if not records:
            continue
        tp = fp = tn = fn = 0
        for r in records:
            true_harm = (r.get("true_harm_binary") or r.get("true_harm") or "").lower()
            pred_harm = (r.get("predicted_harm") or "").lower()
            if not true_harm or not pred_harm:
                continue
            if true_harm == "harmful" and pred_harm == "harmful":
                tp += 1
            elif true_harm == "safe" and pred_harm == "harmful":
                fp += 1
            elif true_harm == "safe" and pred_harm == "safe":
                tn += 1
            elif true_harm == "harmful" and pred_harm == "safe":
                fn += 1
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1_scores.append(f1)
    return sum(f1_scores) / len(f1_scores) if f1_scores else None


# ── W&B query ─────────────────────────────────────────────────────────────────

def fetch_all_distillation_adapters(
    traces_version: str,
) -> dict[tuple[str, str, str], list[tuple[Path, float]]]:
    """
    Fetch all finished distillation sweep runs across every (teacher × student) W&B project.
    Returns a mapping of
        (teacher_slug, student_slug, condition) -> [(adapter_path, val_harm_f1), ...]
    listing every qualifying adapter (all LRs), deduplicated by adapter path
    (most recently created run per path kept, since it overwrites disk weights).
    """
    api = wandb.Api()
    result: dict[tuple, list[tuple[Path, float]]] = defaultdict(list)

    for _, teacher_slug in TEACHER_MODELS:
        for _, student_slug in STUDENT_MODELS:
            project = f"distillation-sweep-{student_slug}--{teacher_slug}"
            print(f"  Querying W&B: {project} ...", end=" ", flush=True)
            try:
                runs = api.runs(path=project, filters={"state": "finished"})
            except Exception as e:
                print(f"[warn] {e}")
                continue

            candidates: dict[Path, tuple] = {}  # adapter_path -> (condition, f1, created_at)
            for run in runs:
                run_condition = (
                    run.summary.get("reasoning_traces_condition")
                    or run.config.get("data", {}).get("reasoning_traces_condition")
                )
                if run_condition not in CONDITIONS:
                    continue

                run_version = run.config.get("data", {}).get("traces_version")
                if run_version is not None:
                    if run_version != traces_version:
                        continue
                elif not run.name.endswith(f"--{traces_version}"):
                    continue

                f1 = run.summary.get("val_harm_f1")
                if f1 is None:
                    continue

                adapter_path_str = run.summary.get("adapter_path")
                if not adapter_path_str:
                    model_save_dir = run.config.get("paths", {}).get("model_save_dir")
                    if not model_save_dir:
                        continue
                    adapter_path_str = model_save_dir + "_adapter"

                adapter_path = Path(adapter_path_str)
                if not adapter_path.is_absolute():
                    adapter_path = PROJECT_ROOT / adapter_path
                if not adapter_path.exists():
                    continue

                created_at = getattr(run, "created_at", "")
                existing = candidates.get(adapter_path)
                if existing is None or created_at > existing[2]:
                    candidates[adapter_path] = (run_condition, float(f1), created_at)

            for adapter_path, (run_condition, f1, _) in candidates.items():
                key = (teacher_slug, student_slug, run_condition)
                result[key].append((adapter_path, f1))

            print(f"{len(candidates)} qualifying runs")

    return result


def fetch_best_by_wandb_f1(
    all_adapters: dict[tuple, list[tuple[Path, float]]]
) -> dict[tuple[str, str, str], tuple[Path, float]]:
    """Select the adapter with the highest W&B val_harm_f1 per (teacher, student, condition)."""
    best = {}
    for key, adapters in all_adapters.items():
        if adapters:
            best[key] = max(adapters, key=lambda x: x[1])
    return best


# ── Cleanup ───────────────────────────────────────────────────────────────────

def cleanup_suboptimal(best_adapters: set[Path], dry_run: bool):
    if not DISTILLATION_SWEEP_DIR.exists():
        print("  models/distillation-sweep/ does not exist, nothing to clean.")
        return
    all_dirs   = [p for p in DISTILLATION_SWEEP_DIR.iterdir() if p.is_dir()]
    to_delete  = [p for p in all_dirs if p not in best_adapters]
    to_keep    = [p for p in all_dirs if p in best_adapters]
    print(f"\n  Cleanup: {len(to_keep)} adapters kept, {len(to_delete)} to delete")
    for path in sorted(to_delete):
        if dry_run:
            print(f"  [dry-run] would delete: {path.name}")
        else:
            shutil.rmtree(path)
            print(f"  deleted: {path.name}")


# ── Mode implementations ──────────────────────────────────────────────────────

def ood_val_mode(traces_version: str, dry_run: bool, force: bool):
    """Submit OOD validation for the best-LR adapter per (teacher × student × condition)."""
    print(f"\nOOD datasets: ToxicChat (train split) + Aegis (validation split)")
    print(f"Output base:  {OOD_VAL_BASE_DIR}")

    print("\nFetching all adapters from W&B ...")
    all_adapters = fetch_all_distillation_adapters(traces_version)
    total = sum(len(v) for v in all_adapters.values())
    print(f"Found {total} adapter(s) across {len(all_adapters)} (teacher, student, condition) groups.")

    best_adapters = fetch_best_by_wandb_f1(all_adapters)
    print(f"Selected best LR per combination by W&B val_harm_f1: {len(best_adapters)} adapter(s).\n")

    submitted, skipped = [], []

    for (teacher_slug, student_slug, condition), (adapter_path, wandb_f1) in sorted(best_adapters.items()):
        student_hf = next(hf for hf, slug in STUDENT_MODELS if slug == student_slug)
        eval_condition = CONDITION_TO_EVAL[condition]

        ood_output_dir = OOD_VAL_BASE_DIR / teacher_slug / student_slug / condition / adapter_path.name
        label = f"{teacher_slug}/{student_slug}/{condition}/{adapter_path.name}"

        if not force and compute_ood_val_f1(ood_output_dir) is not None:
            print(f"  [done] {adapter_path.name} — OOD val results present, skipping (--force to re-run)")
            skipped.append(label)
            continue

        run_label = f"distill-ood-val--{teacher_slug}--{student_slug}--{condition}--{adapter_path.name}"
        print(f"  {'[dry-run] ' if dry_run else ''}submit {adapter_path.name}  "
              f"(wandb_f1={wandb_f1:.4f})")

        if dry_run:
            continue

        export_vars = {
            "STUDENT_MODEL":  student_hf,
            "ADAPTER_PATH":   str(adapter_path),
            "EVAL_CONDITION": eval_condition,
            "OUTPUT_DIR":     str(ood_output_dir),
            "WANDB_RUN_NAME": run_label,
        }
        try:
            job_id = submit_sbatch(OOD_VAL_TEMPLATE, export_vars)
            print(f"    ✓ job {job_id}")
            submitted.append((label, job_id))
        except subprocess.CalledProcessError as e:
            print(f"    ✗ submission failed: {e.stderr}")

    print(f"\nSubmitted {len(submitted)}  |  skipped (done) {len(skipped)}")


def test_mode(
    traces_version: str,
    eval_config: str,
    check_datasets: list[str],
    dry_run: bool,
    force: bool,
    do_cleanup: bool,
):
    """Pick the single adapter with the highest OOD val F1 and submit one test eval."""
    print(f"\nConfig: {eval_config}")
    print("Selecting single best adapter across all combinations by OOD val F1 ...")

    print("\nFetching all adapters from W&B ...")
    all_adapters = fetch_all_distillation_adapters(traces_version)
    print(f"Found {len(all_adapters)} (teacher, student, condition) groups.\n")

    if not dry_run:
        create_logs_dir()

    # Score every adapter that has OOD val results; rank globally.
    candidates: list[tuple[float, str, str, str, Path, float]] = []
    no_ood: list[str] = []
    for (teacher_slug, student_slug, condition), adapters in sorted(all_adapters.items()):
        for adapter_path, wandb_f1 in adapters:
            ood_output_dir = OOD_VAL_BASE_DIR / teacher_slug / student_slug / condition / adapter_path.name
            ood_f1 = compute_ood_val_f1(ood_output_dir)
            if ood_f1 is not None:
                candidates.append((ood_f1, teacher_slug, student_slug, condition, adapter_path, wandb_f1))
            else:
                no_ood.append(f"{teacher_slug}/{student_slug}/{condition}/{adapter_path.name}")

    if not candidates:
        print("  [error] No adapters have OOD val results. Run --mode ood-val first.")
        return

    candidates.sort(key=lambda x: x[0], reverse=True)
    print(f"Adapters scored on OOD val: {len(candidates)} (skipped {len(no_ood)} without OOD results)")
    print("Top 5 by OOD val F1:")
    for ood_f1, t, s, c, a, w in candidates[:5]:
        print(f"  {ood_f1:.4f}  {t}/{s}/{c}/{a.name}  (wandb_f1={w:.4f})")

    best_ood_f1, teacher_slug, student_slug, condition, best_adapter, _ = candidates[0]
    student_hf = next(hf for hf, slug in STUDENT_MODELS if slug == student_slug)
    eval_condition = CONDITION_TO_EVAL[condition]
    label = f"{teacher_slug}/{student_slug}/{condition}"
    output_dir = f"data/safety_experiment/distillation/{teacher_slug}/{student_slug}/{condition}"
    run_label = f"distill--{teacher_slug}--{student_slug}--{condition}"

    print(f"\n  => selected: {label}/{best_adapter.name}  ood_f1={best_ood_f1:.4f}")

    submitted = False
    if not force and _is_done(output_dir, check_datasets):
        print("  [done] Output files present — skipping (--force to re-run)")
    elif dry_run:
        print("  [dry-run] would submit eval job")
    else:
        export_vars = {
            "STUDENT_MODEL":  student_hf,
            "ADAPTER_PATH":   str(best_adapter),
            "EVAL_CONDITION": eval_condition,
            "EVAL_CONFIG":    eval_config,
            "OUTPUT_DIR":     output_dir,
            "WANDB_RUN_NAME": run_label,
        }
        try:
            job_id = submit_sbatch(TEST_EVAL_TEMPLATE, export_vars)
            print(f"  ✓ submitted job {job_id}")
            submitted = True
        except subprocess.CalledProcessError as e:
            print(f"  ✗ submission failed: {e.stderr}")

    if do_cleanup:
        print("\n── Adapter cleanup ──")
        cleanup_suboptimal({best_adapter}, dry_run=dry_run)

    print("\n" + "=" * 60)
    print(f"Best adapter: {label}/{best_adapter.name}  ood_f1={best_ood_f1:.4f}")
    print(f"Submitted: {'yes' if submitted else 'no'}")
    print("=" * 60)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode", choices=["ood-val", "test"], default="ood-val",
        help=(
            "ood-val: submit OOD validation jobs for all trained adapters (default). "
            "test: pick best adapter per combination by OOD val F1 and submit full test eval."
        ),
    )
    parser.add_argument("--dry-run", "-n", action="store_true",
                        help="Print planned actions without submitting jobs or deleting files.")
    parser.add_argument("--force", action="store_true",
                        help="Re-submit jobs even for combinations that already have output files.")
    parser.add_argument("--cleanup", action="store_true",
                        help="(test mode only) Delete adapters not selected as best.")
    parser.add_argument("--traces-version", default=DEFAULT_TRACES_VERSION, metavar="VERSION",
                        help=f"Reasoning traces version (default: {DEFAULT_TRACES_VERSION}).")
    parser.add_argument("--eval-config", default="eval_distillation", metavar="CONFIG",
                        help="Hydra config name for test eval (default: eval_distillation).")
    parser.add_argument("--check-datasets", default=None, metavar="DS1,DS2,...",
                        help="Comma-separated dataset dirs for completion check (test mode). "
                             "Defaults to all EVAL_DATASET_DIRS.")
    args = parser.parse_args()

    traces_version = args.traces_version
    check_datasets = args.check_datasets.split(",") if args.check_datasets else EVAL_DATASET_DIRS

    n_combos = len(TEACHER_MODELS) * len(STUDENT_MODELS) * len(CONDITIONS)
    print_header(
        "Distillation Eval",
        f"mode={args.mode}  "
        f"{len(TEACHER_MODELS)} teachers × {len(STUDENT_MODELS)} students × "
        f"{len(CONDITIONS)} conditions = {n_combos} combinations"
        f"  |  traces: {traces_version}"
        + (" [DRY RUN]" if args.dry_run else "")
        + (" [FORCE]" if args.force else ""),
    )

    if args.mode == "ood-val":
        ood_val_mode(traces_version, args.dry_run, args.force)
    else:
        test_mode(
            traces_version,
            args.eval_config,
            check_datasets,
            args.dry_run,
            args.force,
            args.cleanup,
        )


if __name__ == "__main__":
    main()
