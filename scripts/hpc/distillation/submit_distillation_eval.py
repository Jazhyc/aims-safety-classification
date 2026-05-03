#!/usr/bin/env python3
"""
Select best distillation adapters by reading val_metrics.json from disk and submit eval jobs.

Two modes (--mode):
  ood-val  (default)
      For every (teacher × student × condition) combination, pick the adapter
      with the highest training-time val_harm_f1 across the LR sweep and submit
      one OOD validation job for it. OOD datasets are ToxicChat (train) and
      Aegis (validation). Already-evaluated adapters are skipped unless --force
      is given.

  test
      Read OOD val results from disk and select via marginal means (more robust
      than picking the single global-best cell, which can be a lucky outlier):
        1. Pick the teacher with the highest mean OOD F1 averaged across
           (student × condition).
        2. Pick the student with the highest mean OOD F1 averaged across
           (teacher × condition).
        3. Pick the condition that maximises the average of the chosen teacher's
           per-condition marginal across students and the chosen student's
           per-condition marginal across teachers — matches the per-condition
           bars in the notebook plots.
      Submit one full test-set eval job for that adapter. Requires OOD val to
      have been run first.

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
            if not true_harm:
                continue
            # Parse failures (no predicted_harm) are treated as "safe" — the
            # safety classifier failed to flag, which is the relevant failure
            # mode for downstream use. This penalises models that ramble past
            # max_new_tokens without producing a harm label.
            pred_harm = (r.get("predicted_harm") or "safe").lower()
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


# ── Adapter discovery ─────────────────────────────────────────────────────────

def fetch_all_distillation_adapters(
    traces_version: str,
) -> dict[tuple[str, str, str], list[tuple[Path, float]]]:
    """
    Enumerate adapter directories under DISTILLATION_SWEEP_DIR and read each
    adapter's val_metrics.json (written by causal.py after training) to recover
    val_harm_f1. Returns a mapping of
        (teacher_slug, student_slug, condition) -> [(adapter_path, val_harm_f1), ...]
    listing every qualifying adapter (all LRs).

    Adapter directory naming:
        {teacher_slug}--{student_slug}--{cond_slug}--lr{lr}--{version}_adapter
    """
    known_teachers = {slug for _, slug in TEACHER_MODELS}
    known_students = {slug for _, slug in STUDENT_MODELS}
    suffix = f"--{traces_version}_adapter"

    result: dict[tuple, list[tuple[Path, float]]] = defaultdict(list)
    if not DISTILLATION_SWEEP_DIR.exists():
        return result

    counts: dict[tuple[str, str], int] = defaultdict(int)
    for adapter_path in sorted(DISTILLATION_SWEEP_DIR.iterdir()):
        if not adapter_path.is_dir() or not adapter_path.name.endswith(suffix):
            continue
        stem = adapter_path.name[: -len(suffix)]
        parts = stem.split("--")
        if len(parts) != 4:
            continue
        teacher_slug, student_slug, cond_slug, _lr_part = parts
        if teacher_slug not in known_teachers or student_slug not in known_students:
            continue
        condition = cond_slug.replace("-", "_")
        if condition not in CONDITIONS:
            continue

        metrics_path = adapter_path / "val_metrics.json"
        if not metrics_path.exists():
            continue
        try:
            metrics = json.loads(metrics_path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        f1 = metrics.get("val_harm_f1")
        if f1 is None:
            continue

        result[(teacher_slug, student_slug, condition)].append((adapter_path, float(f1)))
        counts[(teacher_slug, student_slug)] += 1

    for (t, s), n in sorted(counts.items()):
        print(f"  {t} / {s}: {n} adapter(s) with val_metrics.json")

    return result


def fetch_best_by_val_f1(
    all_adapters: dict[tuple, list[tuple[Path, float]]]
) -> dict[tuple[str, str, str], tuple[Path, float]]:
    """Select the adapter with the highest training val_harm_f1 per (teacher, student, condition)."""
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

    print("\nEnumerating adapters from disk ...")
    all_adapters = fetch_all_distillation_adapters(traces_version)
    total = sum(len(v) for v in all_adapters.values())
    print(f"Found {total} adapter(s) across {len(all_adapters)} (teacher, student, condition) groups.")

    best_adapters = fetch_best_by_val_f1(all_adapters)
    print(f"Selected best LR per combination by training val_harm_f1: {len(best_adapters)} adapter(s).\n")

    submitted, skipped = [], []

    for (teacher_slug, student_slug, condition), (adapter_path, val_f1) in sorted(best_adapters.items()):
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
              f"(val_f1={val_f1:.4f})")

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
    """Pick a teacher × student × condition via marginal means and submit one test eval.

    Selection procedure (more robust than picking the single global-best cell, which
    can be a lucky outlier):
      1. For each (teacher × student × condition) combo, take the best LR by OOD F1.
      2. Pick the teacher with the highest mean OOD F1 across all (student, condition) cells.
      3. Pick the student with the highest mean OOD F1 across all (teacher, condition) cells.
      4. For the resulting (teacher, student) pair, pick the condition with the highest OOD F1.
    """
    print(f"\nConfig: {eval_config}")
    print("Selecting (teacher, student) by marginal means, then best condition for the pair ...")

    print("\nEnumerating adapters from disk ...")
    all_adapters = fetch_all_distillation_adapters(traces_version)
    print(f"Found {len(all_adapters)} (teacher, student, condition) groups.\n")

    if not dry_run:
        create_logs_dir()

    # Step 1 — best LR per (teacher, student, condition) by OOD F1.
    best_per_combo: dict[tuple[str, str, str], tuple[Path, float, float]] = {}
    no_ood: list[str] = []
    for (teacher_slug, student_slug, condition), adapters in all_adapters.items():
        best = None
        for adapter_path, val_f1 in adapters:
            ood_dir = OOD_VAL_BASE_DIR / teacher_slug / student_slug / condition / adapter_path.name
            ood_f1 = compute_ood_val_f1(ood_dir)
            if ood_f1 is None:
                no_ood.append(f"{teacher_slug}/{student_slug}/{condition}/{adapter_path.name}")
                continue
            if best is None or ood_f1 > best[1]:
                best = (adapter_path, ood_f1, val_f1)
        if best is not None:
            best_per_combo[(teacher_slug, student_slug, condition)] = best

    if not best_per_combo:
        print("  [error] No adapters have OOD val results. Run --mode ood-val first.")
        return

    print(f"Combos with OOD val results: {len(best_per_combo)} (skipped {len(no_ood)} adapters without OOD results)")

    # Step 2 — marginal teacher means across (student, condition).
    teacher_scores: dict[str, list[float]] = defaultdict(list)
    student_scores: dict[str, list[float]] = defaultdict(list)
    for (t, s, c), (_, ood_f1, _) in best_per_combo.items():
        teacher_scores[t].append(ood_f1)
        student_scores[s].append(ood_f1)
    teacher_means = {t: sum(v) / len(v) for t, v in teacher_scores.items()}
    student_means = {s: sum(v) / len(v) for s, v in student_scores.items()}

    print("\nMarginal teacher means (across students × conditions):")
    for t, m in sorted(teacher_means.items(), key=lambda x: -x[1]):
        print(f"  {m:.4f}  {t}  (n={len(teacher_scores[t])})")
    print("\nMarginal student means (across teachers × conditions):")
    for s, m in sorted(student_means.items(), key=lambda x: -x[1]):
        print(f"  {m:.4f}  {s}  (n={len(student_scores[s])})")

    best_teacher = max(teacher_means, key=teacher_means.get)
    best_student = max(student_means, key=student_means.get)

    # Step 3 — best condition for the chosen pair, decided via marginals (matches
    # the notebook plots): for each condition, take the mean of
    #   - the teacher's marginal across students for that condition, and
    #   - the student's marginal across teachers for that condition.
    teacher_cond_marginal: dict[str, list[float]] = defaultdict(list)
    student_cond_marginal: dict[str, list[float]] = defaultdict(list)
    for (t, s, c), (_, ood_f1, _) in best_per_combo.items():
        if t == best_teacher:
            teacher_cond_marginal[c].append(ood_f1)
        if s == best_student:
            student_cond_marginal[c].append(ood_f1)

    cond_blend: dict[str, float] = {}
    for c in CONDITIONS:
        t_marg = sum(teacher_cond_marginal[c]) / len(teacher_cond_marginal[c]) if teacher_cond_marginal[c] else None
        s_marg = sum(student_cond_marginal[c]) / len(student_cond_marginal[c]) if student_cond_marginal[c] else None
        if t_marg is None or s_marg is None:
            continue
        cond_blend[c] = (t_marg + s_marg) / 2

    if not cond_blend:
        print(f"  [error] No OOD val data for either marginal of {best_teacher}/{best_student}.")
        return

    print(f"\nSelected pair: {best_teacher} / {best_student}")
    print("Conditions ranked by mean(teacher_marginal, student_marginal):")
    for c in sorted(cond_blend, key=lambda x: -cond_blend[x]):
        t_m = sum(teacher_cond_marginal[c]) / len(teacher_cond_marginal[c])
        s_m = sum(student_cond_marginal[c]) / len(student_cond_marginal[c])
        print(f"  blend={cond_blend[c]:.4f}  (teacher_marg={t_m:.4f}, student_marg={s_m:.4f})  {c}")

    condition = max(cond_blend, key=cond_blend.get)
    teacher_slug, student_slug = best_teacher, best_student
    if (teacher_slug, student_slug, condition) not in best_per_combo:
        print(f"  [error] Selected condition {condition} has no OOD val data for "
              f"the {best_teacher}/{best_student} pair specifically.")
        return
    best_adapter, best_ood_f1, _ = best_per_combo[(teacher_slug, student_slug, condition)]
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
