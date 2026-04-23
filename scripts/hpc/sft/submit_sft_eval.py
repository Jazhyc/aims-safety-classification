#!/usr/bin/env python3
"""
Select SFT adapters from W&B and submit eval jobs.

Two modes (--mode):
  ood-val  (default)
      Submit one OOD validation job per adapter for every LR that has been
      trained. OOD datasets are ToxicChat (train) and Aegis (validation) —
      splits not used during training or final test evaluation.
      Already-evaluated adapters are skipped unless --force is given.

  test
      Read OOD val results from disk, pick the adapter with the highest
      average F1 per mode (generation / classification), and submit a single
      full test-set eval job using the eval_sft_baselines config.
      Requires OOD val to have been run first.

Usage:
    python scripts/hpc/submit_sft_eval.py                      # OOD val (all adapters)
    python scripts/hpc/submit_sft_eval.py --dry-run
    python scripts/hpc/submit_sft_eval.py --mode test           # test eval (best by OOD F1)
    python scripts/hpc/submit_sft_eval.py --mode test --dry-run
    python scripts/hpc/submit_sft_eval.py --use-merged          # test mode with merged weights
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import wandb
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))
from slurm_utils import create_logs_dir, submit_sbatch, print_header

WANDB_PROJECT = "sft-hyperparam-sweep"
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_TEST_CONFIG = "eval_sft_baselines"
OOD_VAL_TEMPLATE  = "scripts/hpc/sft/sft_ood_eval_template.sh"
TEST_EVAL_TEMPLATE = "scripts/hpc/sft/sft_eval_template.sh"

PROJECT_ROOT = Path(__file__).resolve().parents[3]
OOD_VAL_BASE_DIR = PROJECT_ROOT / "data" / "safety_experiment" / "ood_validation" / "sft"
OOD_DATASET_DIRS = ["toxic_chat", "aegis"]


# ── W&B helpers ───────────────────────────────────────────────────────────────

def fetch_all_adapters(project: str, classification_only: bool) -> list[tuple[str, float, str]]:
    """
    Query W&B for all finished runs matching the given mode.
    Deduplicates by adapter path, keeping only the most recently created run per path
    (the weights on disk always reflect the last run that wrote to that path).
    Returns list of (adapter_path_str, val_harm_f1, run_name).
    """
    mode_label = "classification" if classification_only else "generation"
    print(f"\n[{mode_label}] Querying W&B project '{project}'...")

    api = wandb.Api()
    runs = api.runs(path=project, filters={"state": "finished"})

    latest_by_path: dict[str, tuple[str, float, str]] = {}  # path -> (run_name, f1, created_at)

    for run in runs:
        summary = run.summary

        run_clf_only = summary.get("classification_only")
        if run_clf_only is None:
            run_clf_only = run.config.get("data", {}).get("classification_only", False)
        if bool(run_clf_only) != classification_only:
            continue

        f1 = summary.get("val_harm_f1")
        if f1 is None:
            print(f"  [skip] {run.name}: no val_harm_f1 in summary")
            continue

        adapter_path_str = summary.get("adapter_path")
        if not adapter_path_str:
            model_save_dir = run.config.get("paths", {}).get("model_save_dir")
            if model_save_dir:
                adapter_path_str = model_save_dir + "_adapter"
                print(f"  [info] {run.name}: derived adapter path from config: {adapter_path_str!r}")
            else:
                print(f"  [skip] {run.name}: no adapter_path in summary and no model_save_dir in config")
                continue

        adapter_path = Path(adapter_path_str)
        if not adapter_path.is_absolute():
            adapter_path = PROJECT_ROOT / adapter_path
        adapter_path_str = str(adapter_path)
        try:
            if not adapter_path.exists():
                print(f"  [skip] {run.name}: adapter not found at {adapter_path_str}")
                continue
        except OSError as e:
            print(f"  [skip] {run.name}: cannot access adapter path ({e})")
            continue

        created_at = run.created_at
        existing = latest_by_path.get(adapter_path_str)
        if existing is None or created_at > existing[2]:
            if existing is not None:
                print(f"  [dup] {adapter_path.name}: replacing {existing[0]} (older) with {run.name} (newer)")
            latest_by_path[adapter_path_str] = (run.name, float(f1), created_at)

    result = [(path, f1, name) for path, (name, f1, _) in latest_by_path.items()]
    print(f"  {len(result)} unique adapter(s)")
    for path, f1, name in sorted(result, key=lambda x: x[1], reverse=True):
        print(f"    {name}: val_harm_f1={f1:.4f}  adapter={Path(path).name}")
    return result


# ── OOD val F1 helpers ────────────────────────────────────────────────────────

def _ood_output_dir(mode_label: str, adapter_path: str) -> Path:
    """Per-adapter OOD val output directory."""
    return OOD_VAL_BASE_DIR / mode_label / Path(adapter_path).name


def compute_ood_val_f1(output_dir: Path) -> float | None:
    """
    Average harm F1 across both OOD validation datasets from local JSONL files.
    Returns None if no prediction files exist yet.
    """
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


# ── Merged model helpers ──────────────────────────────────────────────────────

def _merged_path_for(adapter_path: str) -> str:
    p = Path(adapter_path)
    stem = p.name
    stem = (stem[: -len("_adapter")] + "_merged") if stem.endswith("_adapter") else (stem + "_merged")
    return str(p.parent / stem)


# ── Mode implementations ──────────────────────────────────────────────────────

def ood_val_mode(project: str, dry_run: bool, force: bool):
    """Submit one OOD validation job per adapter (all LRs × both modes)."""
    print(f"\nOOD datasets: ToxicChat (train split) + Aegis (validation split)")
    print(f"Output base:  {OOD_VAL_BASE_DIR}")

    submitted, skipped = [], []

    for clf_only in [False, True]:
        mode_label = "classification" if clf_only else "generation"
        condition  = "finetuned_classification" if clf_only else "finetuned_generation"
        adapters   = fetch_all_adapters(project, clf_only)

        for adapter_path_str, _, run_name in adapters:
            adapter_name = Path(adapter_path_str).name
            output_dir   = _ood_output_dir(mode_label, adapter_path_str)
            label        = f"{mode_label}/{adapter_name}"

            if not force and compute_ood_val_f1(output_dir) is not None:
                print(f"  [done] {label} — OOD val results present, skipping (--force to re-run)")
                skipped.append(label)
                continue

            print(f"  {'[dry-run] ' if dry_run else ''}submit {label}")
            if dry_run:
                continue

            export_vars = {
                "GENERATION_ADAPTER":     adapter_path_str if not clf_only else "MISSING",
                "CLASSIFICATION_ADAPTER": adapter_path_str if clf_only     else "MISSING",
                "EVAL_CONDITION":         condition,
                "OUTPUT_DIR":             str(output_dir),
                "WANDB_RUN_NAME":         f"sft-ood-val--{mode_label}--{adapter_name}",
            }
            try:
                job_id = submit_sbatch(OOD_VAL_TEMPLATE, export_vars)
                print(f"    ✓ job {job_id}  (logs/slurm/sft_ood_eval-{job_id}.out)")
                submitted.append((label, job_id))
            except subprocess.CalledProcessError as e:
                print(f"    ✗ submission failed: {e.stderr}")

    print(f"\nSubmitted {len(submitted)}  |  skipped (done) {len(skipped)}")


def test_mode(project: str, config_name: str, use_merged: bool, dry_run: bool):
    """Pick best adapter per mode by OOD val F1 and submit full test-set eval."""
    print(f"\nConfig: {config_name}")
    print("Selecting best adapter per mode by average OOD val F1 ...")

    best: dict[str, tuple[str, float]] = {}  # mode_label -> (adapter_path, ood_f1)

    for clf_only in [False, True]:
        mode_label = "classification" if clf_only else "generation"
        adapters   = fetch_all_adapters(project, clf_only)

        print(f"\n  [{mode_label}]")
        for adapter_path_str, _, run_name in adapters:
            output_dir = _ood_output_dir(mode_label, adapter_path_str)
            ood_f1     = compute_ood_val_f1(output_dir)
            if ood_f1 is None:
                print(f"    [skip] {run_name}: no OOD val results — run --mode ood-val first")
                continue
            print(f"    {run_name}: ood_f1={ood_f1:.4f}")
            if mode_label not in best or ood_f1 > best[mode_label][1]:
                best[mode_label] = (adapter_path_str, ood_f1)

    gen_adapter = best.get("generation",     (None, None))[0]
    clf_adapter = best.get("classification", (None, None))[0]
    gen_f1      = best.get("generation",     (None, None))[1]
    clf_f1      = best.get("classification", (None, None))[1]

    print("\n" + "=" * 60)
    print("Selected adapters (by OOD val F1):")
    print(f"  generation:     {Path(gen_adapter).name if gen_adapter else 'NOT FOUND'}  "
          + (f"(ood_f1={gen_f1:.4f})" if gen_f1 is not None else ""))
    print(f"  classification: {Path(clf_adapter).name if clf_adapter else 'NOT FOUND'}  "
          + (f"(ood_f1={clf_f1:.4f})" if clf_f1 is not None else ""))
    print("=" * 60)

    if gen_adapter is None and clf_adapter is None:
        print("\nNo OOD val results found. Run --mode ood-val first.")
        return

    if use_merged:
        gen_merged = _merged_path_for(gen_adapter) if gen_adapter else None
        clf_merged = _merged_path_for(clf_adapter) if clf_adapter else None
        missing = []
        for label, adapter, merged in [("generation", gen_adapter, gen_merged),
                                        ("classification", clf_adapter, clf_merged)]:
            if merged is None:
                continue
            exists = Path(merged).exists()
            print(f"  {label} merged: {merged}  [{'✓' if exists else '✗ MISSING'}]")
            if not exists:
                missing.append((label, adapter, merged))
        if missing:
            print("\nMerged models missing. Run scripts/dpo/merge_adapter.py first.")
            if not dry_run:
                return

    if dry_run:
        print("\n--dry-run: not submitting job.")
        return

    if use_merged:
        submitted = []
        for label, adapter, condition, merged in [
            ("generation",     gen_adapter, "finetuned_generation",     gen_merged if gen_adapter else None),
            ("classification", clf_adapter, "finetuned_classification", clf_merged if clf_adapter else None),
        ]:
            if merged is None:
                continue
            try:
                job_id = submit_sbatch("scripts/hpc/sft/sft_eval_merged_template.sh",
                                       {"MERGED_MODEL_PATH": merged, "CONDITION": condition})
                print(f"✓ Submitted {label} merged-model eval job: {job_id}")
                submitted.append(job_id)
            except subprocess.CalledProcessError as e:
                print(f"✗ Submission failed for {label}: {e.stderr}")
    else:
        export_vars = {
            "GENERATION_ADAPTER":     gen_adapter or "MISSING",
            "CLASSIFICATION_ADAPTER": clf_adapter or "MISSING",
            "CONFIG_NAME":            config_name,
        }
        try:
            job_id = submit_sbatch(TEST_EVAL_TEMPLATE, export_vars)
            print(f"\n✓ Submitted test eval job: {job_id}")
            print(f"  Monitor: squeue -j {job_id}")
            print(f"  Logs:    logs/slurm/sft_eval-{job_id}.out")
        except subprocess.CalledProcessError as e:
            print(f"\n✗ Submission failed: {e.stderr}")


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
            "test: pick best adapter per mode by OOD val F1 and submit full test eval."
        ),
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Print planned actions without submitting jobs.")
    parser.add_argument("--force", action="store_true",
                        help="Re-submit OOD val jobs even if output files already exist.")
    parser.add_argument("--project", type=str, default=WANDB_PROJECT,
                        help=f"W&B project to query (default: {WANDB_PROJECT}).")
    parser.add_argument("--config-name", type=str, default=DEFAULT_TEST_CONFIG,
                        help=f"Hydra config for test mode (default: {DEFAULT_TEST_CONFIG}).")
    parser.add_argument(
        "--use-merged", action="store_true",
        help=(
            "Test mode only: evaluate using pre-merged models instead of LoRA adapters. "
            "Run scripts/dpo/merge_adapter.py first."
        ),
    )
    args = parser.parse_args()

    print_header(
        "SFT Eval",
        f"mode={args.mode}  project={args.project}"
        + (f"  config={args.config_name}" if args.mode == "test" else "")
        + (" [merged]" if args.use_merged else "")
        + (" [DRY RUN]" if args.dry_run else ""),
    )

    if not args.dry_run:
        create_logs_dir()

    if args.mode == "ood-val":
        ood_val_mode(args.project, args.dry_run, args.force)
    else:
        test_mode(args.project, args.config_name, args.use_merged, args.dry_run)


if __name__ == "__main__":
    main()
