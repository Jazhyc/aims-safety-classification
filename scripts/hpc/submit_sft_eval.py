#!/usr/bin/env python3
"""
Select best SFT adapters from W&B and submit an eval job.

Queries the sft-hyperparam-sweep W&B project, finds the finished run with the
highest val_harm_f1 for each sweep type (generation and classification-only),
derives the adapter path from the run config, and submits a SLURM eval job.

By default, adapters are served via vLLM's native LoRA support (no disk overhead).
Use --use-merged only if you have pre-merged models and want to avoid LoRA overhead.

Usage:
    python scripts/hpc/submit_sft_eval.py              # submit SLURM job (LoRA)
    python scripts/hpc/submit_sft_eval.py --dry-run    # print selected adapters only
    python scripts/hpc/submit_sft_eval.py --use-merged # eval on pre-merged models
"""

import argparse
import subprocess
from pathlib import Path

import wandb
from dotenv import load_dotenv

load_dotenv()

from slurm_utils import create_logs_dir, submit_sbatch, print_header

WANDB_PROJECT = "sft-hyperparam-sweep"
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

# Project root — resolve relative adapter paths saved by older training runs
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def fetch_best_adapter(project: str, classification_only: bool) -> tuple[str, float] | tuple[None, None]:
    """
    Query W&B for finished runs matching the given mode, ranked by val_harm_f1.

    Returns (adapter_path, val_harm_f1) for the best run, or (None, None) if no
    qualifying run is found.
    """
    mode_label = "classification-only" if classification_only else "generation"
    print(f"\n[{mode_label}] Querying W&B project '{project}'...")

    api = wandb.Api()
    runs = api.runs(
        path=project,
        filters={"state": "finished"},
    )

    best_f1 = -1.0
    best_adapter = None
    best_run_name = None

    for run in runs:
        summary = run.summary

        # Filter by sweep type — prefer summary flag, fall back to run config
        run_clf_only = summary.get("classification_only")
        if run_clf_only is None:
            run_clf_only = run.config.get("data", {}).get("classification_only", False)
        run_clf_only = bool(run_clf_only)
        if run_clf_only != classification_only:
            continue

        f1 = summary.get("val_harm_f1")
        if f1 is None:
            print(f"  [skip] {run.name}: no val_harm_f1 in summary")
            continue

        adapter_path_str = summary.get("adapter_path")
        if not adapter_path_str:
            # Fall back to deriving path from run config (paths/model_save_dir + _adapter)
            model_save_dir = run.config.get("paths", {}).get("model_save_dir")
            if model_save_dir:
                adapter_path_str = model_save_dir + "_adapter"
                print(f"  [info] {run.name}: no adapter_path in summary, derived from config: {adapter_path_str!r}")
            else:
                print(f"  [skip] {run.name}: no adapter_path in summary and no model_save_dir in config")
                continue

        adapter_path = Path(adapter_path_str)
        if not adapter_path.is_absolute():
            adapter_path = PROJECT_ROOT / adapter_path
        adapter_path_str = str(adapter_path)
        if not adapter_path.exists():
            print(f"  [skip] {run.name}: adapter not found locally at {adapter_path_str}")
            continue

        print(f"  {run.name}: val_harm_f1={f1:.4f}  adapter={adapter_path_str}")

        if f1 > best_f1:
            best_f1 = f1
            best_adapter = adapter_path_str
            best_run_name = run.name

    if best_adapter is None:
        print(f"  No qualifying runs found for {mode_label}.")
        return None, None

    print(f"  => Best: {best_run_name}  val_harm_f1={best_f1:.4f}")
    return best_adapter, best_f1


def _merged_path_for(adapter_path: str) -> str:
    """Derive the merged model path from an adapter path (_adapter -> _merged)."""
    p = Path(adapter_path)
    stem = p.name
    if stem.endswith("_adapter"):
        stem = stem[: -len("_adapter")] + "_merged"
    else:
        stem = stem + "_merged"
    return str(p.parent / stem)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print selected adapters without submitting a job."
    )
    parser.add_argument(
        "--project", type=str, default=WANDB_PROJECT,
        help=f"W&B project to query (default: {WANDB_PROJECT})."
    )
    parser.add_argument(
        "--use-merged", action="store_true",
        help=(
            "Evaluate using pre-merged models instead of LoRA adapters. "
            "Requires ~16GB disk per model. "
            "Merged models must already exist at the derived path (<adapter>_merged). "
            "Run scripts/merge_adapter.py first if they don't."
        ),
    )
    args = parser.parse_args()

    print_header(
        "SFT Eval — Best Adapter Selection",
        f"Querying W&B project: {args.project}"
        + (" [merged-model mode]" if args.use_merged else ""),
    )

    gen_adapter, gen_f1 = fetch_best_adapter(args.project, classification_only=False)
    clf_adapter, clf_f1 = fetch_best_adapter(args.project, classification_only=True)

    print("\n" + "=" * 60)
    print("Selected adapters:")
    if gen_adapter:
        print(f"  generation:     {gen_adapter}  (val_harm_f1={gen_f1:.4f})")
    else:
        print("  generation:     NOT FOUND")
    if clf_adapter:
        print(f"  classification: {clf_adapter}  (val_harm_f1={clf_f1:.4f})")
    else:
        print("  classification: NOT FOUND")
    print("=" * 60)

    if gen_adapter is None and clf_adapter is None:
        print("\nNo adapters found. Ensure sweeps are complete and logged to W&B.")
        return

    if args.use_merged:
        # Derive merged model paths and check they exist
        gen_merged = _merged_path_for(gen_adapter) if gen_adapter else None
        clf_merged = _merged_path_for(clf_adapter) if clf_adapter else None

        print("\nMerged model paths:")
        missing = []
        for label, adapter, merged in [
            ("generation", gen_adapter, gen_merged),
            ("classification", clf_adapter, clf_merged),
        ]:
            if merged is None:
                continue
            exists = Path(merged).exists()
            status = "✓ exists" if exists else "✗ MISSING"
            print(f"  {label}: {merged}  [{status}]")
            if not exists:
                missing.append((label, adapter, merged))

        if missing:
            print("\nSome merged models are missing. Run the following to create them:")
            for label, adapter, merged in missing:
                print(f"  python scripts/merge_adapter.py \\")
                print(f"      --base-model {BASE_MODEL} \\")
                print(f"      --adapter {adapter} \\")
                print(f"      --output {merged}")
            if not args.dry_run:
                print("\nAborting submission until merged models are ready.")
                return

    if args.dry_run:
        print("\n--dry-run: not submitting job.")
        return

    create_logs_dir()

    if args.use_merged:
        # Submit one job per merged model (each loads a separate vLLM instance)
        submitted = []
        for label, adapter, condition, merged in [
            ("generation", gen_adapter, "finetuned_generation", gen_merged if gen_adapter else None),
            ("classification", clf_adapter, "finetuned_classification", clf_merged if clf_adapter else None),
        ]:
            if merged is None:
                continue
            export_vars = {
                "MERGED_MODEL_PATH": merged,
                "CONDITION": condition,
            }
            try:
                job_id = submit_sbatch("scripts/hpc/sft_eval_merged_template.sh", export_vars)
                print(f"\n✓ Submitted {label} merged-model eval job: {job_id}")
                print(f"  Monitor: squeue -j {job_id}")
                print(f"  Logs:    logs/slurm/sft_eval_merged-{job_id}.out")
                submitted.append(job_id)
            except subprocess.CalledProcessError as e:
                print(f"\n✗ Submission failed for {label}: {e.stderr}")
    else:
        export_vars = {
            "GENERATION_ADAPTER":     gen_adapter or "MISSING",
            "CLASSIFICATION_ADAPTER": clf_adapter or "MISSING",
        }
        try:
            job_id = submit_sbatch("scripts/hpc/sft_eval_template.sh", export_vars)
            print(f"\n✓ Submitted eval job: {job_id}")
            print(f"  Monitor: squeue -j {job_id}")
            print(f"  Logs:    logs/slurm/sft_eval-{job_id}.out")
        except subprocess.CalledProcessError as e:
            print(f"\n✗ Submission failed: {e.stderr}")


if __name__ == "__main__":
    main()
