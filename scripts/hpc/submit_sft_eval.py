#!/usr/bin/env python3
"""
Select best SFT adapters from W&B and submit an eval job.

Queries the sft-hyperparam-sweep W&B project, finds the finished run with the
highest val_harm_f1 for each sweep type (generation and classification-only),
derives the adapter path from the run config, and submits a SLURM eval job.

Usage:
    python scripts/hpc/submit_sft_eval.py              # submit SLURM job
    python scripts/hpc/submit_sft_eval.py --dry-run    # print selected adapters only
"""

import argparse
import subprocess
from pathlib import Path

import wandb
from dotenv import load_dotenv

load_dotenv()

from slurm_utils import create_logs_dir, submit_sbatch, print_header

WANDB_PROJECT = "sft-hyperparam-sweep"


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

        # Filter by sweep type using summary flag logged by causal.py
        run_clf_only = bool(summary.get("classification_only", False))
        if run_clf_only != classification_only:
            continue

        f1 = summary.get("val_harm_f1")
        if f1 is None:
            print(f"  [skip] {run.name}: no val_harm_f1 in summary")
            continue

        adapter_path_str = summary.get("adapter_path")
        if not adapter_path_str:
            print(f"  [skip] {run.name}: no adapter_path in summary (re-run to populate)")
            continue

        if not Path(adapter_path_str).exists():
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
    args = parser.parse_args()

    print_header(
        "SFT Eval — Best Adapter Selection",
        f"Querying W&B project: {args.project}"
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

    if args.dry_run:
        print("\n--dry-run: not submitting job.")
        return

    create_logs_dir()

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
