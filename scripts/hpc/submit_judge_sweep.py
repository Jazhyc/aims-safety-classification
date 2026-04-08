#!/usr/bin/env python3
"""
Submit DPO-judge hyperparameter sweep on SLURM.

Sweeps beta × epochs on fixed judge pairs (data/dpo_pairs/judge/).
All runs are independent (no dependencies) so they can run in parallel.

Usage:
  python scripts/hpc/submit_judge_sweep.py
  python scripts/hpc/submit_judge_sweep.py --dry-run
  python scripts/hpc/submit_judge_sweep.py --betas 0.3 0.5 --epochs 1 2
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path

from slurm_utils import create_logs_dir, print_header, print_job_summary

TEMPLATE = "scripts/hpc/judge_sweep_template.sh"


def submit_run(
    beta: float,
    epochs: int,
    base_model: str,
    base_sft_adapter: str,
    partition: str,
    time: str,
    mem: str,
    cpus: int,
    dry_run: bool = False,
) -> str:
    export_vars = {
        "PROJECT_ROOT":     str(Path.cwd()),
        "BASE_MODEL":       base_model,
        "BASE_SFT_ADAPTER": base_sft_adapter,
        "JUDGE_BETA":       str(beta),
        "JUDGE_EPOCHS":     str(epochs),
        "SEED":             "22",
    }
    export_str = ",".join(f"{k}={v}" for k, v in export_vars.items())
    cmd = [
        "sbatch",
        f"--partition={partition}",
        f"--time={time}",
        f"--mem={mem}",
        f"--cpus-per-task={cpus}",
        "--gpus-per-node=a100:1",
        f"--export={export_str}",
        TEMPLATE,
    ]
    if dry_run:
        print("DRY RUN:", " ".join(shlex.quote(c) for c in cmd))
        return f"DRYRUN_beta{beta}_e{epochs}"

    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return result.stdout.strip().split()[-1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--betas",   nargs="+", type=float, default=[0.1, 0.3, 0.5])
    p.add_argument("--epochs",  nargs="+", type=int,   default=[1, 2])
    p.add_argument("--base-model",       default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--base-sft-adapter", default="trained_models/causal/hyperparam_sweep/lr_5e-05_e_5_adapter")
    p.add_argument("--partition", default="gpumedium")
    p.add_argument("--time",      default="08:00:00")
    p.add_argument("--mem",       default="48G")
    p.add_argument("--cpus",      type=int, default=4)
    p.add_argument("--dry-run",   action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not Path(TEMPLATE).exists():
        raise FileNotFoundError(f"Template not found: {TEMPLATE}")

    print_header("DPO-Judge Sweep — SLURM Submission")
    if not args.dry_run:
        create_logs_dir()

    jobs: list[tuple[str, str]] = []
    for beta in args.betas:
        for epochs in args.epochs:
            run_name = f"beta{beta}_e{epochs}"
            job_id = submit_run(
                beta=beta,
                epochs=epochs,
                base_model=args.base_model,
                base_sft_adapter=args.base_sft_adapter,
                partition=args.partition,
                time=args.time,
                mem=args.mem,
                cpus=args.cpus,
                dry_run=args.dry_run,
            )
            jobs.append((run_name, job_id))

    print("\nSubmitted runs:")
    for name, job_id in jobs:
        print(f"  {name:20s} -> {job_id}")

    if not args.dry_run:
        print_job_summary(jobs)


if __name__ == "__main__":
    main()
