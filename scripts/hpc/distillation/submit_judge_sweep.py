#!/usr/bin/env python3
"""
Submit DPO condition hyperparameter sweep on SLURM.

Sweeps beta × epochs on fixed pairs for a given condition (judge or hard).
All runs are independent and run in parallel.

Usage:
  python scripts/hpc/submit_judge_sweep.py --condition judge
  python scripts/hpc/submit_judge_sweep.py --condition hard --unbalanced
  python scripts/hpc/submit_judge_sweep.py --condition judge --dry-run
  python scripts/hpc/submit_judge_sweep.py --condition hard --betas 0.3 0.5 --epochs 1 2 3
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from slurm_utils import create_logs_dir, print_header, print_job_summary

TEMPLATE = "scripts/hpc/distillation/judge_sweep_template.sh"


def submit_run(
    condition: str,
    beta: float,
    epochs: int,
    base_model: str,
    base_sft_adapter: str,
    unbalanced: bool,
    harmful_weight: float,
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
        "CONDITION":        condition,
        "SWEEP_BETA":       str(beta),
        "SWEEP_EPOCHS":     str(epochs),
        "SEED":             "22",
        "UNBALANCED":       "1" if unbalanced else "0",
        "HARMFUL_WEIGHT":   str(harmful_weight),
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
        return f"DRYRUN_{condition}_beta{beta}_e{epochs}"

    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return result.stdout.strip().split()[-1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--condition", choices=["judge", "hard"], required=True)
    p.add_argument("--betas",     nargs="+", type=float, default=[0.1, 0.3, 0.5])
    p.add_argument("--epochs",    nargs="+", type=int,   default=[1, 2])
    p.add_argument("--unbalanced", action="store_true",
                   help="Use unbalanced pairs with loss weighting (for hard condition).")
    p.add_argument("--harmful-weight", type=float, default=1.4)
    p.add_argument("--base-model",       default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--base-sft-adapter", default="trained_models/causal/hyperparam_sweep/lr_5e-05_e_5_adapter")
    p.add_argument("--partition", default="gpumedium")
    p.add_argument("--time",      default="08:00:00")
    p.add_argument("--mem",       default="24G")
    p.add_argument("--cpus",      type=int, default=2)
    p.add_argument("--dry-run",   action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not Path(TEMPLATE).exists():
        raise FileNotFoundError(f"Template not found: {TEMPLATE}")

    print_header(f"DPO-{args.condition.capitalize()} Sweep — SLURM Submission")
    if not args.dry_run:
        create_logs_dir()

    jobs: list[tuple[str, str]] = []
    for beta in args.betas:
        for epochs in args.epochs:
            run_name = f"{args.condition}_beta{beta}_e{epochs}"
            job_id = submit_run(
                condition=args.condition,
                beta=beta,
                epochs=epochs,
                base_model=args.base_model,
                base_sft_adapter=args.base_sft_adapter,
                unbalanced=args.unbalanced,
                harmful_weight=args.harmful_weight,
                partition=args.partition,
                time=args.time,
                mem=args.mem,
                cpus=args.cpus,
                dry_run=args.dry_run,
            )
            jobs.append((run_name, job_id))

    print("\nSubmitted runs:")
    for name, job_id in jobs:
        print(f"  {name:30s} -> {job_id}")

    if not args.dry_run:
        print_job_summary(jobs)


if __name__ == "__main__":
    main()
