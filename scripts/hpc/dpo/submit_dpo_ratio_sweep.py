#!/usr/bin/env python3
"""
Submit ratio-sweep DPO jobs on SLURM.

Each job mixes all hard pairs with RATIO fraction of judge pairs, balances,
trains DPO, and evaluates. Requires that pairs_wrong_only.jsonl (hard) and
pairs_judge_bad_only.jsonl (judge) already exist from prior runs.

Usage:
    python scripts/hpc/dpo/submit_dpo_ratio_sweep.py
    python scripts/hpc/dpo/submit_dpo_ratio_sweep.py --dry-run
    python scripts/hpc/dpo/submit_dpo_ratio_sweep.py --ratios 0.0 0.25 0.5 0.75 1.0
    python scripts/hpc/dpo/submit_dpo_ratio_sweep.py --hard-pairs-dir data/dpo_pairs/hard \\
        --judge-pairs-dir data/dpo_pairs/judge
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from slurm_utils import create_logs_dir, print_header, print_job_summary

TEMPLATE = "scripts/hpc/dpo/dpo_conditions_template.sh"

DEFAULT_RATIOS = [0.0, 0.25, 0.50, 0.75, 1.0]


def _model_tag(model_id: str) -> str:
    """Compact stable model tag for output directory names."""
    tag = model_id.split("/")[-1].lower()
    tag = tag.replace("meta-", "").replace("google/", "")
    tag = tag.replace("llama-3.1-8b-instruct", "llama31-8b")
    tag = tag.replace("llama-3-8b-instruct", "llama3-8b")
    tag = tag.replace("gemma-3-12b-it", "gemma3-12b")
    tag = tag.replace("_", "-")
    return tag


def submit_stage(export_vars: dict, sbatch_opts: list[str],
                 dry_run: bool = False) -> str:
    env = {**export_vars, "STAGE": "ratio_dpo"}
    export_str = ",".join(f"{k}={v}" for k, v in env.items())
    cmd = ["sbatch", *sbatch_opts, f"--export={export_str}", TEMPLATE]
    if dry_run:
        print("DRY RUN:", " ".join(shlex.quote(c) for c in cmd))
        return f"DRYRUN_ratio_{export_vars['RATIO_TAG']}"
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return result.stdout.strip().split()[-1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--ratios", nargs="+", type=float, default=DEFAULT_RATIOS,
                   metavar="R", help="Ratio values to sweep (fraction of judge pairs added).")
    p.add_argument("--dry-run", action="store_true")

    # Pair source paths
    p.add_argument("--hard-pairs-dir", default="data/dpo_pairs/hard",
                   help="Directory containing pairs_wrong_only.jsonl.")
    p.add_argument("--judge-pairs-dir", default="data/dpo_pairs/judge",
                   help="Directory containing pairs_judge_bad_only.jsonl.")

    # Model
    p.add_argument("--base-model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--base-sft-adapter", default="Jazhyc/llama-3.1-8b-sft-generation")
    p.add_argument("--seed", type=int, default=22)
    p.add_argument("--dpo-beta", type=float, default=0.3)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--name-prefix", default=None,
                   help="Optional prefix for output/data dirs. Defaults to a tag derived from --base-model.")

    # SLURM
    p.add_argument("--partition", default="gpumedium")
    p.add_argument("--time", default="04:00:00")
    p.add_argument("--mem", default="24G")
    p.add_argument("--cpus", type=int, default=2)
    p.add_argument("--gpu-type", default="a100")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not Path(TEMPLATE).exists():
        raise FileNotFoundError(f"Template not found: {TEMPLATE}")

    print_header("DPO Ratio Sweep — SLURM Submission")
    if not args.dry_run:
        create_logs_dir()

    hard_pairs_file  = f"{args.hard_pairs_dir}/pairs_wrong_only.jsonl"
    judge_pairs_file = f"{args.judge_pairs_dir}/pairs_judge_bad_only.jsonl"

    sbatch_opts = [
        f"--partition={args.partition}",
        f"--time={args.time}",
        f"--mem={args.mem}",
        f"--cpus-per-task={args.cpus}",
        f"--gpus-per-node={args.gpu_type}:1",
    ]

    model_tag = args.name_prefix or _model_tag(args.base_model)
    beta_tag = f"b{args.dpo_beta}"
    epoch_tag = f"e{args.epochs}"
    seed_tag = f"s{args.seed}"

    jobs: list[tuple[str, str]] = []
    for ratio in args.ratios:
        ratio_tag = f"{ratio:.2f}"
        export_vars = {
            "PROJECT_ROOT":      str(Path.cwd()),
            "BASE_MODEL":        args.base_model,
            "BASE_SFT_ADAPTER":  args.base_sft_adapter,
            "RATIO":             str(ratio),
            "RATIO_TAG":         ratio_tag,
            "HARD_PAIRS_FILE":   hard_pairs_file,
            "JUDGE_PAIRS_FILE":  judge_pairs_file,
            "SEED":              str(args.seed),
            "DPO_BETA":          str(args.dpo_beta),
            "EPOCHS":            str(args.epochs),
            "RATIO_PAIRS_DIR":   f"data/dpo_pairs/{model_tag}-ratio-r{ratio_tag}-{beta_tag}-{epoch_tag}-{seed_tag}",
            "RATIO_BALANCED_DIR": f"data/dpo_pairs/{model_tag}-ratio-r{ratio_tag}-{beta_tag}-{epoch_tag}-{seed_tag}_balanced",
            "RATIO_DPO_OUTPUT":  f"trained_models/causal/{model_tag}-ratio-r{ratio_tag}-{beta_tag}-{epoch_tag}-{seed_tag}",
        }
        job_id = submit_stage(export_vars, sbatch_opts, dry_run=args.dry_run)
        jobs.append((f"ratio={ratio_tag}", job_id))
        print(f"  ratio={ratio_tag:5s} -> {job_id}")

    if not args.dry_run:
        print_job_summary(jobs)


if __name__ == "__main__":
    main()
