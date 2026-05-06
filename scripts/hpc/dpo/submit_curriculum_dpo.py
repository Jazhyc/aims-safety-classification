#!/usr/bin/env python3
"""
Submit curriculum DPO training on SLURM.

Phase 1: train on hard balanced pairs (policy starts at SFT).
Phase 2: train on judge balanced pairs (policy starts at phase-1 DPO, reference = SFT).

Both phases run sequentially within a single SLURM job. Requires that the
balanced pair files from hard_dpo and judge_dpo stages already exist.

Usage:
    python scripts/hpc/dpo/submit_curriculum_dpo.py --dry-run
    python scripts/hpc/dpo/submit_curriculum_dpo.py
    python scripts/hpc/dpo/submit_curriculum_dpo.py --phase1-epochs 2 --phase2-epochs 1
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


def _model_tag(model_id: str) -> str:
    tag = model_id.split("/")[-1].lower()
    tag = tag.replace("meta-", "")
    tag = tag.replace("llama-3.1-8b-instruct", "llama31-8b")
    tag = tag.replace("llama-3-8b-instruct", "llama3-8b")
    tag = tag.replace("_", "-")
    return tag


def submit_stage(export_vars: dict, sbatch_opts: list[str],
                 dry_run: bool = False) -> str:
    env = {**export_vars, "STAGE": "curriculum_dpo"}
    export_str = ",".join(f"{k}={v}" for k, v in env.items())
    cmd = ["sbatch", *sbatch_opts, f"--export={export_str}", TEMPLATE]
    if dry_run:
        print("DRY RUN:", " ".join(shlex.quote(c) for c in cmd))
        return "DRYRUN_curriculum_dpo"
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return result.stdout.strip().split()[-1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--dry-run", action="store_true")

    # Pair sources (must already exist from hard_dpo / judge_dpo runs)
    p.add_argument("--hard-balanced-dir", default="data/dpo_pairs/hard_balanced",
                   help="Directory containing balanced hard pairs (from hard_dpo stage).")
    p.add_argument("--judge-balanced-dir", default="data/dpo_pairs/judge_balanced",
                   help="Directory containing balanced judge pairs (from judge_dpo stage).")

    # Model
    p.add_argument("--base-model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--base-sft-adapter", default="Jazhyc/llama-3.1-8b-sft-generation")
    p.add_argument("--seed", type=int, default=22)
    p.add_argument("--dpo-beta", type=float, default=0.3)
    p.add_argument("--name-prefix", default=None,
                   help="Optional prefix for output dirs. Defaults to a tag derived from --base-model.")

    # Curriculum epochs
    p.add_argument("--phase1-epochs", type=int, default=2,
                   help="Epochs for phase 1 (hard pairs).")
    p.add_argument("--phase2-epochs", type=int, default=1,
                   help="Epochs for phase 2 (judge pairs, policy = phase-1 DPO).")

    # Output paths — beta is included in the name so reruns with different beta don't clobber.
    p.add_argument("--phase1-output", default=None,
                   help="Output directory for phase-1 adapter "
                        "(default: trained_models/causal/dpo-curriculum-phase1-beta{beta}).")
    p.add_argument("--curriculum-output", default=None,
                   help="Final output directory for phase-2 (curriculum) adapter "
                        "(default: trained_models/causal/dpo-curriculum-beta{beta}).")

    # SLURM — two full training runs in one job, allow extra time
    p.add_argument("--partition", default="gpumedium")
    p.add_argument("--time", default="4:00:00",
                   help="Wall time for the combined two-phase training job.")
    p.add_argument("--mem", default="24G")
    p.add_argument("--cpus", type=int, default=2)
    p.add_argument("--gpu-type", default="a100")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not Path(TEMPLATE).exists():
        raise FileNotFoundError(f"Template not found: {TEMPLATE}")

    print_header("Curriculum DPO — SLURM Submission")
    if not args.dry_run:
        create_logs_dir()

    model_tag = args.name_prefix or _model_tag(args.base_model)
    beta_tag = f"b{args.dpo_beta}"
    seed_tag = f"s{args.seed}"
    p1_tag = f"p1e{args.phase1_epochs}"
    p2_tag = f"p2e{args.phase2_epochs}"
    phase1_output = args.phase1_output or f"trained_models/causal/{model_tag}-curriculum-phase1-{beta_tag}-{p1_tag}-{seed_tag}"
    curriculum_output = args.curriculum_output or f"trained_models/causal/{model_tag}-curriculum-{beta_tag}-{p1_tag}-{p2_tag}-{seed_tag}"

    export_vars = {
        "PROJECT_ROOT":              str(Path.cwd()),
        "BASE_MODEL":                args.base_model,
        "BASE_SFT_ADAPTER":          args.base_sft_adapter,
        "HARD_BALANCED_DIR":         args.hard_balanced_dir,
        "JUDGE_BALANCED_DIR":        args.judge_balanced_dir,
        "SEED":                      str(args.seed),
        "DPO_BETA":                  str(args.dpo_beta),
        "PHASE1_EPOCHS":             str(args.phase1_epochs),
        "PHASE2_EPOCHS":             str(args.phase2_epochs),
        "CURRICULUM_PHASE1_OUTPUT":  phase1_output,
        "CURRICULUM_OUTPUT":         curriculum_output,
    }

    sbatch_opts = [
        f"--partition={args.partition}",
        f"--time={args.time}",
        f"--mem={args.mem}",
        f"--cpus-per-task={args.cpus}",
        f"--gpus-per-node={args.gpu_type}:1",
    ]

    job_id = submit_stage(export_vars, sbatch_opts, dry_run=args.dry_run)
    jobs = [("curriculum_dpo", job_id)]
    print(f"  curriculum_dpo -> {job_id}")

    if not args.dry_run:
        print_job_summary(jobs)


if __name__ == "__main__":
    main()
