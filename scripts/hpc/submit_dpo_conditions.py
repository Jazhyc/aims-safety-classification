#!/usr/bin/env python3
"""
Submit DPO condition stages on SLURM with dependencies.

Stages:
  hard_dpo      — DPO on hard-only annotated pairs
  judge_dpo     — DPO with LLM-judge intent rejecteds (Gemma 27B)
  augment_prep  — WildGuard easy filtering + DPO pair merge (pre-steps A/B/C)
  sft_aug       — Retrain SFT with augmented data
  dpo_aug       — DPO on augmented merged pairs using augmented SFT init

Dependency chain:
  hard_dpo -> augment_prep -> sft_aug -> dpo_aug
  judge_dpo runs independently.

Usage:
  python scripts/hpc/submit_dpo_conditions.py
  python scripts/hpc/submit_dpo_conditions.py --dry-run
  python scripts/hpc/submit_dpo_conditions.py --start-from judge_dpo --judge-from-samples --judge-epochs 1 --judge-beta 0.3
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path

from slurm_utils import create_logs_dir, print_header, print_job_summary


TEMPLATE = "scripts/hpc/dpo_conditions_template.sh"

STAGE_ORDER = ["hard_dpo", "judge_dpo", "augment_prep", "sft_aug", "dpo_aug"]

DEPENDENCIES = {
    "hard_dpo":     None,
    "judge_dpo":    None,
    "augment_prep": "hard_dpo",
    "sft_aug":      "augment_prep",
    "dpo_aug":      "sft_aug",
}

# Stages that finish quickly (pair gen / data prep, no full DPO training loop)
SHORT_STAGES = {"hard_dpo", "augment_prep"}


def submit_stage(
    stage: str,
    export_vars: dict[str, str],
    sbatch_opts: list[str],
    dependency: str | None = None,
    dry_run: bool = False,
) -> str:
    env = {**export_vars, "STAGE": stage}
    export_str = ",".join(f"{k}={v}" for k, v in env.items())
    cmd = ["sbatch", *sbatch_opts]
    if dependency:
        cmd.append(f"--dependency=afterok:{dependency}")
    cmd += [f"--export={export_str}", TEMPLATE]

    if dry_run:
        print("DRY RUN:", " ".join(shlex.quote(c) for c in cmd))
        return f"DRYRUN_{stage}"

    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return result.stdout.strip().split()[-1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # ── Pipeline control ───────────────────────────────────────────────────
    p.add_argument("--dry-run", action="store_true",
                   help="Print sbatch commands without submitting.")
    p.add_argument("--start-from", choices=STAGE_ORDER, default=None,
                   help="Skip all stages before this one (resume after failure).")
    p.add_argument("--force", action="store_true",
                   help="Re-run all pipeline steps, ignoring cached outputs.")
    p.add_argument("--force-from", type=int, default=None, metavar="N",
                   help="Re-run from pipeline step N onward.")
    p.add_argument("--force-augment", action="store_true",
                   help="Re-run augmentation pre-steps A/B/C even if cached.")

    # ── Model ──────────────────────────────────────────────────────────────
    p.add_argument("--base-model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--base-sft-adapter",
                   default="trained_models/causal/hyperparam_sweep/lr_5e-05_e_5_adapter")
    p.add_argument("--judge-model", default="google/gemma-3-27b-it")
    p.add_argument("--seed", type=int, default=22)

    # ── Judge-specific overrides ────────────────────────────────────────────
    p.add_argument("--judge-from-samples", action="store_true",
                   help="Skip sample generation for judge_dpo; reuse cached parsed_samples.jsonl.")
    p.add_argument("--judge-epochs", type=int, default=None,
                   help="Override DPO epochs for judge_dpo only.")
    p.add_argument("--judge-beta", type=float, default=None,
                   help="Override DPO beta for judge_dpo only.")

    # ── SLURM resources ────────────────────────────────────────────────────
    p.add_argument("--short-partition", default="gpushort")
    p.add_argument("--long-partition", default="gpumedium")
    p.add_argument("--short-time", default="04:00:00")
    p.add_argument("--long-time", default="10:00:00")
    p.add_argument("--short-mem", default="32G")
    p.add_argument("--long-mem", default="48G")
    p.add_argument("--short-cpus", type=int, default=2)
    p.add_argument("--long-cpus", type=int, default=4)
    p.add_argument("--all-gpushort", action="store_true",
                   help="Force all stages onto gpushort (may timeout for training stages).")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not Path(TEMPLATE).exists():
        raise FileNotFoundError(f"Template not found: {TEMPLATE}")

    print_header("DPO Conditions — SLURM Submission")
    if not args.dry_run:
        create_logs_dir()

    # Variables that the bash template cannot sensibly default (model IDs, seeds,
    # pipeline control flags). All path defaults live in the template itself.
    export_vars = {
        "PROJECT_ROOT":      str(Path.cwd()),
        "BASE_MODEL":        args.base_model,
        "BASE_SFT_ADAPTER":  args.base_sft_adapter,
        "JUDGE_MODEL":       args.judge_model,
        "SEED":              str(args.seed),
        "FORCE":             "1" if args.force else "0",
        "FORCE_FROM":        "" if args.force_from is None else str(args.force_from),
        "FORCE_AUGMENT":     "1" if args.force_augment else "0",
        "JUDGE_FROM_SAMPLES": "1" if args.judge_from_samples else "",
        "JUDGE_EPOCHS":      "" if args.judge_epochs is None else str(args.judge_epochs),
        "JUDGE_BETA":        "" if args.judge_beta is None else str(args.judge_beta),
    }

    def sbatch_opts(stage: str) -> list[str]:
        use_short = args.all_gpushort or stage in SHORT_STAGES
        partition = args.short_partition if use_short else args.long_partition
        time      = args.short_time      if use_short else args.long_time
        mem       = args.short_mem       if use_short else args.long_mem
        cpus      = args.short_cpus      if use_short else args.long_cpus
        return [
            f"--partition={partition}",
            f"--time={time}",
            f"--mem={mem}",
            f"--cpus-per-task={cpus}",
            "--gpus-per-node=a100:1",
        ]

    stages = STAGE_ORDER[STAGE_ORDER.index(args.start_from):] if args.start_from else STAGE_ORDER

    submitted: dict[str, str] = {}
    jobs: list[tuple[str, str]] = []

    for stage in stages:
        dep_stage  = DEPENDENCIES[stage]
        dep_job_id = submitted.get(dep_stage) if dep_stage else None
        job_id = submit_stage(
            stage, export_vars, sbatch_opts(stage),
            dependency=dep_job_id, dry_run=args.dry_run,
        )
        submitted[stage] = job_id
        jobs.append((stage, job_id))

    print("\nSubmitted stages:")
    for stage, job_id in jobs:
        print(f"  {stage:14s} -> {job_id}")

    if not args.dry_run:
        print_job_summary(jobs)


if __name__ == "__main__":
    main()
