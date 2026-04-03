#!/usr/bin/env python3
"""
Submit all preference-learning conditions on SLURM with dependencies.

Submitted stages:
  1) hard_dpo      (DPO on hard-only annotated pairs)
  2) judge_dpo     (hard + LLM-judge intent rejecteds)
  3) augment_prep  (WildGuard easy filtering + merge_dpo_pairs pre-steps)
  4) sft_aug       (retrain SFT with sft_examples.jsonl)
  5) dpo_aug       (DPO from augmented merged pairs using augmented SFT init)

Default dependency chain:
  hard_dpo -> augment_prep -> sft_aug -> dpo_aug
  judge_dpo runs in parallel (independent branch).

Usage:
  python scripts/hpc/submit_preference_conditions.py
  python scripts/hpc/submit_preference_conditions.py --dry-run
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path

from slurm_utils import create_logs_dir, print_header, print_job_summary


TEMPLATE = "scripts/hpc/preference_conditions_template.sh"


def _export_arg(values: dict[str, str]) -> str:
    return ",".join(f"{k}={v}" for k, v in values.items())


def submit_stage(
    stage: str,
    export_vars: dict[str, str],
    template: str,
    sbatch_opts: list[str] | None = None,
    dependency: str | None = None,
    dry_run: bool = False,
) -> str:
    env = {**export_vars, "STAGE": stage}
    cmd = ["sbatch"]
    if sbatch_opts:
        cmd.extend(sbatch_opts)
    if dependency:
        cmd.append(f"--dependency=afterok:{dependency}")
    cmd.append(f"--export={_export_arg(env)}")
    cmd.append(template)

    if dry_run:
        print("DRY RUN:", " ".join(shlex.quote(c) for c in cmd))
        return f"DRYRUN_{stage}"

    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return result.stdout.strip().split()[-1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--dry-run", action="store_true", help="Print sbatch commands only.")
    p.add_argument("--template", default=TEMPLATE, help="SLURM template script path.")
    p.add_argument("--start-from", default=None,
                   choices=["hard_dpo", "judge_dpo", "augment_prep", "sft_aug", "dpo_aug"],
                   help="Skip all stages before this one (useful for resuming after a failure).")

    p.add_argument("--base-model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--base-sft-adapter", default="trained_models/causal/hyperparam_sweep/lr_5e-05_e_5_adapter")
    p.add_argument("--judge-model", default="google/gemma-3-27b-it")
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--seed", type=int, default=22)
    p.add_argument("--force", action="store_true",
                   help="Pass --force to run_preference_pipeline.py stages.")
    p.add_argument("--force-from", type=int, default=None,
                   help="Pass --force-from N to run_preference_pipeline.py stages.")
    p.add_argument("--force-augment", action="store_true",
                   help="Pass --force-augment to augmentation stages.")
    p.add_argument("--all-gpushort", action="store_true",
                   help="Run all stages on gpushort (4h max; may timeout for long training stages).")

    # Lean default resources (tuned for faster scheduling)
    p.add_argument("--short-partition", default="gpushort")
    p.add_argument("--long-partition", default="gpumedium")
    p.add_argument("--short-time", default="04:00:00")
    p.add_argument("--long-time", default="10:00:00")
    p.add_argument("--short-mem", default="32G")
    p.add_argument("--long-mem", default="48G")
    p.add_argument("--short-cpus", type=int, default=4)
    p.add_argument("--long-cpus", type=int, default=6)

    p.add_argument("--wandb-project", default="intention-jailbreak")
    p.add_argument("--wandb-project-sft", default="sft-hyperparam-sweep")

    p.add_argument("--hard-pairs-dir", default="data/dpo_pairs/hard")
    p.add_argument("--hard-balanced-dir", default="data/dpo_pairs/hard_balanced")
    p.add_argument("--hard-dpo-output", default="trained_models/causal/dpo-hard")
    p.add_argument("--sft-pred-dir-hard", default="data/predictions/sft_hard")

    p.add_argument("--judge-pairs-dir", default="data/dpo_pairs/judge")
    p.add_argument("--judge-balanced-dir", default="data/dpo_pairs/judge_balanced")
    p.add_argument("--judge-dpo-output", default="trained_models/causal/dpo-judge")
    p.add_argument("--sft-pred-dir-judge", default="data/predictions/sft_judge")

    p.add_argument("--wildguard-candidates", default="data/wildguard_easy/candidates.jsonl")
    p.add_argument("--wildguard-model-path", default="models/modernbert-ensemble/final_model")
    p.add_argument("--wildguard-pairs-dir", default="data/wildguard_easy/dpo_pairs")
    p.add_argument("--sft-examples-output", default="data/wildguard_easy/sft_examples.jsonl")
    p.add_argument("--augmented-merged-pairs-dir", default="data/dpo_pairs/augmented")

    p.add_argument("--aug-sft-model-save-dir", default="models/sft/llm_sweep/augmented_sft")
    p.add_argument("--aug-sft-output-dir", default="data/train_results/sft_augmented")
    p.add_argument("--aug-sft-logs-dir", default="logs/sft_augmented")
    p.add_argument("--aug-sft-pred-dir", default="data/predictions/sft_augmented")
    p.add_argument("--aug-sft-wandb-name", default="sft-augmented")

    p.add_argument("--aug-balanced-dir", default="data/dpo_pairs/augmented_balanced")
    p.add_argument("--aug-dpo-output", default="trained_models/causal/dpo-augmented")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    template_path = Path(args.template)
    if not template_path.exists():
        raise FileNotFoundError(f"Template not found: {template_path}")

    print_header("Preference Conditions — SLURM Submission")
    if not args.dry_run:
        create_logs_dir()

    export_vars = {
        "PROJECT_ROOT": str(Path.cwd()),
        "BASE_MODEL": args.base_model,
        "BASE_SFT_ADAPTER": args.base_sft_adapter,
        "JUDGE_MODEL": args.judge_model,
        "MAX_MODEL_LEN": str(args.max_model_len),
        "SEED": str(args.seed),
        "FORCE": "1" if args.force else "0",
        "FORCE_FROM": "" if args.force_from is None else str(args.force_from),
        "FORCE_AUGMENT": "1" if args.force_augment else "0",
        "WANDB_PROJECT": args.wandb_project,
        "WANDB_PROJECT_SFT": args.wandb_project_sft,
        "HARD_PAIRS_DIR": args.hard_pairs_dir,
        "HARD_BALANCED_DIR": args.hard_balanced_dir,
        "HARD_DPO_OUTPUT": args.hard_dpo_output,
        "SFT_PRED_DIR_HARD": args.sft_pred_dir_hard,
        "JUDGE_PAIRS_DIR": args.judge_pairs_dir,
        "JUDGE_BALANCED_DIR": args.judge_balanced_dir,
        "JUDGE_DPO_OUTPUT": args.judge_dpo_output,
        "SFT_PRED_DIR_JUDGE": args.sft_pred_dir_judge,
        "ANNOTATED_PAIRS_DIR": args.hard_pairs_dir,
        "WILDGUARD_CANDIDATES": args.wildguard_candidates,
        "WILDGUARD_MODEL_PATH": args.wildguard_model_path,
        "WILDGUARD_PAIRS_DIR": args.wildguard_pairs_dir,
        "SFT_EXAMPLES_OUTPUT": args.sft_examples_output,
        "AUGMENTED_MERGED_PAIRS_DIR": args.augmented_merged_pairs_dir,
        "AUG_SFT_MODEL_SAVE_DIR": args.aug_sft_model_save_dir,
        "AUG_SFT_ADAPTER_PATH": f"{args.aug_sft_model_save_dir}_adapter",
        "AUG_SFT_OUTPUT_DIR": args.aug_sft_output_dir,
        "AUG_SFT_LOGS_DIR": args.aug_sft_logs_dir,
        "AUG_SFT_PRED_DIR": args.aug_sft_pred_dir,
        "AUG_SFT_WANDB_NAME": args.aug_sft_wandb_name,
        "AUG_BALANCED_DIR": args.aug_balanced_dir,
        "AUG_DPO_OUTPUT": args.aug_dpo_output,
    }

    jobs: list[tuple[str, str]] = []
    submitted_job_ids: dict[str, str] = {}

    stage_order = ["hard_dpo", "judge_dpo", "augment_prep", "sft_aug", "dpo_aug"]
    base_dependencies = {
        "hard_dpo": None,
        "judge_dpo": None,
        "augment_prep": "hard_dpo",
        "sft_aug": "augment_prep",
        "dpo_aug": "sft_aug",
    }

    if args.start_from:
        start_idx = stage_order.index(args.start_from)
        stages_to_submit = stage_order[start_idx:]
    else:
        stages_to_submit = stage_order

    def stage_sbatch_opts(stage: str) -> list[str]:
        short = [
            f"--partition={args.short_partition}",
            f"--time={args.short_time}",
            f"--mem={args.short_mem}",
            f"--cpus-per-task={args.short_cpus}",
            "--gpus-per-node=a100:1",
        ]
        long = [
            f"--partition={args.long_partition}",
            f"--time={args.long_time}",
            f"--mem={args.long_mem}",
            f"--cpus-per-task={args.long_cpus}",
            "--gpus-per-node=a100:1",
        ]
        if args.all_gpushort:
            return short
        if stage in {"hard_dpo", "augment_prep"}:
            return short
        return long

    for stage in stages_to_submit:
        dep_stage = base_dependencies[stage]
        dep_job_id = submitted_job_ids.get(dep_stage) if dep_stage else None
        job_id = submit_stage(
            stage,
            export_vars,
            args.template,
            sbatch_opts=stage_sbatch_opts(stage),
            dependency=dep_job_id,
            dry_run=args.dry_run,
        )
        submitted_job_ids[stage] = job_id
        jobs.append((stage, job_id))

    print("\nSubmitted stages:")
    for stage, job_id in jobs:
        print(f"  {stage:12s} -> {job_id}")

    if not args.dry_run:
        print_job_summary(jobs)


if __name__ == "__main__":
    main()
