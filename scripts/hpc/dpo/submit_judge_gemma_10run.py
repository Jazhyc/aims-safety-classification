#!/usr/bin/env python3
"""
Submit Gemma-judge DPO conditions for multiple canonical tags (e.g., a..j).

Per tag, this submits:
  1) judge_dpo            (hard + judge bad_match)
  2) judge_standalone_dpo (judge-only bad_match)
  3) judge_decent_dpo     (judge-only bad_match + decent_match)
  4) union_decent_dpo     (hard + judge bad_match + decent_match)

All stages use the tag-specific canonical samples:
  data/dpo_pairs/<exp-prefix>-<tag>-canonical-t0.8/parsed_samples.jsonl
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

    p.add_argument("--exp-prefix", default="llama31-8b-k10")
    p.add_argument("--tags", nargs="+", default=list("abcdefghij"))

    p.add_argument("--base-model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--base-sft-adapter", default="Jazhyc/llama-3.1-8b-sft-generation")
    p.add_argument("--judge-model", default="google/gemma-3-27b-it")
    p.add_argument("--judge-tensor-parallel", type=int, default=1)

    p.add_argument("--seed", type=int, default=22)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--k-samples", type=int, default=10)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--learning-rate", type=float, default=5e-5)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--dpo-beta", type=float, default=0.3)
    p.add_argument("--attn-implementation", default="sdpa")

    p.add_argument("--save-strategy", choices=["no", "epoch", "steps"], default="epoch")
    p.add_argument("--save-total-limit", type=int, default=3)
    p.add_argument("--save-steps", type=int, default=200)
    p.add_argument("--select-best-ood-checkpoint", action="store_true", default=True)
    p.add_argument("--ood-eval-gpu-memory-utilization", type=float, default=0.75)

    p.add_argument("--project-root", default=str(Path.cwd()))
    p.add_argument("--wandb-project", default="intention-jailbreak")

    p.add_argument("--partition", default="gpushort")
    p.add_argument("--gpu", default="rtx_pro_6000:1")
    p.add_argument("--time", default="04:00:00")
    p.add_argument("--mem", default="32G")
    p.add_argument("--cpus", type=int, default=4)

    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def sbatch_opts(args: argparse.Namespace, log_name: str) -> list[str]:
    return [
        f"--partition={args.partition}",
        f"--time={args.time}",
        f"--mem={args.mem}",
        f"--cpus-per-task={args.cpus}",
        f"--gpus-per-node={args.gpu}",
        f"--output=logs/slurm/{log_name}-%j.out",
    ]


def main() -> None:
    args = parse_args()
    if not Path(TEMPLATE).exists():
        raise FileNotFoundError(f"Template not found: {TEMPLATE}")

    print_header(
        "Gemma Judge DPO 10-Run Submission",
        f"tags={','.join(args.tags)}" + (" [DRY RUN]" if args.dry_run else ""),
    )

    if not args.dry_run:
        create_logs_dir()

    jobs: list[tuple[str, str]] = []

    for tag in args.tags:
        exp = f"{args.exp_prefix}-{tag}"
        canonical_samples = f"data/dpo_pairs/{exp}-canonical-t0.8/parsed_samples.jsonl"

        common = {
            "PROJECT_ROOT": args.project_root,
            "BASE_MODEL": args.base_model,
            "BASE_SFT_ADAPTER": args.base_sft_adapter,
            "JUDGE_MODEL": args.judge_model,
            "JUDGE_TENSOR_PARALLEL": str(args.judge_tensor_parallel),
            "SEED": str(args.seed),
            "TEMPERATURE": str(args.temperature),
            "K_SAMPLES": str(args.k_samples),
            "MAX_MODEL_LEN": str(args.max_model_len),
            "EPOCHS": str(args.epochs),
            "JUDGE_EPOCHS": str(args.epochs),
            "LEARNING_RATE": str(args.learning_rate),
            "BATCH_SIZE": str(args.batch_size),
            "GRAD_ACCUM": str(args.grad_accum),
            "DPO_BETA": str(args.dpo_beta),
            "JUDGE_BETA": str(args.dpo_beta),
            "ATTN_IMPL": args.attn_implementation,
            "WANDB_PROJECT": args.wandb_project,
            "CANONICAL_SAMPLES": canonical_samples,
            "DPO_SAVE_STRATEGY": args.save_strategy,
            "DPO_SAVE_TOTAL_LIMIT": str(args.save_total_limit),
            "DPO_SAVE_STEPS": str(args.save_steps),
            "SELECT_BEST_OOD_CHECKPOINT": "1" if args.select_best_ood_checkpoint else "0",
            "OOD_EVAL_GPU_MEMORY_UTILIZATION": str(args.ood_eval_gpu_memory_utilization),
        }

        # A) hard + judge bad_match
        judge_dpo_env = {
            **common,
            "JUDGE_PAIRS_DIR": f"data/dpo_pairs/{exp}-judge-gemma-bad",
            "JUDGE_BALANCED_DIR": f"data/dpo_pairs/{exp}-judge-gemma-bad_balanced",
            "JUDGE_DPO_OUTPUT": f"trained_models/causal/{exp}-judge-gemma-bad-b{args.dpo_beta}-e{args.epochs}-s{args.seed}",
        }
        jid_judge_dpo = submit_stage(
            stage="judge_dpo",
            export_vars=judge_dpo_env,
            sbatch_opts=sbatch_opts(args, f"{exp}-judge-gemma-bad"),
            dry_run=args.dry_run,
        )
        jobs.append((f"{exp}:judge_dpo", jid_judge_dpo))

        # B) judge-only bad_match
        standalone_env = {
            **common,
            "JUDGE_STANDALONE_PAIRS_DIR": f"data/dpo_pairs/{exp}-judge-gemma-standalone",
            "JUDGE_STANDALONE_BALANCED_DIR": f"data/dpo_pairs/{exp}-judge-gemma-standalone_balanced",
            "JUDGE_STANDALONE_DPO_OUTPUT": f"trained_models/causal/{exp}-judge-gemma-standalone-b{args.dpo_beta}-e{args.epochs}-s{args.seed}",
        }
        jid_standalone = submit_stage(
            stage="judge_standalone_dpo",
            export_vars=standalone_env,
            sbatch_opts=sbatch_opts(args, f"{exp}-judge-gemma-standalone"),
            dry_run=args.dry_run,
        )
        jobs.append((f"{exp}:judge_standalone_dpo", jid_standalone))

        # C) judge-only bad_match + decent_match (depends on standalone pairs)
        decent_env = {
            **common,
            "JUDGE_STANDALONE_PAIRS_DIR": f"data/dpo_pairs/{exp}-judge-gemma-standalone",
            "JUDGE_DECENT_BALANCED_DIR": f"data/dpo_pairs/{exp}-judge-gemma-decent_balanced",
            "JUDGE_DECENT_DPO_OUTPUT": f"trained_models/causal/{exp}-judge-gemma-decent-b{args.dpo_beta}-e{args.epochs}-s{args.seed}",
        }
        jid_decent = submit_stage(
            stage="judge_decent_dpo",
            export_vars=decent_env,
            sbatch_opts=sbatch_opts(args, f"{exp}-judge-gemma-decent"),
            dependency=jid_standalone,
            dry_run=args.dry_run,
        )
        jobs.append((f"{exp}:judge_decent_dpo", jid_decent))

        # D) hard + judge bad_match + decent_match (depends on judge_dpo pairs)
        union_env = {
            **common,
            "JUDGE_PAIRS_DIR": f"data/dpo_pairs/{exp}-judge-gemma-bad",
            "UNION_DECENT_BALANCED_DIR": f"data/dpo_pairs/{exp}-hard-plus-gemma-decent_balanced",
            "UNION_DECENT_DPO_OUTPUT": f"trained_models/causal/{exp}-hard-plus-gemma-decent-b{args.dpo_beta}-e{args.epochs}-s{args.seed}",
        }
        jid_union = submit_stage(
            stage="union_decent_dpo",
            export_vars=union_env,
            sbatch_opts=sbatch_opts(args, f"{exp}-hard-plus-gemma-decent"),
            dependency=jid_judge_dpo,
            dry_run=args.dry_run,
        )
        jobs.append((f"{exp}:union_decent_dpo", jid_union))

    print("\nSubmitted jobs:")
    for label, jid in jobs:
        print(f"  {label:45s} -> {jid}")

    if not args.dry_run:
        print_job_summary(jobs)


if __name__ == "__main__":
    main()
