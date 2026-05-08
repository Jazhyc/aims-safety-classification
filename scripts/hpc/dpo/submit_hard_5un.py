#!/usr/bin/env python3
"""
Submit reproducible 5-run hard-DPO protocol on SLURM.

Per run tag (default: a,b,c,d,e), this submits:
  1) canonical   (T=0.8 sample generation)
  2) hard_dpo    (train with checkpoint saving + best OOD checkpoint selection)
  3) ood_eval    (OOD validation for the selected adapter)

All stages for a run are chained via dependencies.
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

    # Run set
    p.add_argument("--exp-prefix", default="llama31-8b-k10")
    p.add_argument("--tags", nargs="+", default=["a", "b", "c", "d", "e"])

    # Model/hparams
    p.add_argument("--base-model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--base-sft-adapter", default="Jazhyc/llama-3.1-8b-sft-generation")
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

    # Checkpoint/OOD selection
    p.add_argument("--save-strategy", choices=["no", "epoch", "steps"], default="epoch")
    p.add_argument("--save-total-limit", type=int, default=3)
    p.add_argument("--save-steps", type=int, default=200)
    p.add_argument("--select-best-ood-checkpoint", action="store_true", default=True)

    # Paths
    p.add_argument("--project-root", default=str(Path.cwd()))
    p.add_argument("--wandb-project", default="intention-jailbreak")
    p.add_argument("--ood-output-base", default="data/safety_experiment/ood_validation/dpo_hard_5run")

    # SLURM resources
    p.add_argument("--partition", default="gpushort")
    p.add_argument("--gpu", default="rtx_pro_6000:1")
    p.add_argument("--gen-time", default="02:00:00")
    p.add_argument("--train-time", default="04:00:00")
    p.add_argument("--eval-time", default="02:00:00")
    p.add_argument("--mem", default="32G")
    p.add_argument("--cpus", type=int, default=4)

    # Controls
    p.add_argument("--skip-ood-eval", action="store_true")
    p.add_argument("--dry-run", action="store_true")

    return p.parse_args()


def sbatch_opts(args: argparse.Namespace, stage: str, log_name: str) -> list[str]:
    if stage == "canonical":
        time = args.gen_time
    elif stage == "hard_dpo":
        time = args.train_time
    else:
        time = args.eval_time

    return [
        f"--partition={args.partition}",
        f"--time={time}",
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
        "Hard DPO 5-Run Submission",
        f"tags={','.join(args.tags)}" + (" [DRY RUN]" if args.dry_run else ""),
    )

    if not args.dry_run:
        create_logs_dir()

    jobs: list[tuple[str, str]] = []

    for tag in args.tags:
        exp = f"{args.exp_prefix}-{tag}"
        model_out = f"trained_models/causal/{exp}-hard-b{args.dpo_beta}-e{args.epochs}-s{args.seed}"
        adapter_out = f"{model_out}_adapter"
        canonical_samples = f"data/dpo_pairs/{exp}-canonical-t0.8/parsed_samples.jsonl"

        common = {
            "PROJECT_ROOT": args.project_root,
            "BASE_MODEL": args.base_model,
            "BASE_SFT_ADAPTER": args.base_sft_adapter,
            "SEED": str(args.seed),
            "TEMPERATURE": str(args.temperature),
            "K_SAMPLES": str(args.k_samples),
            "MAX_MODEL_LEN": str(args.max_model_len),
            "EPOCHS": str(args.epochs),
            "LEARNING_RATE": str(args.learning_rate),
            "BATCH_SIZE": str(args.batch_size),
            "GRAD_ACCUM": str(args.grad_accum),
            "DPO_BETA": str(args.dpo_beta),
            "ATTN_IMPL": args.attn_implementation,
            "WANDB_PROJECT": args.wandb_project,
            "CANONICAL_SAMPLES": canonical_samples,
            "DPO_SAVE_STRATEGY": args.save_strategy,
            "DPO_SAVE_TOTAL_LIMIT": str(args.save_total_limit),
            "DPO_SAVE_STEPS": str(args.save_steps),
            "SELECT_BEST_OOD_CHECKPOINT": "1" if args.select_best_ood_checkpoint else "0",
        }

        # 1) canonical
        gen_env = {
            **common,
        }
        gen_job = submit_stage(
            stage="canonical",
            export_vars=gen_env,
            sbatch_opts=sbatch_opts(args, "canonical", f"{exp}-gen"),
            dependency=None,
            dry_run=args.dry_run,
        )
        jobs.append((f"{exp}:canonical", gen_job))

        # 2) hard_dpo
        hard_env = {
            **common,
            "HARD_PAIRS_DIR": f"data/dpo_pairs/{exp}-hard",
            "HARD_BALANCED_DIR": f"data/dpo_pairs/{exp}-hard_balanced",
            "HARD_DPO_OUTPUT": model_out,
        }
        hard_job = submit_stage(
            stage="hard_dpo",
            export_vars=hard_env,
            sbatch_opts=sbatch_opts(args, "hard_dpo", f"{exp}-hard"),
            dependency=gen_job,
            dry_run=args.dry_run,
        )
        jobs.append((f"{exp}:hard_dpo", hard_job))

        # 3) OOD eval (optional)
        if not args.skip_ood_eval:
            ood_env = {
                **common,
                "OOD_EVAL_ADAPTER": adapter_out,
                "OOD_EVAL_OUTPUT": args.ood_output_base,
            }
            ood_job = submit_stage(
                stage="ood_eval",
                export_vars=ood_env,
                sbatch_opts=sbatch_opts(args, "ood_eval", f"{exp}-ood"),
                dependency=hard_job,
                dry_run=args.dry_run,
            )
            jobs.append((f"{exp}:ood_eval", ood_job))

    print("\nSubmitted jobs:")
    for label, jid in jobs:
        print(f"  {label:40s} -> {jid}")

    if not args.dry_run:
        print_job_summary(jobs)


if __name__ == "__main__":
    main()
