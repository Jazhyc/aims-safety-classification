#!/usr/bin/env python3
"""
Submit Gemma 3 12B DPO stages on SLURM.

Uses Jeremias's SFT model (Jazhyc/gemma-3-12b-intent-jailbreak-classifier-sft)
as the starting point for DPO training. Gemma 3 12B is a ForConditionalGeneration
model so training uses _load_causal_lm (language tower extraction) and
_fix_lora_keys_for_vllm (key renaming for vLLM evaluation).

Model type detection:
  --no-lora   : GEMMA_SFT_ADAPTER is a merged full model — load directly for
                generation (no LoRARequest) and init a fresh LoRA for training.
  (default)   : GEMMA_SFT_ADAPTER is a LoRA adapter on top of GEMMA_BASE_MODEL.

Stages:
  gemma_canonical  — generate T=0.8 samples from Gemma SFT
  gemma_hard_dpo   — hard-mislabel DPO training + eval

Usage:
    python scripts/hpc/dpo/submit_gemma_dpo.py --dry-run
    python scripts/hpc/dpo/submit_gemma_dpo.py
    python scripts/hpc/dpo/submit_gemma_dpo.py --start-from gemma_hard_dpo
    python scripts/hpc/dpo/submit_gemma_dpo.py --no-lora
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

STAGE_ORDER = ["gemma_canonical", "gemma_hard_dpo"]
DEPENDENCIES: dict[str, str | None] = {
    "gemma_canonical": None,
    "gemma_hard_dpo":  "gemma_canonical",
}


def submit_stage(stage: str, export_vars: dict, sbatch_opts: list[str],
                 dependency: str | None = None, dry_run: bool = False) -> str:
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
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--start-from", choices=STAGE_ORDER, default=None,
                   help="Skip stages before this one.")

    # Gemma model
    p.add_argument("--gemma-base-model", default="google/gemma-3-12b-it",
                   help="Base model for Gemma DPO (used when SFT is a LoRA adapter).")
    p.add_argument("--gemma-sft-adapter",
                   default="Jazhyc/gemma-3-12b-intent-jailbreak-classifier-sft",
                   help="Jeremias's SFT model: HF repo ID of either the merged full model "
                        "or a LoRA adapter on --gemma-base-model.")
    p.add_argument("--no-lora", action="store_true",
                   help="Treat --gemma-sft-adapter as a merged model (no LoRA). "
                        "Sets GEMMA_NO_LORA=1 and GEMMA_INIT_NEW_LORA=1 in the template.")

    # Training
    p.add_argument("--seed", type=int, default=22)
    p.add_argument("--dpo-beta", type=float, default=0.3)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--eval-lora-rank", type=int, default=16,
                   help="LoRA rank passed to eval_safety_classifier (set 32 for rank-32 adapters).")
    p.add_argument("--name-prefix", default="gemma3-12b",
                   help="Prefix used in output directory names.")

    # SLURM — generation stage needs GPU for vLLM; training needs A100
    p.add_argument("--gen-partition", default="gpushort")
    p.add_argument("--gen-time", default="02:00:00")
    p.add_argument("--gen-gpu", default="a100:1")
    p.add_argument("--train-partition", default="gpumedium")
    p.add_argument("--train-time", default="04:00:00")
    p.add_argument("--train-gpu", default="a100:1")
    p.add_argument("--mem", default="32G")
    p.add_argument("--cpus", type=int, default=4)

    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not Path(TEMPLATE).exists():
        raise FileNotFoundError(f"Template not found: {TEMPLATE}")

    print_header("Gemma DPO — SLURM Submission")
    if not args.dry_run:
        create_logs_dir()

    beta_tag = f"b{args.dpo_beta}"
    epoch_tag = f"e{args.epochs}"
    seed_tag = f"s{args.seed}"
    export_vars = {
        "PROJECT_ROOT":          str(Path.cwd()),
        "GEMMA_BASE_MODEL":      args.gemma_base_model,
        "GEMMA_SFT_ADAPTER":     args.gemma_sft_adapter,
        "GEMMA_NO_LORA":         "1" if args.no_lora else "0",
        "GEMMA_INIT_NEW_LORA":   "1" if args.no_lora else "0",
        "SEED":                  str(args.seed),
        "DPO_BETA":              str(args.dpo_beta),
        "EPOCHS":                str(args.epochs),
        "EVAL_LORA_RANK":        str(args.eval_lora_rank),
        "GEMMA_HARD_DPO_OUTPUT": f"trained_models/causal/{args.name_prefix}-hard-dpo-{beta_tag}-{epoch_tag}-{seed_tag}",
    }

    def sbatch_opts(stage: str) -> list[str]:
        if stage == "gemma_canonical":
            partition, time, gpu = args.gen_partition, args.gen_time, args.gen_gpu
        else:
            partition, time, gpu = args.train_partition, args.train_time, args.train_gpu
        return [
            f"--partition={partition}",
            f"--time={time}",
            f"--mem={args.mem}",
            f"--cpus-per-task={args.cpus}",
            f"--gpus-per-node={gpu}",
        ]

    stages = STAGE_ORDER[STAGE_ORDER.index(args.start_from):] if args.start_from else STAGE_ORDER

    submitted: dict[str, str] = {}
    jobs: list[tuple[str, str]] = []

    for stage in stages:
        dep_stage  = DEPENDENCIES[stage]
        dep_job_id = submitted.get(dep_stage) if dep_stage else None
        job_id = submit_stage(stage, export_vars, sbatch_opts(stage),
                               dependency=dep_job_id, dry_run=args.dry_run)
        submitted[stage] = job_id
        jobs.append((stage, job_id))
        print(f"  {stage:<20s} -> {job_id}")

    if not args.dry_run:
        print_job_summary(jobs)


if __name__ == "__main__":
    main()
