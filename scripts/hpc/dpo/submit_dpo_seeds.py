"""
Submit DPO multi-seed runs for reproducibility analysis.

Seed 22 is the baseline run already completed (canonical at data/dpo_pairs/train_t0.8/).
This script submits 4 additional seeds, each with:
  - Fresh canonical T=0.8 generation
  - All DPO conditions (hard, judge, standalone, decent, union_decent, union)
  - Seed-specific output dirs so nothing collides

Usage (from project root, on Habrok login node):
    python scripts/hpc/submit_dpo_seeds.py
    python scripts/hpc/submit_dpo_seeds.py --seeds 42 123 456 789 --dry-run
"""

import argparse
import subprocess
import sys
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────────

TEMPLATE = "scripts/hpc/dpo_conditions_template.sh"
JUDGE_MODEL = "google/gemma-3-27b-it"
GPTOSS_JOB_ID = "28565108"  # already completed, used for union dependency

PARTITION_A    = "gpumedium"    # RTX Pro 6000 — A100s have incompatible CUDA driver
GPU_A          = "rtx_pro_6000:1"
PARTITION_J    = "gpumedium"    # RTX Pro 6000 for gemma 27B judge (96 GB VRAM)
GPU_J          = "rtx_pro_6000:1"

MEM_SMALL = "40GB"
MEM_LARGE = "64GB"
TIME_CAN  = "02:00:00"
TIME_HARD = "06:00:00"
TIME_JUDG = "08:00:00"
TIME_SMAL = "06:00:00"


# ── Helpers ──────────────────────────────────────────────────────────────────

def sbatch(job_name: str, output_log: str, time: str, mem: str,
           partition: str, gpu: str, exports: dict,
           dependency: str = None, dry_run: bool = False) -> str:
    """Submit a job and return the job ID string."""
    export_str = ",".join(f"{k}={v}" for k, v in exports.items())
    cmd = [
        "sbatch", "--parsable",
        f"--partition={partition}",
        f"--gpus-per-node={gpu}",
        f"--mem={mem}",
        f"--time={time}",
        f"--job-name={job_name}",
        f"--output={output_log}",
        f"--export=ALL,{export_str}",
    ]
    if dependency:
        cmd.append(f"--dependency={dependency}")
    cmd.append(TEMPLATE)

    print(f"  {'[DRY]' if dry_run else '[SUB]'} {job_name}")
    if dry_run:
        print(f"        {' '.join(cmd)}\n")
        return f"DRY_{job_name}"

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR submitting {job_name}:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)
    jid = result.stdout.strip()
    print(f"        → job {jid}")
    return jid


def submit_seed(seed: int, dry_run: bool = False) -> None:
    s = str(seed)
    print(f"\n{'='*60}")
    print(f"  Submitting seed {seed}")
    print(f"{'='*60}")

    # Seed-specific paths
    can_dir      = f"data/dpo_pairs/train_t0.8_s{s}"
    can_samples  = f"{can_dir}/parsed_samples.jsonl"

    hard_pairs   = f"data/dpo_pairs/hard_s{s}"
    hard_bal     = f"data/dpo_pairs/hard_balanced_s{s}"
    hard_out     = f"trained_models/causal/dpo-hard-s{s}"
    hard_pred    = f"data/predictions/sft_hard_s{s}"

    judge_pairs  = f"data/dpo_pairs/judge_s{s}"
    judge_bal    = f"data/dpo_pairs/judge_balanced_s{s}"
    judge_out    = f"trained_models/causal/dpo-judge-s{s}"
    judge_pred   = f"data/predictions/sft_judge_s{s}"

    sa_pairs     = f"data/dpo_pairs/judge_standalone_s{s}"
    sa_bal       = f"data/dpo_pairs/judge_standalone_balanced_s{s}"
    sa_out       = f"trained_models/causal/dpo-judge-standalone-s{s}"
    sa_pred      = f"data/predictions/sft_judge_standalone_s{s}"

    dec_bal      = f"data/dpo_pairs/judge_decent_balanced_s{s}"
    dec_out      = f"trained_models/causal/dpo-judge-decent-s{s}"
    dec_pred     = f"data/predictions/sft_judge_decent_s{s}"

    und_bal      = f"data/dpo_pairs/union_decent_balanced_s{s}"
    und_out      = f"trained_models/causal/dpo-union-decent-s{s}"
    und_pred     = f"data/predictions/sft_union_decent_s{s}"

    union_pairs  = f"data/dpo_pairs/judge_union_s{s}"
    union_bal    = f"data/dpo_pairs/judge_union_balanced_s{s}"
    union_out    = f"trained_models/causal/dpo-judge-union-s{s}"
    union_pred   = f"data/predictions/sft_judge_union_s{s}"

    logs = f"logs/slurm"

    # ── canonical ────────────────────────────────────────────────────────────
    jid_can = sbatch(
        job_name=f"can-s{s}", output_log=f"{logs}/can-s{s}-%j.out",
        time=TIME_CAN, mem=MEM_SMALL, partition=PARTITION_A, gpu=GPU_A,
        exports=dict(STAGE="canonical", SEED=s, CANONICAL_SAMPLES=can_samples),
        dry_run=dry_run,
    )

    dep_can = f"afterok:{jid_can}"

    # ── hard_dpo ─────────────────────────────────────────────────────────────
    sbatch(
        job_name=f"hard-s{s}", output_log=f"{logs}/hard-s{s}-%j.out",
        time=TIME_HARD, mem=MEM_SMALL, partition=PARTITION_A, gpu=GPU_A,
        exports=dict(
            STAGE="hard_dpo", SEED=s,
            CANONICAL_SAMPLES=can_samples,
            HARD_PAIRS_DIR=hard_pairs, HARD_BALANCED_DIR=hard_bal,
            HARD_DPO_OUTPUT=hard_out, SFT_PRED_DIR_HARD=hard_pred,
        ),
        dependency=dep_can, dry_run=dry_run,
    )

    # ── judge_dpo ────────────────────────────────────────────────────────────
    jid_judge = sbatch(
        job_name=f"judge-s{s}", output_log=f"{logs}/judge-s{s}-%j.out",
        time=TIME_JUDG, mem=MEM_LARGE, partition=PARTITION_J, gpu=GPU_J,
        exports=dict(
            STAGE="judge_dpo", SEED=s,
            JUDGE_MODEL=JUDGE_MODEL, JUDGE_TENSOR_PARALLEL=1,
            CANONICAL_SAMPLES=can_samples,
            JUDGE_PAIRS_DIR=judge_pairs, JUDGE_BALANCED_DIR=judge_bal,
            JUDGE_DPO_OUTPUT=judge_out, SFT_PRED_DIR_JUDGE=judge_pred,
        ),
        dependency=dep_can, dry_run=dry_run,
    )

    # ── judge_standalone_dpo ─────────────────────────────────────────────────
    jid_sa = sbatch(
        job_name=f"sa-s{s}", output_log=f"{logs}/sa-s{s}-%j.out",
        time=TIME_JUDG, mem=MEM_LARGE, partition=PARTITION_J, gpu=GPU_J,
        exports=dict(
            STAGE="judge_standalone_dpo", SEED=s,
            JUDGE_MODEL=JUDGE_MODEL, JUDGE_TENSOR_PARALLEL=1,
            CANONICAL_SAMPLES=can_samples,
            JUDGE_STANDALONE_PAIRS_DIR=sa_pairs,
            JUDGE_STANDALONE_BALANCED_DIR=sa_bal,
            JUDGE_STANDALONE_DPO_OUTPUT=sa_out,
            SFT_PRED_DIR_JUDGE_STANDALONE=sa_pred,
        ),
        dependency=dep_can, dry_run=dry_run,
    )

    # ── judge_decent_dpo (needs standalone done) ─────────────────────────────
    sbatch(
        job_name=f"dec-s{s}", output_log=f"{logs}/dec-s{s}-%j.out",
        time=TIME_SMAL, mem=MEM_SMALL, partition=PARTITION_A, gpu=GPU_A,
        exports=dict(
            STAGE="judge_decent_dpo", SEED=s,
            JUDGE_STANDALONE_PAIRS_DIR=sa_pairs,
            JUDGE_DECENT_BALANCED_DIR=dec_bal,
            JUDGE_DECENT_DPO_OUTPUT=dec_out,
            SFT_PRED_DIR_JUDGE_DECENT=dec_pred,
        ),
        dependency=f"afterok:{jid_sa}", dry_run=dry_run,
    )

    # ── union_decent_dpo (needs judge done) ──────────────────────────────────
    sbatch(
        job_name=f"und-s{s}", output_log=f"{logs}/und-s{s}-%j.out",
        time=TIME_SMAL, mem=MEM_SMALL, partition=PARTITION_A, gpu=GPU_A,
        exports=dict(
            STAGE="union_decent_dpo", SEED=s,
            JUDGE_PAIRS_DIR=judge_pairs,
            UNION_DECENT_BALANCED_DIR=und_bal,
            UNION_DECENT_DPO_OUTPUT=und_out,
            SFT_PRED_DIR_UNION_DECENT=und_pred,
        ),
        dependency=f"afterok:{jid_judge}", dry_run=dry_run,
    )

    # ── judge_union_dpo (needs judge done; gptoss already done) ──────────────
    # Note: gptoss verdicts are fixed (same for all seeds — judge_gptoss ran
    # from the seed-22 canonical, so union is only valid for seed 22).
    # For other seeds we skip union since the gptoss base doesn't match.
    if seed == 22:
        sbatch(
            job_name=f"union-s{s}", output_log=f"{logs}/union-s{s}-%j.out",
            time=TIME_SMAL, mem=MEM_SMALL, partition=PARTITION_A, gpu=GPU_A,
            exports=dict(
                STAGE="judge_union_dpo", SEED=s,
                JUDGE_PAIRS_DIR=judge_pairs,
                JUDGE_GPTOSS_PAIRS_DIR="data/dpo_pairs/judge_gptoss",
                JUDGE_UNION_PAIRS_DIR=union_pairs,
                JUDGE_UNION_BALANCED_DIR=union_bal,
                JUDGE_UNION_DPO_OUTPUT=union_out,
                SFT_PRED_DIR_JUDGE_UNION=union_pred,
            ),
            dependency=f"afterok:{jid_judge}", dry_run=dry_run,
        )
    else:
        print(f"  [SKIP] union-s{s} — gptoss verdicts only match seed 22")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789],
                   help="Seeds to submit (seed 22 is the existing run).")
    p.add_argument("--dry-run", action="store_true",
                   help="Print commands without submitting.")
    return p.parse_args()


def main():
    args = parse_args()
    Path("logs/slurm").mkdir(parents=True, exist_ok=True)

    print(f"Submitting {len(args.seeds)} seeds: {args.seeds}")
    print("Seed 22 (existing run) is NOT resubmitted.\n")

    for seed in args.seeds:
        if seed == 22:
            print("Skipping seed 22 — already completed.")
            continue
        submit_seed(seed, dry_run=args.dry_run)

    print("\nDone. Monitor with: watch -n 30 \"squeue -u $USER -o '%.10i %.15j %.10T %.8R'\"")


if __name__ == "__main__":
    main()
