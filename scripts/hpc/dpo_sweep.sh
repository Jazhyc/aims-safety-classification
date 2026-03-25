#!/bin/bash
#SBATCH --job-name=dpo-sweep
#SBATCH --time=00:50:00
#SBATCH --mem=16GB
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=a100:1
#SBATCH --output=logs/slurm/dpo-sweep-%A_%a.out
#SBATCH --error=logs/slurm/dpo-sweep-%A_%a.err

# ── DPO hyperparameter sweep (29 runs total) ─────────────────────────────────
#
# Submit with --array=<range>, e.g.:
#   sbatch --array=1       scripts/hpc/dpo_sweep.sh   # Group A: unbalanced ref
#   sbatch --array=2-5     scripts/hpc/dpo_sweep.sh   # Group B: balancing comparison
#   sbatch --array=6-13    scripts/hpc/dpo_sweep.sh   # Group C-us: β sweep, undersampling
#   sbatch --array=14-21   scripts/hpc/dpo_sweep.sh   # Group C-wl: β sweep, weighted loss
#   sbatch --array=22-25   scripts/hpc/dpo_sweep.sh   # Group D-us: epochs/ls, undersampling
#   sbatch --array=26-29   scripts/hpc/dpo_sweep.sh   # Group D-wl: epochs/ls, weighted loss
#   sbatch --array=6-29    scripts/hpc/dpo_sweep.sh   # Groups C+D (all)
#   sbatch --array=1-29    scripts/hpc/dpo_sweep.sh   # everything
#
# ── GROUP A: Unbalanced reference ────────────────────────────────────────────
# Run |  β   | loss    | lr    | e | ls   | hw  | pairs
#  1  | 0.50 | sigmoid | 5e-5  | 1 | 0.00 | 1.0 | unbal   (Phase 2 best, 1 epoch)
#
# ── GROUP B: Balancing method comparison ─────────────────────────────────────
# Run |  β   | loss    | lr    | e | ls   | hw  | pairs
#  2  | 0.50 | sigmoid | 5e-5  | 1 | 0.00 | 1.0 | bal     undersampling
#  3  | 0.50 | sigmoid | 5e-5  | 1 | 0.10 | 1.0 | bal     undersampling + label smoothing
#  4  | 0.50 | sigmoid | 5e-5  | 1 | 0.00 | 1.4 | unbal   weighted loss
#  5  | 0.50 | sigmoid | 5e-5  | 1 | 0.10 | 1.4 | unbal   weighted loss + label smoothing
#
# ── GROUP C-us: β sweep — undersampling ──────────────────────────────────────
# Run |  β   | loss    | lr    | e | ls   | hw  | pairs
#  6  | 0.10 | sigmoid | 5e-5  | 1 | 0.00 | 1.0 | bal
#  7  | 0.15 | sigmoid | 5e-5  | 1 | 0.00 | 1.0 | bal
#  8  | 0.20 | sigmoid | 5e-5  | 1 | 0.00 | 1.0 | bal
#  9  | 0.30 | sigmoid | 5e-5  | 1 | 0.00 | 1.0 | bal
# 10  | 0.40 | sigmoid | 5e-5  | 1 | 0.00 | 1.0 | bal
# 11  | 0.50 | sigmoid | 5e-5  | 1 | 0.00 | 1.0 | bal
# 12  | 0.60 | sigmoid | 5e-5  | 1 | 0.00 | 1.0 | bal
# 13  | 0.70 | sigmoid | 5e-5  | 1 | 0.00 | 1.0 | bal
#
# ── GROUP C-wl: β sweep — weighted loss ──────────────────────────────────────
# Run |  β   | loss    | lr    | e | ls   | hw  | pairs
# 14  | 0.10 | sigmoid | 5e-5  | 1 | 0.00 | 1.4 | unbal
# 15  | 0.15 | sigmoid | 5e-5  | 1 | 0.00 | 1.4 | unbal
# 16  | 0.20 | sigmoid | 5e-5  | 1 | 0.00 | 1.4 | unbal
# 17  | 0.30 | sigmoid | 5e-5  | 1 | 0.00 | 1.4 | unbal
# 18  | 0.40 | sigmoid | 5e-5  | 1 | 0.00 | 1.4 | unbal
# 19  | 0.50 | sigmoid | 5e-5  | 1 | 0.00 | 1.4 | unbal
# 20  | 0.60 | sigmoid | 5e-5  | 1 | 0.00 | 1.4 | unbal
# 21  | 0.70 | sigmoid | 5e-5  | 1 | 0.00 | 1.4 | unbal
#
# ── GROUP D-us: epochs + label smoothing — undersampling ─────────────────────
# Run |  β   | loss    | lr    | e | ls   | hw  | pairs
# 22  | 0.50 | sigmoid | 5e-5  | 2 | 0.00 | 1.0 | bal
# 23  | 0.50 | sigmoid | 5e-5  | 3 | 0.00 | 1.0 | bal
# 24  | 0.50 | sigmoid | 5e-5  | 1 | 0.05 | 1.0 | bal
# 25  | 0.50 | sigmoid | 5e-5  | 1 | 0.00 | 1.0 | bal     seed=42 stability check
#
# ── GROUP D-wl: epochs + label smoothing — weighted loss ─────────────────────
# Run |  β   | loss    | lr    | e | ls   | hw  | pairs
# 26  | 0.50 | sigmoid | 5e-5  | 2 | 0.00 | 1.4 | unbal
# 27  | 0.50 | sigmoid | 5e-5  | 3 | 0.00 | 1.4 | unbal
# 28  | 0.50 | sigmoid | 5e-5  | 1 | 0.05 | 1.4 | unbal
# 29  | 0.50 | sigmoid | 5e-5  | 1 | 0.00 | 1.4 | unbal   seed=42 stability check
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail
mkdir -p logs/slurm

module purge
module load CUDA/12.8.0
module load Python/3.12.3-GCCcore-13.3.0

cd "$HOME/repos/intent-gen"
source .venv/bin/activate
source ~/.wandb_secrets

PROJECTS_DIR="${PROJECTS_DIR:-/scratch/s4351495}"
export HF_HOME="${HF_HOME:-${PROJECTS_DIR}/huggingface_cache}"
SFT_ADAPTER="${SFT_ADAPTER:-${PROJECTS_DIR}/trained_models/causal/hyperparam_sweep/lr_5e-05_e_5_adapter}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=online
export VLLM_USE_V1=0

PAIRS_UNBAL="data/dpo_pairs/train_t0.8/dpo_pairs.jsonl"
PAIRS_BAL="data/dpo_pairs/train_t0.8_balanced/dpo_pairs.jsonl"

case $SLURM_ARRAY_TASK_ID in
  # ── Group A: unbalanced reference ──────────────────────────────────────────
  1)  BETA=0.5;  LOSS=sigmoid; LR=5e-5; EPOCHS=1; LS=0.0;  HW=1.0; PAIRS="$PAIRS_UNBAL"; SEED=22 ;;
  # ── Group B: balancing method comparison ───────────────────────────────────
  2)  BETA=0.5;  LOSS=sigmoid; LR=5e-5; EPOCHS=1; LS=0.0;  HW=1.0; PAIRS="$PAIRS_BAL";   SEED=22 ;;
  3)  BETA=0.5;  LOSS=sigmoid; LR=5e-5; EPOCHS=1; LS=0.1;  HW=1.0; PAIRS="$PAIRS_BAL";   SEED=22 ;;
  4)  BETA=0.5;  LOSS=sigmoid; LR=5e-5; EPOCHS=1; LS=0.0;  HW=1.4; PAIRS="$PAIRS_UNBAL"; SEED=22 ;;
  5)  BETA=0.5;  LOSS=sigmoid; LR=5e-5; EPOCHS=1; LS=0.1;  HW=1.4; PAIRS="$PAIRS_UNBAL"; SEED=22 ;;
  # ── Group C-us: β sweep, undersampling ─────────────────────────────────────
  6)  BETA=0.1;  LOSS=sigmoid; LR=5e-5; EPOCHS=1; LS=0.0;  HW=1.0; PAIRS="$PAIRS_BAL";   SEED=22 ;;
  7)  BETA=0.15; LOSS=sigmoid; LR=5e-5; EPOCHS=1; LS=0.0;  HW=1.0; PAIRS="$PAIRS_BAL";   SEED=22 ;;
  8)  BETA=0.2;  LOSS=sigmoid; LR=5e-5; EPOCHS=1; LS=0.0;  HW=1.0; PAIRS="$PAIRS_BAL";   SEED=22 ;;
  9)  BETA=0.3;  LOSS=sigmoid; LR=5e-5; EPOCHS=1; LS=0.0;  HW=1.0; PAIRS="$PAIRS_BAL";   SEED=22 ;;
  10) BETA=0.4;  LOSS=sigmoid; LR=5e-5; EPOCHS=1; LS=0.0;  HW=1.0; PAIRS="$PAIRS_BAL";   SEED=22 ;;
  11) BETA=0.5;  LOSS=sigmoid; LR=5e-5; EPOCHS=1; LS=0.0;  HW=1.0; PAIRS="$PAIRS_BAL";   SEED=22 ;;
  12) BETA=0.6;  LOSS=sigmoid; LR=5e-5; EPOCHS=1; LS=0.0;  HW=1.0; PAIRS="$PAIRS_BAL";   SEED=22 ;;
  13) BETA=0.7;  LOSS=sigmoid; LR=5e-5; EPOCHS=1; LS=0.0;  HW=1.0; PAIRS="$PAIRS_BAL";   SEED=22 ;;
  # ── Group C-wl: β sweep, weighted loss ─────────────────────────────────────
  14) BETA=0.1;  LOSS=sigmoid; LR=5e-5; EPOCHS=1; LS=0.0;  HW=1.4; PAIRS="$PAIRS_UNBAL"; SEED=22 ;;
  15) BETA=0.15; LOSS=sigmoid; LR=5e-5; EPOCHS=1; LS=0.0;  HW=1.4; PAIRS="$PAIRS_UNBAL"; SEED=22 ;;
  16) BETA=0.2;  LOSS=sigmoid; LR=5e-5; EPOCHS=1; LS=0.0;  HW=1.4; PAIRS="$PAIRS_UNBAL"; SEED=22 ;;
  17) BETA=0.3;  LOSS=sigmoid; LR=5e-5; EPOCHS=1; LS=0.0;  HW=1.4; PAIRS="$PAIRS_UNBAL"; SEED=22 ;;
  18) BETA=0.4;  LOSS=sigmoid; LR=5e-5; EPOCHS=1; LS=0.0;  HW=1.4; PAIRS="$PAIRS_UNBAL"; SEED=22 ;;
  19) BETA=0.5;  LOSS=sigmoid; LR=5e-5; EPOCHS=1; LS=0.0;  HW=1.4; PAIRS="$PAIRS_UNBAL"; SEED=22 ;;
  20) BETA=0.6;  LOSS=sigmoid; LR=5e-5; EPOCHS=1; LS=0.0;  HW=1.4; PAIRS="$PAIRS_UNBAL"; SEED=22 ;;
  21) BETA=0.7;  LOSS=sigmoid; LR=5e-5; EPOCHS=1; LS=0.0;  HW=1.4; PAIRS="$PAIRS_UNBAL"; SEED=22 ;;
  # ── Group D-us: epochs + label smoothing, undersampling ────────────────────
  22) BETA=0.5;  LOSS=sigmoid; LR=5e-5; EPOCHS=2; LS=0.0;  HW=1.0; PAIRS="$PAIRS_BAL";   SEED=22 ;;
  23) BETA=0.5;  LOSS=sigmoid; LR=5e-5; EPOCHS=3; LS=0.0;  HW=1.0; PAIRS="$PAIRS_BAL";   SEED=22 ;;
  24) BETA=0.5;  LOSS=sigmoid; LR=5e-5; EPOCHS=1; LS=0.05; HW=1.0; PAIRS="$PAIRS_BAL";   SEED=22 ;;
  25) BETA=0.5;  LOSS=sigmoid; LR=5e-5; EPOCHS=1; LS=0.0;  HW=1.0; PAIRS="$PAIRS_BAL";   SEED=42 ;;
  # ── Group D-wl: epochs + label smoothing, weighted loss ────────────────────
  26) BETA=0.5;  LOSS=sigmoid; LR=5e-5; EPOCHS=2; LS=0.0;  HW=1.4; PAIRS="$PAIRS_UNBAL"; SEED=22 ;;
  27) BETA=0.5;  LOSS=sigmoid; LR=5e-5; EPOCHS=3; LS=0.0;  HW=1.4; PAIRS="$PAIRS_UNBAL"; SEED=22 ;;
  28) BETA=0.5;  LOSS=sigmoid; LR=5e-5; EPOCHS=1; LS=0.05; HW=1.4; PAIRS="$PAIRS_UNBAL"; SEED=22 ;;
  29) BETA=0.5;  LOSS=sigmoid; LR=5e-5; EPOCHS=1; LS=0.0;  HW=1.4; PAIRS="$PAIRS_UNBAL"; SEED=42 ;;
  # ── Group E: β sweep with hw=1.4, 2 and 3 epochs ──────────────────────────
  30) BETA=0.1;  LOSS=sigmoid; LR=5e-5; EPOCHS=2; LS=0.0;  HW=1.4; PAIRS="$PAIRS_UNBAL"; SEED=22 ;;
  31) BETA=0.2;  LOSS=sigmoid; LR=5e-5; EPOCHS=2; LS=0.0;  HW=1.4; PAIRS="$PAIRS_UNBAL"; SEED=22 ;;
  32) BETA=0.3;  LOSS=sigmoid; LR=5e-5; EPOCHS=2; LS=0.0;  HW=1.4; PAIRS="$PAIRS_UNBAL"; SEED=22 ;;
  33) BETA=0.1;  LOSS=sigmoid; LR=5e-5; EPOCHS=3; LS=0.0;  HW=1.4; PAIRS="$PAIRS_UNBAL"; SEED=22 ;;
  34) BETA=0.2;  LOSS=sigmoid; LR=5e-5; EPOCHS=3; LS=0.0;  HW=1.4; PAIRS="$PAIRS_UNBAL"; SEED=22 ;;
  35) BETA=0.5;  LOSS=sigmoid; LR=5e-5; EPOCHS=4; LS=0.0;  HW=1.4; PAIRS="$PAIRS_UNBAL"; SEED=22 ;;
esac

BAL_TAG=$([ "$PAIRS" = "$PAIRS_BAL" ] && echo "bal" || echo "unbal")
HW_TAG=$([ "$(echo "$HW > 1.0" | bc -l)" = "1" ] && echo "hw${HW}" || echo "")
RUN_TAG="${BAL_TAG}${HW_TAG:+_$HW_TAG}"

RUN_NAME="dpo-run${SLURM_ARRAY_TASK_ID}-b${BETA}-lr${LR}-e${EPOCHS}-ls${LS}-${RUN_TAG}-s${SEED}"
OUTPUT_DIR="trained_models/causal/dpo_sweep/run_$(printf '%02d' $SLURM_ARRAY_TASK_ID)_b${BETA}_lr${LR}_e${EPOCHS}_ls${LS}_${RUN_TAG}_s${SEED}"

echo "========================================"
echo "Run    : $SLURM_ARRAY_TASK_ID"
echo "β=$BETA  loss=$LOSS  lr=$LR  epochs=$EPOCHS  ls=$LS  hw=$HW  seed=$SEED"
echo "pairs  : $PAIRS"
echo "output : $OUTPUT_DIR"
echo "========================================"

python scripts/train_dpo.py \
    --pairs-path      "$PAIRS" \
    --adapter-path    "$SFT_ADAPTER" \
    --base-model      meta-llama/Llama-3.1-8B-Instruct \
    --output-dir      "$OUTPUT_DIR" \
    --beta            "$BETA" \
    --loss-type       "$LOSS" \
    --label-smoothing "$LS" \
    --harmful-weight  "$HW" \
    --epochs          "$EPOCHS" \
    --learning-rate   "$LR" \
    --batch-size      2 \
    --gradient-accumulation 8 \
    --seed            "$SEED" \
    --wandb-project   dpo-sweep \
    --wandb-run       "$RUN_NAME" \
    --skip-eval

echo "[DONE] Run $SLURM_ARRAY_TASK_ID finished"
