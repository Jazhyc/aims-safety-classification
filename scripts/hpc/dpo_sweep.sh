#!/bin/bash
#SBATCH --job-name=dpo-sweep
#SBATCH --time=00:50:00
#SBATCH --mem=48GB
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=a100:1
#SBATCH --output=logs/slurm/dpo-sweep-%A_%a.out
#SBATCH --error=logs/slurm/dpo-sweep-%A_%a.err

# ── DPO hyperparameter sweep ──────────────────────────────────────────────────
#
# Submit a range with --array=<range>, e.g.:
#   sbatch --array=1     scripts/hpc/dpo_sweep.sh   # unbalanced reference
#   sbatch --array=2-5   scripts/hpc/dpo_sweep.sh   # balancing method comparison
#   sbatch --array=6-15  scripts/hpc/dpo_sweep.sh   # Phase 4 proper β/epoch/ls sweep
#   sbatch --array=1-15  scripts/hpc/dpo_sweep.sh   # everything
#
# ── GROUP A: Unbalanced reference (to show balancing helped) ─────────────────
# Run |  β  | loss    | lr    | e | ls  | hw  | pairs    | note
#  1  | 0.5 | sigmoid | 5e-5  | 1 | 0.0 | 1.0 | unbal    | best Phase 2 config, unbalanced
#
# ── GROUP B: Balancing method comparison (same config, different strategy) ───
# Run |  β  | loss    | lr    | e | ls  | hw  | pairs    | note
#  2  | 0.5 | sigmoid | 5e-5  | 1 | 0.0 | 1.0 | bal(us)  | undersampling  ← Phase 3 run_16
#  3  | 0.5 | sigmoid | 5e-5  | 1 | 0.1 | 1.0 | bal(us)  | undersampling + ls  ← Phase 3 run_17
#  4  | 0.5 | sigmoid | 5e-5  | 1 | 0.0 | 1.4 | unbal    | weighted loss (hw=1.4)
#  5  | 0.5 | sigmoid | 5e-5  | 1 | 0.1 | 1.4 | unbal    | weighted loss + ls
#
# ── GROUP C: Phase 4 — β sweep (fix best balancing method from B) ────────────
# Run |  β  | loss    | lr    | e | ls  | hw  | pairs
#  6  | 0.2 | sigmoid | 5e-5  | 1 | 0.0 | TBD | TBD
#  7  | 0.3 | sigmoid | 5e-5  | 1 | 0.0 | TBD | TBD
#  8  | 0.4 | sigmoid | 5e-5  | 1 | 0.0 | TBD | TBD
#  9  | 0.5 | sigmoid | 5e-5  | 1 | 0.0 | TBD | TBD  (duplicate of best for stability)
# 10  | 0.6 | sigmoid | 5e-5  | 1 | 0.0 | TBD | TBD
# 11  | 0.7 | sigmoid | 5e-5  | 1 | 0.0 | TBD | TBD
#
# ── GROUP D: Phase 4 — epochs and label smoothing ────────────────────────────
# Run |  β  | loss    | lr    | e | ls  | hw  | pairs
# 12  | 0.5 | sigmoid | 5e-5  | 2 | 0.0 | TBD | TBD
# 13  | 0.5 | sigmoid | 5e-5  | 3 | 0.0 | TBD | TBD
# 14  | 0.5 | sigmoid | 5e-5  | 1 | 0.05| TBD | TBD
# 15  | 0.5 | sigmoid | 5e-5  | 1 | 0.0 | TBD | TBD  seed=42 (stability check)
#
# NOTE: Before running Groups C/D (runs 6-15), check Group B results to pick
# the best balancing method (undersampling or weighted loss) and set BEST_* below.

# ── Set these after Group B results are in ───────────────────────────────────
# Best balancing method from Group B:
#   undersampling → BEST_HW=1.0  BEST_PAIRS=balanced
#   weighted loss → BEST_HW=1.4  BEST_PAIRS=unbalanced
BEST_HW="${BEST_HW:-1.0}"
BEST_PAIRS_TYPE="${BEST_PAIRS_TYPE:-balanced}"   # "balanced" or "unbalanced"
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail
mkdir -p logs/slurm

module purge
module load CUDA/12.1.1
module load Python/3.11.3-GCCcore-12.3.0

cd "$HOME/repos/intent-gen"
source .venv/bin/activate

PROJECTS_DIR="${PROJECTS_DIR:-/scratch/s4351495}"
export HF_HOME="${HF_HOME:-${PROJECTS_DIR}/huggingface_cache}"
SFT_ADAPTER="${SFT_ADAPTER:-${PROJECTS_DIR}/trained_models/causal/hyperparam_sweep/lr_0.0005_e_5_adapter}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=online

PAIRS_UNBAL="data/dpo_pairs/train_t0.8/dpo_pairs.jsonl"
PAIRS_BAL="data/dpo_pairs/train_t0.8_balanced/dpo_pairs.jsonl"

if [ "$BEST_PAIRS_TYPE" = "balanced" ]; then
    BEST_PAIRS="$PAIRS_BAL"
else
    BEST_PAIRS="$PAIRS_UNBAL"
fi

case $SLURM_ARRAY_TASK_ID in
  # ── Group A: unbalanced reference ──────────────────────────────────────────
  1)  BETA=0.5; LOSS=sigmoid; LR=5e-5; EPOCHS=1; LS=0.0; HW=1.0;  PAIRS="$PAIRS_UNBAL"; SEED=22 ;;
  # ── Group B: balancing method comparison ───────────────────────────────────
  2)  BETA=0.5; LOSS=sigmoid; LR=5e-5; EPOCHS=1; LS=0.0; HW=1.0;  PAIRS="$PAIRS_BAL";   SEED=22 ;;
  3)  BETA=0.5; LOSS=sigmoid; LR=5e-5; EPOCHS=1; LS=0.1; HW=1.0;  PAIRS="$PAIRS_BAL";   SEED=22 ;;
  4)  BETA=0.5; LOSS=sigmoid; LR=5e-5; EPOCHS=1; LS=0.0; HW=1.4;  PAIRS="$PAIRS_UNBAL"; SEED=22 ;;
  5)  BETA=0.5; LOSS=sigmoid; LR=5e-5; EPOCHS=1; LS=0.1; HW=1.4;  PAIRS="$PAIRS_UNBAL"; SEED=22 ;;
  # ── Group C: β sweep (use best balancing from B) ───────────────────────────
  6)  BETA=0.2; LOSS=sigmoid; LR=5e-5; EPOCHS=1; LS=0.0; HW="$BEST_HW"; PAIRS="$BEST_PAIRS"; SEED=22 ;;
  7)  BETA=0.3; LOSS=sigmoid; LR=5e-5; EPOCHS=1; LS=0.0; HW="$BEST_HW"; PAIRS="$BEST_PAIRS"; SEED=22 ;;
  8)  BETA=0.4; LOSS=sigmoid; LR=5e-5; EPOCHS=1; LS=0.0; HW="$BEST_HW"; PAIRS="$BEST_PAIRS"; SEED=22 ;;
  9)  BETA=0.5; LOSS=sigmoid; LR=5e-5; EPOCHS=1; LS=0.0; HW="$BEST_HW"; PAIRS="$BEST_PAIRS"; SEED=22 ;;
  10) BETA=0.6; LOSS=sigmoid; LR=5e-5; EPOCHS=1; LS=0.0; HW="$BEST_HW"; PAIRS="$BEST_PAIRS"; SEED=22 ;;
  11) BETA=0.7; LOSS=sigmoid; LR=5e-5; EPOCHS=1; LS=0.0; HW="$BEST_HW"; PAIRS="$BEST_PAIRS"; SEED=22 ;;
  # ── Group D: epochs + label smoothing ──────────────────────────────────────
  12) BETA=0.5; LOSS=sigmoid; LR=5e-5; EPOCHS=2; LS=0.0; HW="$BEST_HW"; PAIRS="$BEST_PAIRS"; SEED=22 ;;
  13) BETA=0.5; LOSS=sigmoid; LR=5e-5; EPOCHS=3; LS=0.0; HW="$BEST_HW"; PAIRS="$BEST_PAIRS"; SEED=22 ;;
  14) BETA=0.5; LOSS=sigmoid; LR=5e-5; EPOCHS=1; LS=0.05;HW="$BEST_HW"; PAIRS="$BEST_PAIRS"; SEED=22 ;;
  15) BETA=0.5; LOSS=sigmoid; LR=5e-5; EPOCHS=1; LS=0.0; HW="$BEST_HW"; PAIRS="$BEST_PAIRS"; SEED=42 ;;
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
    --wandb-run       "$RUN_NAME"

echo "[DONE] Run $SLURM_ARRAY_TASK_ID finished"
