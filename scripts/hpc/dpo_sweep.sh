#!/bin/bash
#SBATCH --job-name=dpo-sweep
#SBATCH --time=00:50:00
#SBATCH --mem=48GB
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=a100:1
#SBATCH --output=logs/slurm/dpo-sweep-%A_%a.out
#SBATCH --error=logs/slurm/dpo-sweep-%A_%a.err

# ── Unified DPO hyperparameter sweep ─────────────────────────────────────────
# All runs in one place. Submit a subset with --array=<range>, e.g.:
#   sbatch --array=1-8   scripts/hpc/dpo_sweep.sh   # Phase 1: β × loss_type
#   sbatch --array=9-15  scripts/hpc/dpo_sweep.sh   # Phase 2: lr/epochs/ls
#   sbatch --array=16-19 scripts/hpc/dpo_sweep.sh   # Phase 3: balanced pairs
#   sbatch --array=1-19  scripts/hpc/dpo_sweep.sh   # Everything
#
# ── Phase 1: β × loss_type (unbalanced pairs) ───────────────────────────────
# Run | β    | loss    | lr    | epochs | ls   | pairs
#  1  | 0.1  | sigmoid | 5e-5  | 3      | 0.0  | unbalanced
#  2  | 0.2  | sigmoid | 5e-5  | 3      | 0.0  | unbalanced
#  3  | 0.3  | sigmoid | 5e-5  | 3      | 0.0  | unbalanced
#  4  | 0.5  | sigmoid | 5e-5  | 3      | 0.0  | unbalanced
#  5  | 0.1  | ipo     | 5e-5  | 3      | 0.0  | unbalanced
#  6  | 0.2  | ipo     | 5e-5  | 3      | 0.0  | unbalanced
#  7  | 0.3  | ipo     | 5e-5  | 3      | 0.0  | unbalanced
#  8  | 0.5  | ipo     | 5e-5  | 3      | 0.0  | unbalanced
#
# ── Phase 2: lr / epochs / label_smoothing (unbalanced pairs) ────────────────
# Run | β    | loss    | lr    | epochs | ls   | pairs
#  9  | 0.5  | sigmoid | 1e-5  | 3      | 0.0  | unbalanced
# 10  | 0.5  | sigmoid | 5e-5  | 3      | 0.0  | unbalanced  (p1 reference)
# 11  | 0.5  | sigmoid | 5e-5  | 1      | 0.0  | unbalanced
# 12  | 0.5  | sigmoid | 5e-5  | 3      | 0.1  | unbalanced
# 13  | 0.5  | sigmoid | 5e-5  | 3      | 0.2  | unbalanced
# 14  | 0.7  | sigmoid | 5e-5  | 3      | 0.0  | unbalanced
# 15  | 1.0  | sigmoid | 5e-5  | 3      | 0.0  | unbalanced
#
# ── Phase 3: balanced pairs (50/50 safe/harmful) ─────────────────────────────
# Run | β    | loss    | lr    | epochs | ls   | pairs
# 16  | 0.5  | sigmoid | 5e-5  | 1      | 0.0  | balanced
# 17  | 0.5  | sigmoid | 5e-5  | 1      | 0.1  | balanced
# 18  | 0.5  | ipo     | 5e-5  | 1      | 0.0  | balanced
# 19  | 0.3  | ipo     | 5e-5  | 1      | 0.0  | balanced

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

PAIRS_UNBALANCED="data/dpo_pairs/train_t0.8/dpo_pairs.jsonl"
PAIRS_BALANCED="data/dpo_pairs/train_t0.8_balanced/dpo_pairs.jsonl"

case $SLURM_ARRAY_TASK_ID in
  # Phase 1
  1)  BETA=0.1; LOSS=sigmoid; LR=5e-5; EPOCHS=3; LS=0.0; PAIRS="$PAIRS_UNBALANCED" ;;
  2)  BETA=0.2; LOSS=sigmoid; LR=5e-5; EPOCHS=3; LS=0.0; PAIRS="$PAIRS_UNBALANCED" ;;
  3)  BETA=0.3; LOSS=sigmoid; LR=5e-5; EPOCHS=3; LS=0.0; PAIRS="$PAIRS_UNBALANCED" ;;
  4)  BETA=0.5; LOSS=sigmoid; LR=5e-5; EPOCHS=3; LS=0.0; PAIRS="$PAIRS_UNBALANCED" ;;
  5)  BETA=0.1; LOSS=ipo;     LR=5e-5; EPOCHS=3; LS=0.0; PAIRS="$PAIRS_UNBALANCED" ;;
  6)  BETA=0.2; LOSS=ipo;     LR=5e-5; EPOCHS=3; LS=0.0; PAIRS="$PAIRS_UNBALANCED" ;;
  7)  BETA=0.3; LOSS=ipo;     LR=5e-5; EPOCHS=3; LS=0.0; PAIRS="$PAIRS_UNBALANCED" ;;
  8)  BETA=0.5; LOSS=ipo;     LR=5e-5; EPOCHS=3; LS=0.0; PAIRS="$PAIRS_UNBALANCED" ;;
  # Phase 2
  9)  BETA=0.5; LOSS=sigmoid; LR=1e-5; EPOCHS=3; LS=0.0; PAIRS="$PAIRS_UNBALANCED" ;;
  10) BETA=0.5; LOSS=sigmoid; LR=5e-5; EPOCHS=3; LS=0.0; PAIRS="$PAIRS_UNBALANCED" ;;
  11) BETA=0.5; LOSS=sigmoid; LR=5e-5; EPOCHS=1; LS=0.0; PAIRS="$PAIRS_UNBALANCED" ;;
  12) BETA=0.5; LOSS=sigmoid; LR=5e-5; EPOCHS=3; LS=0.1; PAIRS="$PAIRS_UNBALANCED" ;;
  13) BETA=0.5; LOSS=sigmoid; LR=5e-5; EPOCHS=3; LS=0.2; PAIRS="$PAIRS_UNBALANCED" ;;
  14) BETA=0.7; LOSS=sigmoid; LR=5e-5; EPOCHS=3; LS=0.0; PAIRS="$PAIRS_UNBALANCED" ;;
  15) BETA=1.0; LOSS=sigmoid; LR=5e-5; EPOCHS=3; LS=0.0; PAIRS="$PAIRS_UNBALANCED" ;;
  # Phase 3
  16) BETA=0.5; LOSS=sigmoid; LR=5e-5; EPOCHS=1; LS=0.0; PAIRS="$PAIRS_BALANCED"   ;;
  17) BETA=0.5; LOSS=sigmoid; LR=5e-5; EPOCHS=1; LS=0.1; PAIRS="$PAIRS_BALANCED"   ;;
  18) BETA=0.5; LOSS=ipo;     LR=5e-5; EPOCHS=1; LS=0.0; PAIRS="$PAIRS_BALANCED"   ;;
  19) BETA=0.3; LOSS=ipo;     LR=5e-5; EPOCHS=1; LS=0.0; PAIRS="$PAIRS_BALANCED"   ;;
esac

BALANCED_TAG=$([ "$PAIRS" = "$PAIRS_BALANCED" ] && echo "bal" || echo "unbal")
RUN_NAME="dpo-run${SLURM_ARRAY_TASK_ID}-${LOSS}-b${BETA}-lr${LR}-e${EPOCHS}-ls${LS}-${BALANCED_TAG}"
OUTPUT_DIR="trained_models/causal/dpo_sweep/run_$(printf '%02d' $SLURM_ARRAY_TASK_ID)_${LOSS}_b${BETA}_lr${LR}_e${EPOCHS}_ls${LS}_${BALANCED_TAG}"

echo "========================================"
echo "Run    : $SLURM_ARRAY_TASK_ID"
echo "loss   : $LOSS  β=$BETA  lr=$LR  epochs=$EPOCHS  ls=$LS"
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
    --epochs          "$EPOCHS" \
    --learning-rate   "$LR" \
    --batch-size      2 \
    --gradient-accumulation 8 \
    --seed            22 \
    --wandb-project   dpo-sweep \
    --wandb-run       "$RUN_NAME"

echo "[DONE] Run $SLURM_ARRAY_TASK_ID finished"
