#!/bin/bash
#SBATCH --job-name=dpo-sweep-p1
#SBATCH --array=1-8
#SBATCH --time=00:50:00
#SBATCH --mem=48GB
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=a100:1
#SBATCH --output=logs/slurm/dpo-sweep-p1-%A_%a.out
#SBATCH --error=logs/slurm/dpo-sweep-p1-%A_%a.err

# ── Phase 1: β × loss_type sweep ─────────────────────────────────────────────
# Run | β    | loss_type
#  1  | 0.1  | sigmoid
#  2  | 0.2  | sigmoid
#  3  | 0.3  | sigmoid
#  4  | 0.5  | sigmoid
#  5  | 0.1  | ipo
#  6  | 0.2  | ipo
#  7  | 0.3  | ipo
#  8  | 0.5  | ipo

set -euo pipefail
mkdir -p logs/slurm

module purge
module load CUDA/12.1.1
module load Python/3.11.3-GCCcore-12.3.0

cd "$HOME/repos/intent-gen"
source .venv/bin/activate

# ── User-specific paths (override these for different users/clusters) ────────
PROJECTS_DIR="${PROJECTS_DIR:-/projects/s4351495}"
export HF_HOME="${HF_HOME:-${PROJECTS_DIR}/huggingface_cache}"
SFT_ADAPTER="${SFT_ADAPTER:-${PROJECTS_DIR}/trained_models/causal/hyperparam_sweep/lr_0.0005_e_5_adapter}"
# ─────────────────────────────────────────────────────────────────────────────
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=online

# ── Hyperparameter grid ───────────────────────────────────────────────────────
case $SLURM_ARRAY_TASK_ID in
  1) BETA=0.1; LOSS=sigmoid ;;
  2) BETA=0.2; LOSS=sigmoid ;;
  3) BETA=0.3; LOSS=sigmoid ;;
  4) BETA=0.5; LOSS=sigmoid ;;
  5) BETA=0.1; LOSS=ipo     ;;
  6) BETA=0.2; LOSS=ipo     ;;
  7) BETA=0.3; LOSS=ipo     ;;
  8) BETA=0.5; LOSS=ipo     ;;
esac

RUN_NAME="dpo-sweep-p1-beta${BETA}-${LOSS}"
OUTPUT_DIR="trained_models/causal/dpo_sweep/run_$(printf '%02d' $SLURM_ARRAY_TASK_ID)_beta${BETA}_${LOSS}"

echo "========================================"
echo "SLURM array task : $SLURM_ARRAY_TASK_ID"
echo "β                : $BETA"
echo "loss_type        : $LOSS"
echo "output_dir       : $OUTPUT_DIR"
echo "wandb run        : $RUN_NAME"
echo "========================================"

python scripts/train_dpo.py \
    --pairs-path     data/dpo_pairs/train_t0.8/dpo_pairs.jsonl \
    --adapter-path   "$SFT_ADAPTER" \
    --base-model     meta-llama/Llama-3.1-8B-Instruct \
    --output-dir     "$OUTPUT_DIR" \
    --beta           "$BETA" \
    --loss-type      "$LOSS" \
    --label-smoothing 0.0 \
    --epochs         3 \
    --learning-rate  5e-5 \
    --batch-size     2 \
    --gradient-accumulation 8 \
    --wandb-project  dpo-sweep \
    --wandb-run      "$RUN_NAME"

echo "[DONE] Task $SLURM_ARRAY_TASK_ID finished"
