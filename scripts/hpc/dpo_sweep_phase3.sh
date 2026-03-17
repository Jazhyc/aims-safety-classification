#!/bin/bash
#SBATCH --job-name=dpo-sweep-p3
#SBATCH --array=1-4
#SBATCH --time=00:50:00
#SBATCH --mem=48GB
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=a100:1
#SBATCH --output=logs/slurm/dpo-sweep-p3-%A_%a.out
#SBATCH --error=logs/slurm/dpo-sweep-p3-%A_%a.err

# ── Phase 3: balanced DPO pairs (50/50 safe/harmful) ─────────────────────────
# Key change vs Phase 2: --pairs-path points to balanced file (728 pairs, 50/50)
# which fixes the 58.4% safe bias that was killing harmful recall.
#
# Run | loss    | β   | lr    | epochs | label_smoothing
#  1  | sigmoid | 0.5 | 5e-5  | 1      | 0.0   ← best from p2 (run_03), now balanced
#  2  | sigmoid | 0.5 | 5e-5  | 1      | 0.1   ← best from p2 (run_04), now balanced
#  3  | ipo     | 0.5 | 5e-5  | 1      | 0.0   ← IPO: less sensitive to off-policy chosen
#  4  | ipo     | 0.3 | 5e-5  | 1      | 0.0   ← IPO with lower β

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

BALANCED_PAIRS="data/dpo_pairs/train_t0.8_balanced/dpo_pairs.jsonl"
SEED=22

case $SLURM_ARRAY_TASK_ID in
  1) LOSS=sigmoid; BETA=0.5; LR=5e-5; EPOCHS=1; LS=0.0 ;;
  2) LOSS=sigmoid; BETA=0.5; LR=5e-5; EPOCHS=1; LS=0.1 ;;
  3) LOSS=ipo;     BETA=0.5; LR=5e-5; EPOCHS=1; LS=0.0 ;;
  4) LOSS=ipo;     BETA=0.3; LR=5e-5; EPOCHS=1; LS=0.0 ;;
esac

RUN_NAME="dpo-sweep-p3-run${SLURM_ARRAY_TASK_ID}-${LOSS}-beta${BETA}-lr${LR}-e${EPOCHS}-ls${LS}"
OUTPUT_DIR="trained_models/causal/dpo_sweep/p3_run_$(printf '%02d' $SLURM_ARRAY_TASK_ID)_${LOSS}_beta${BETA}_lr${LR}_e${EPOCHS}_ls${LS}"

echo "========================================"
echo "SLURM array task : $SLURM_ARRAY_TASK_ID"
echo "loss_type        : $LOSS"
echo "β                : $BETA"
echo "lr               : $LR"
echo "epochs           : $EPOCHS"
echo "label_smoothing  : $LS"
echo "pairs            : $BALANCED_PAIRS"
echo "output_dir       : $OUTPUT_DIR"
echo "========================================"

python scripts/train_dpo.py \
    --pairs-path      "$BALANCED_PAIRS" \
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
    --seed            "$SEED" \
    --wandb-project   dpo-sweep \
    --wandb-run       "$RUN_NAME"

echo "[DONE] Task $SLURM_ARRAY_TASK_ID finished"
