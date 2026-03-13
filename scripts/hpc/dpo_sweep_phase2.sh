#!/bin/bash
#SBATCH --job-name=dpo-sweep-p2
#SBATCH --array=1-7
#SBATCH --time=00:50:00
#SBATCH --mem=48GB
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=a100:1
#SBATCH --output=logs/slurm/dpo-sweep-p2-%A_%a.out
#SBATCH --error=logs/slurm/dpo-sweep-p2-%A_%a.err

# ── Phase 2: lr + epochs + label_smoothing sweep ─────────────────────────────
# Fill in BEST_BETA and BEST_LOSS from Phase 1 W&B results before submitting.
#
# Run | lr    | epochs | label_smoothing | seed
#  1  | 1e-5  | 3      | 0.0             | 22
#  2  | 5e-5  | 3      | 0.0             | 22   ← reference (same as Phase 1 best)
#  3  | 5e-5  | 1      | 0.0             | 22
#  4  | 5e-5  | 3      | 0.1             | 22
#  5  | 5e-5  | 3      | 0.2             | 22
#  6  | best  | best   | best            | 22   ← combined best, seed 22
#  7  | best  | best   | best            | 42   ← combined best, seed 42

set -euo pipefail
mkdir -p logs/slurm

module purge
module load CUDA/12.1.1
module load Python/3.11.3-GCCcore-12.3.0

cd "$HOME/repos/intent-gen"
source .venv/bin/activate

export HF_HOME="$HOME/.cache/huggingface"
export WANDB_MODE=online

# ── SET THESE FROM PHASE 1 RESULTS ───────────────────────────────────────────
BEST_BETA=0.3       # <-- update after Phase 1
BEST_LOSS=sigmoid   # <-- update after Phase 1
# ─────────────────────────────────────────────────────────────────────────────

case $SLURM_ARRAY_TASK_ID in
  1) LR=1e-5; EPOCHS=3; LS=0.0;  SEED=22 ;;
  2) LR=5e-5; EPOCHS=3; LS=0.0;  SEED=22 ;;
  3) LR=5e-5; EPOCHS=1; LS=0.0;  SEED=22 ;;
  4) LR=5e-5; EPOCHS=3; LS=0.1;  SEED=22 ;;
  5) LR=5e-5; EPOCHS=3; LS=0.2;  SEED=22 ;;
  6) LR=5e-5; EPOCHS=3; LS=0.0;  SEED=22 ;;   
  7) LR=5e-5; EPOCHS=3; LS=0.0;  SEED=42 ;;   
esac

RUN_NAME="dpo-sweep-p2-run${SLURM_ARRAY_TASK_ID}-beta${BEST_BETA}-${BEST_LOSS}-lr${LR}-e${EPOCHS}-ls${LS}-s${SEED}"
OUTPUT_DIR="trained_models/causal/dpo_sweep/p2_run_$(printf '%02d' $SLURM_ARRAY_TASK_ID)_lr${LR}_e${EPOCHS}_ls${LS}_s${SEED}"

echo "========================================"
echo "SLURM array task : $SLURM_ARRAY_TASK_ID"
echo "β                : $BEST_BETA"
echo "loss_type        : $BEST_LOSS"
echo "lr               : $LR"
echo "epochs           : $EPOCHS"
echo "label_smoothing  : $LS"
echo "seed             : $SEED"
echo "output_dir       : $OUTPUT_DIR"
echo "========================================"

python scripts/train_dpo.py \
    --pairs-path     data/dpo_pairs/train_t0.8/dpo_pairs.jsonl \
    --adapter-path   trained_models/causal/hyperparam_sweep/lr_0.0005_e_5_adapter \
    --base-model     meta-llama/Llama-3.1-8B-Instruct \
    --output-dir     "$OUTPUT_DIR" \
    --beta           "$BEST_BETA" \
    --loss-type      "$BEST_LOSS" \
    --label-smoothing "$LS" \
    --epochs         "$EPOCHS" \
    --learning-rate  "$LR" \
    --batch-size     2 \
    --gradient-accumulation 8 \
    --seed           "$SEED" \
    --wandb-project  dpo-sweep \
    --wandb-run      "$RUN_NAME"

echo "[DONE] Task $SLURM_ARRAY_TASK_ID finished"
