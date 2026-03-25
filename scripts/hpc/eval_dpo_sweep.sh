#!/bin/bash
#SBATCH --job-name=eval-dpo-sweep
#SBATCH --time=00:30:00
#SBATCH --mem=32GB
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=a100:1
#SBATCH --output=logs/slurm/eval-dpo-sweep-%A_%a.out
#SBATCH --error=logs/slurm/eval-dpo-sweep-%A_%a.err

# Evaluate DPO sweep adapters on the test set.
# Each task evaluates one adapter and saves predictions + metrics.
# After all tasks finish, run compare_dpo_sweep.py locally to get a combined table.
#
# Submit Group C+D runs:
#   sbatch --array=6-29  scripts/hpc/eval_dpo_sweep.sh
#
# Submit everything (requires all adapters to exist):
#   sbatch --array=1-29  scripts/hpc/eval_dpo_sweep.sh
#
# Run IDs match dpo_sweep.sh exactly — same numbering.

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
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Map run ID → adapter path (must match output dirs from dpo_sweep.sh)
# Adapter dir = OUTPUT_DIR + "_adapter"  (set by train_dpo.py)
case $SLURM_ARRAY_TASK_ID in
  # Group A
  1)  ADAPTER=$(ls -d trained_models/causal/dpo_sweep/run_01_*_adapter 2>/dev/null | head -1) ;;
  # Group B
  2)  ADAPTER=$(ls -d trained_models/causal/dpo_sweep/run_02_*_adapter 2>/dev/null | head -1) ;;
  3)  ADAPTER=$(ls -d trained_models/causal/dpo_sweep/run_03_*_adapter 2>/dev/null | head -1) ;;
  4)  ADAPTER=$(ls -d trained_models/causal/dpo_sweep/run_04_*_adapter 2>/dev/null | head -1) ;;
  5)  ADAPTER=$(ls -d trained_models/causal/dpo_sweep/run_05_*_adapter 2>/dev/null | head -1) ;;
  # Group C-us
  6)  ADAPTER=$(ls -d trained_models/causal/dpo_sweep/run_06_*_adapter 2>/dev/null | head -1) ;;
  7)  ADAPTER=$(ls -d trained_models/causal/dpo_sweep/run_07_*_adapter 2>/dev/null | head -1) ;;
  8)  ADAPTER=$(ls -d trained_models/causal/dpo_sweep/run_08_*_adapter 2>/dev/null | head -1) ;;
  9)  ADAPTER=$(ls -d trained_models/causal/dpo_sweep/run_09_*_adapter 2>/dev/null | head -1) ;;
  10) ADAPTER=$(ls -d trained_models/causal/dpo_sweep/run_10_*_adapter 2>/dev/null | head -1) ;;
  11) ADAPTER=$(ls -d trained_models/causal/dpo_sweep/run_11_*_adapter 2>/dev/null | head -1) ;;
  12) ADAPTER=$(ls -d trained_models/causal/dpo_sweep/run_12_*_adapter 2>/dev/null | head -1) ;;
  13) ADAPTER=$(ls -d trained_models/causal/dpo_sweep/run_13_*_adapter 2>/dev/null | head -1) ;;
  # Group C-wl
  14) ADAPTER=$(ls -d trained_models/causal/dpo_sweep/run_14_*_adapter 2>/dev/null | head -1) ;;
  15) ADAPTER=$(ls -d trained_models/causal/dpo_sweep/run_15_*_adapter 2>/dev/null | head -1) ;;
  16) ADAPTER=$(ls -d trained_models/causal/dpo_sweep/run_16_*_adapter 2>/dev/null | head -1) ;;
  17) ADAPTER=$(ls -d trained_models/causal/dpo_sweep/run_17_*_adapter 2>/dev/null | head -1) ;;
  18) ADAPTER=$(ls -d trained_models/causal/dpo_sweep/run_18_*_adapter 2>/dev/null | head -1) ;;
  19) ADAPTER=$(ls -d trained_models/causal/dpo_sweep/run_19_*_adapter 2>/dev/null | head -1) ;;
  20) ADAPTER=$(ls -d trained_models/causal/dpo_sweep/run_20_*_adapter 2>/dev/null | head -1) ;;
  21) ADAPTER=$(ls -d trained_models/causal/dpo_sweep/run_21_*_adapter 2>/dev/null | head -1) ;;
  # Group D-us
  22) ADAPTER=$(ls -d trained_models/causal/dpo_sweep/run_22_*_adapter 2>/dev/null | head -1) ;;
  23) ADAPTER=$(ls -d trained_models/causal/dpo_sweep/run_23_*_adapter 2>/dev/null | head -1) ;;
  24) ADAPTER=$(ls -d trained_models/causal/dpo_sweep/run_24_*_adapter 2>/dev/null | head -1) ;;
  25) ADAPTER=$(ls -d trained_models/causal/dpo_sweep/run_25_*_adapter 2>/dev/null | head -1) ;;
  # Group D-wl
  26) ADAPTER=$(ls -d trained_models/causal/dpo_sweep/run_26_*_adapter 2>/dev/null | head -1) ;;
  27) ADAPTER=$(ls -d trained_models/causal/dpo_sweep/run_27_*_adapter 2>/dev/null | head -1) ;;
  28) ADAPTER=$(ls -d trained_models/causal/dpo_sweep/run_28_*_adapter 2>/dev/null | head -1) ;;
  29) ADAPTER=$(ls -d trained_models/causal/dpo_sweep/run_29_*_adapter 2>/dev/null | head -1) ;;
  # Group E
  30) ADAPTER=$(ls -d trained_models/causal/dpo_sweep/run_30_*_adapter 2>/dev/null | head -1) ;;
  31) ADAPTER=$(ls -d trained_models/causal/dpo_sweep/run_31_*_adapter 2>/dev/null | head -1) ;;
  32) ADAPTER=$(ls -d trained_models/causal/dpo_sweep/run_32_*_adapter 2>/dev/null | head -1) ;;
  33) ADAPTER=$(ls -d trained_models/causal/dpo_sweep/run_33_*_adapter 2>/dev/null | head -1) ;;
  34) ADAPTER=$(ls -d trained_models/causal/dpo_sweep/run_34_*_adapter 2>/dev/null | head -1) ;;
  35) ADAPTER=$(ls -d trained_models/causal/dpo_sweep/run_35_*_adapter 2>/dev/null | head -1) ;;
esac

if [ -z "$ADAPTER" ] || [ ! -d "$ADAPTER" ]; then
    echo "ERROR: adapter not found for run $SLURM_ARRAY_TASK_ID"
    exit 1
fi

OUTPUT_DIR="data/predictions/dpo_sweep/run_$(printf '%02d' $SLURM_ARRAY_TASK_ID)"

echo "========================================"
echo "Run $SLURM_ARRAY_TASK_ID"
echo "Adapter : $ADAPTER"
echo "Output  : $OUTPUT_DIR"
echo "========================================"

python scripts/baselines/eval_sft_baseline.py \
    --adapter-path "$ADAPTER" \
    --base-model   meta-llama/Llama-3.1-8B-Instruct \
    --output-dir   "$OUTPUT_DIR"

echo "[DONE] Run $SLURM_ARRAY_TASK_ID → $OUTPUT_DIR"
