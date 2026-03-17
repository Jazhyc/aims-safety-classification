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
# Submit all phase 3 runs:
#   sbatch --array=16-19 scripts/hpc/eval_dpo_sweep.sh
#
# Submit everything (requires all adapters to exist):
#   sbatch --array=1-19  scripts/hpc/eval_dpo_sweep.sh
#
# Run IDs match dpo_sweep.sh exactly — same numbering.

set -euo pipefail
mkdir -p logs/slurm

module purge
module load CUDA/12.1.1
module load Python/3.11.3-GCCcore-12.3.0

cd "$HOME/repos/intent-gen"
source .venv/bin/activate

PROJECTS_DIR="${PROJECTS_DIR:-/scratch/s4351495}"
export HF_HOME="${HF_HOME:-${PROJECTS_DIR}/huggingface_cache}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Map run ID → adapter path (must match output dirs from dpo_sweep.sh)
# Adapter dir = OUTPUT_DIR + "_adapter"  (set by train_dpo.py)
# Use: find trained_models/causal/dpo_sweep/ -maxdepth 1 -name "run_XX*adapter" -type d
case $SLURM_ARRAY_TASK_ID in
  # Group A: unbalanced reference
  1)  ADAPTER=$(ls -d trained_models/causal/dpo_sweep/run_01_*_adapter 2>/dev/null | head -1) ;;
  # Group B: balancing method comparison
  2)  ADAPTER=$(ls -d trained_models/causal/dpo_sweep/run_02_*_adapter 2>/dev/null | head -1) ;;
  3)  ADAPTER=$(ls -d trained_models/causal/dpo_sweep/run_03_*_adapter 2>/dev/null | head -1) ;;
  4)  ADAPTER=$(ls -d trained_models/causal/dpo_sweep/run_04_*_adapter 2>/dev/null | head -1) ;;
  5)  ADAPTER=$(ls -d trained_models/causal/dpo_sweep/run_05_*_adapter 2>/dev/null | head -1) ;;
  # Group C: β sweep
  6)  ADAPTER=$(ls -d trained_models/causal/dpo_sweep/run_06_*_adapter 2>/dev/null | head -1) ;;
  7)  ADAPTER=$(ls -d trained_models/causal/dpo_sweep/run_07_*_adapter 2>/dev/null | head -1) ;;
  8)  ADAPTER=$(ls -d trained_models/causal/dpo_sweep/run_08_*_adapter 2>/dev/null | head -1) ;;
  9)  ADAPTER=$(ls -d trained_models/causal/dpo_sweep/run_09_*_adapter 2>/dev/null | head -1) ;;
  10) ADAPTER=$(ls -d trained_models/causal/dpo_sweep/run_10_*_adapter 2>/dev/null | head -1) ;;
  11) ADAPTER=$(ls -d trained_models/causal/dpo_sweep/run_11_*_adapter 2>/dev/null | head -1) ;;
  # Group D: epochs + ls
  12) ADAPTER=$(ls -d trained_models/causal/dpo_sweep/run_12_*_adapter 2>/dev/null | head -1) ;;
  13) ADAPTER=$(ls -d trained_models/causal/dpo_sweep/run_13_*_adapter 2>/dev/null | head -1) ;;
  14) ADAPTER=$(ls -d trained_models/causal/dpo_sweep/run_14_*_adapter 2>/dev/null | head -1) ;;
  15) ADAPTER=$(ls -d trained_models/causal/dpo_sweep/run_15_*_adapter 2>/dev/null | head -1) ;;
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

python scripts/eval_sft_baseline.py \
    --adapter-path "$ADAPTER" \
    --base-model   meta-llama/Llama-3.1-8B-Instruct \
    --output-dir   "$OUTPUT_DIR"

echo "[DONE] Run $SLURM_ARRAY_TASK_ID → $OUTPUT_DIR"
