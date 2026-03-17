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
case $SLURM_ARRAY_TASK_ID in
  # Phase 1 (unbalanced)
  1)  ADAPTER="trained_models/causal/dpo_sweep/run_01_sigmoid_b0.1_lr5e-5_e3_ls0.0_unbal_adapter" ;;
  2)  ADAPTER="trained_models/causal/dpo_sweep/run_02_sigmoid_b0.2_lr5e-5_e3_ls0.0_unbal_adapter" ;;
  3)  ADAPTER="trained_models/causal/dpo_sweep/run_03_sigmoid_b0.3_lr5e-5_e3_ls0.0_unbal_adapter" ;;
  4)  ADAPTER="trained_models/causal/dpo_sweep/run_04_sigmoid_b0.5_lr5e-5_e3_ls0.0_unbal_adapter" ;;
  5)  ADAPTER="trained_models/causal/dpo_sweep/run_05_ipo_b0.1_lr5e-5_e3_ls0.0_unbal_adapter"     ;;
  6)  ADAPTER="trained_models/causal/dpo_sweep/run_06_ipo_b0.2_lr5e-5_e3_ls0.0_unbal_adapter"     ;;
  7)  ADAPTER="trained_models/causal/dpo_sweep/run_07_ipo_b0.3_lr5e-5_e3_ls0.0_unbal_adapter"     ;;
  8)  ADAPTER="trained_models/causal/dpo_sweep/run_08_ipo_b0.5_lr5e-5_e3_ls0.0_unbal_adapter"     ;;
  # Phase 2 (unbalanced)
  9)  ADAPTER="trained_models/causal/dpo_sweep/run_09_sigmoid_b0.5_lr1e-5_e3_ls0.0_unbal_adapter" ;;
  10) ADAPTER="trained_models/causal/dpo_sweep/run_10_sigmoid_b0.5_lr5e-5_e3_ls0.0_unbal_adapter" ;;
  11) ADAPTER="trained_models/causal/dpo_sweep/run_11_sigmoid_b0.5_lr5e-5_e1_ls0.0_unbal_adapter" ;;
  12) ADAPTER="trained_models/causal/dpo_sweep/run_12_sigmoid_b0.5_lr5e-5_e3_ls0.1_unbal_adapter" ;;
  13) ADAPTER="trained_models/causal/dpo_sweep/run_13_sigmoid_b0.5_lr5e-5_e3_ls0.2_unbal_adapter" ;;
  14) ADAPTER="trained_models/causal/dpo_sweep/run_14_sigmoid_b0.7_lr5e-5_e3_ls0.0_unbal_adapter" ;;
  15) ADAPTER="trained_models/causal/dpo_sweep/run_15_sigmoid_b1.0_lr5e-5_e3_ls0.0_unbal_adapter" ;;
  # Phase 3 (balanced) — use existing adapter dirs
  16) ADAPTER="trained_models/causal/dpo_sweep/p3_run_01_sigmoid_beta0.5_lr5e-5_e1_ls0.0_adapter" ;;
  17) ADAPTER="trained_models/causal/dpo_sweep/p3_run_02_sigmoid_beta0.5_lr5e-5_e1_ls0.1_adapter" ;;
  18) ADAPTER="trained_models/causal/dpo_sweep/p3_run_03_ipo_beta0.5_lr5e-5_e1_ls0.0_adapter"     ;;
  19) ADAPTER="trained_models/causal/dpo_sweep/p3_run_04_ipo_beta0.3_lr5e-5_e1_ls0.0_adapter"     ;;
esac

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
