#!/bin/bash
#SBATCH --job-name=judge-sweep
#SBATCH --time=08:00:00
#SBATCH --mem=48G
#SBATCH --partition=gpumedium
#SBATCH --gpus-per-node=a100:1
#SBATCH --output=logs/slurm/%x-%j.out

set -euo pipefail

module purge
module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.8.0

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${PROJECT_ROOT}"

source .venv/bin/activate
if [ -f "${HOME}/.wandb_secrets" ]; then
  # shellcheck disable=SC1090
  source "${HOME}/.wandb_secrets"
fi

export PYTHONUNBUFFERED=1

BASE_MODEL="${BASE_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
BASE_SFT_ADAPTER="${BASE_SFT_ADAPTER:-trained_models/causal/hyperparam_sweep/lr_5e-05_e_5_adapter}"
JUDGE_BETA="${JUDGE_BETA:-0.3}"
JUDGE_EPOCHS="${JUDGE_EPOCHS:-1}"
SEED="${SEED:-22}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"

OUTPUT_DIR="trained_models/causal/dpo-judge-sweep/beta${JUDGE_BETA}_e${JUDGE_EPOCHS}"

echo "======================================================================"
echo " DPO-Judge Sweep Run"
echo "  Beta        : ${JUDGE_BETA}"
echo "  Epochs      : ${JUDGE_EPOCHS}"
echo "  Output dir  : ${OUTPUT_DIR}"
echo "======================================================================"

# Pairs and balanced pairs already exist from the main judge_dpo run.
# Skip steps 1 (pair gen) and 6 (compare). Force step 3 onward so each
# sweep run trains fresh even if a previous run used the same output dir.
python scripts/run_preference_pipeline.py \
  --adapter-path    "${BASE_SFT_ADAPTER}" \
  --base-model      "${BASE_MODEL}" \
  --pairs-dir       "data/dpo_pairs/judge" \
  --balanced-pairs-dir "data/dpo_pairs/judge_balanced" \
  --dpo-output-dir  "${OUTPUT_DIR}" \
  --sft-pred-dir    "data/predictions/sft_hard" \
  --epochs          "${JUDGE_EPOCHS}" \
  --learning-rate   "${LEARNING_RATE}" \
  --batch-size      "${BATCH_SIZE}" \
  --grad-accum      "${GRAD_ACCUM}" \
  --dpo-beta        "${JUDGE_BETA}" \
  --seed            "${SEED}" \
  --wandb-project   "intention-jailbreak" \
  --force-from 3 \
  --skip-steps 1,6

echo ""
echo "Run beta=${JUDGE_BETA} epochs=${JUDGE_EPOCHS} completed."
echo "Predictions at: ${OUTPUT_DIR}/predictions/"
