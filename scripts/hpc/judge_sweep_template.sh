#!/bin/bash
#SBATCH --job-name=dpo-sweep
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
CONDITION="${CONDITION:-judge}"       # judge | hard
SWEEP_BETA="${SWEEP_BETA:-0.3}"
SWEEP_EPOCHS="${SWEEP_EPOCHS:-1}"
SEED="${SEED:-22}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
HARMFUL_WEIGHT="${HARMFUL_WEIGHT:-1.0}"
UNBALANCED="${UNBALANCED:-0}"

PAIRS_DIR="data/dpo_pairs/${CONDITION}"
BALANCED_DIR="data/dpo_pairs/${CONDITION}_balanced"
OUTPUT_DIR="trained_models/causal/dpo-${CONDITION}-sweep/beta${SWEEP_BETA}_e${SWEEP_EPOCHS}"

echo "======================================================================"
echo " DPO Condition Sweep"
echo "  Condition   : ${CONDITION}"
echo "  Beta        : ${SWEEP_BETA}"
echo "  Epochs      : ${SWEEP_EPOCHS}"
echo "  Unbalanced  : ${UNBALANCED}  (harmful_weight=${HARMFUL_WEIGHT})"
echo "  Output dir  : ${OUTPUT_DIR}"
echo "======================================================================"

EXTRA_ARGS=()
if [ "${UNBALANCED}" = "1" ]; then
  EXTRA_ARGS+=(--unbalanced --harmful-weight "${HARMFUL_WEIGHT}")
fi

python scripts/run_preference_pipeline.py \
  --adapter-path       "${BASE_SFT_ADAPTER}" \
  --base-model         "${BASE_MODEL}" \
  --pairs-dir          "${PAIRS_DIR}" \
  --balanced-pairs-dir "${BALANCED_DIR}" \
  --dpo-output-dir     "${OUTPUT_DIR}" \
  --sft-pred-dir       "data/predictions/sft_hard" \
  --epochs             "${SWEEP_EPOCHS}" \
  --learning-rate      "${LEARNING_RATE}" \
  --batch-size         "${BATCH_SIZE}" \
  --grad-accum         "${GRAD_ACCUM}" \
  --dpo-beta           "${SWEEP_BETA}" \
  --seed               "${SEED}" \
  --wandb-project      "intention-jailbreak" \
  "${EXTRA_ARGS[@]}" \
  --force-from 3 \
  --skip-steps 1,6

echo ""
echo "Run ${CONDITION} beta=${SWEEP_BETA} epochs=${SWEEP_EPOCHS} completed."
echo "Predictions at: ${OUTPUT_DIR}/predictions/"
