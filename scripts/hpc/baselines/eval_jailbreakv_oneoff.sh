#!/bin/bash
#SBATCH --job-name=jailbreakv_oneoff
#SBATCH --time=04:00:00
#SBATCH --mem=32GB
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=rtx_pro_6000:1
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/slurm/%x-%j.out

# Run a single fine-tuned condition (or GRPO) on the three JailbreakV-28K splits.
# Outputs land in data/safety_experiment/jailbreakv/<OUTPUT_NAME>/<dataset>/.
#
# Required env vars (passed via --export=ALL,...):
#   BASE_MODEL        — HF id of the base model (ignored for grpo_classification)
#   CONDITION         — one of finetuned_generation / grpo_classification / ...
#   ADAPTER_PATH      — local path to the LoRA adapter (use any string for grpo_classification)
#   OUTPUT_NAME       — subdir under data/safety_experiment/jailbreakv/
#
# Optional:
#   LORA_RANK         — defaults to 16 (32 for v7 distillation adapters)
#   WANDB_RUN_NAME    — defaults to OUTPUT_NAME

set -euo pipefail

module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.8.0

source .venv/bin/activate

mkdir -p "logs/slurm"

: "${BASE_MODEL:?BASE_MODEL is required}"
: "${CONDITION:?CONDITION is required}"
: "${OUTPUT_NAME:?OUTPUT_NAME is required}"
ADAPTER_PATH="${ADAPTER_PATH:-NOT_USED}"
LORA_RANK="${LORA_RANK:-16}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-${OUTPUT_NAME}}"

OUTPUT_DIR="data/safety_experiment/jailbreakv/${OUTPUT_NAME}"

echo "Base model:   ${BASE_MODEL}"
echo "Condition:    ${CONDITION}"
echo "Adapter path: ${ADAPTER_PATH}"
echo "Output dir:   ${OUTPUT_DIR}"
echo "LoRA rank:    ${LORA_RANK}"

python scripts/eval_safety_classifier.py \
    --config-name=eval_jailbreakv_oneoff \
    "model.name=${BASE_MODEL}" \
    "experiment.conditions=[${CONDITION}]" \
    "finetuned.generation_adapter=${ADAPTER_PATH}" \
    "paths.output_dir=${OUTPUT_DIR}" \
    "lora.rank=${LORA_RANK}" \
    "wandb.run_name=${WANDB_RUN_NAME}"
