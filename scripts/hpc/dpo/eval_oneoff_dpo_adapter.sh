#!/bin/bash
#SBATCH --job-name=dpo_eval_oneoff
#SBATCH --time=01:00:00
#SBATCH --mem=32GB
#SBATCH --partition=gpumedium
#SBATCH --gpus-per-node=rtx_pro_6000:1
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/slurm/%x-%j.out

# One-off DPO adapter evaluation. Runs the standard DPO eval config
# (configs/experiments/dpo/eval_dpo_condition.yaml) against the 5 OOD test sets
# plus annotated-intents test+val, identical to the pipeline's step 5 eval.
#
# Variables (override with --export when sbatch'ing):
#   ADAPTER_PATH  — local path to the DPO LoRA adapter directory
#                   (default: the Niclas-J-M k10-hard-b03-e3-s22 adapter)
#   OUTPUT_NAME   — short name used to nest outputs under
#                   data/safety_experiment/dpo/<OUTPUT_NAME>/
#                   (default: derived from ADAPTER_PATH basename, sans _adapter)
#   WANDB_RUN_NAME — optional W&B run name; if set, enables W&B logging.

set -euo pipefail

module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.8.0

source .venv/bin/activate

ADAPTER_PATH="${ADAPTER_PATH:-models/dpo/llama31-8b-k10-e-hard-b03-e3-s22_adapter}"

if [ -z "${OUTPUT_NAME:-}" ]; then
    OUTPUT_NAME="$(basename "${ADAPTER_PATH}")"
    OUTPUT_NAME="${OUTPUT_NAME%_adapter}"
fi

OUTPUT_DIR="data/safety_experiment/dpo/${OUTPUT_NAME}"

echo "Adapter:    ${ADAPTER_PATH}"
echo "Output dir: ${OUTPUT_DIR}"

CMD=(
    python scripts/eval_safety_classifier.py
    --config-name=dpo/eval_dpo_condition
    "finetuned.generation_adapter=${ADAPTER_PATH}"
    "paths.output_dir=${OUTPUT_DIR}"
    "lora.rank=16"
)

if [ -n "${WANDB_RUN_NAME:-}" ]; then
    CMD+=("wandb.enabled=true" "wandb.run_name=${WANDB_RUN_NAME}")
fi

"${CMD[@]}"
