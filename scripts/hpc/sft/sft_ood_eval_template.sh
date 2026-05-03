#!/bin/bash
#SBATCH --job-name=sft_ood_eval
#SBATCH --time=01:30:00
#SBATCH --mem=32GB
#SBATCH --partition=gpumedium
#SBATCH --gpus-per-node=rtx_pro_6000:1
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/slurm/%x-%j.out

module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.8.0

source .venv/bin/activate

# OOD validation for a single SFT adapter (one condition per job).
# Variables set by submit_sft_eval.py in ood-val mode:
#   GENERATION_ADAPTER      — adapter path (or "MISSING" for classification jobs)
#   CLASSIFICATION_ADAPTER  — adapter path (or "MISSING" for generation jobs)
#   EVAL_CONDITION          — finetuned_generation or finetuned_classification
#   OUTPUT_DIR              — per-adapter output directory
#   WANDB_RUN_NAME          — W&B run name
#   BASE_MODEL              — (optional) HF model name override; defaults to the
#                             eval config (Llama 3.1 8B). Set this to evaluate
#                             adapters trained on a different base, e.g.
#                             BASE_MODEL=google/gemma-3-12b-it for the Gemma sweep.

# Build optional model.name override only when BASE_MODEL is set.
MODEL_OVERRIDE=()
if [ -n "${BASE_MODEL:-}" ]; then
    MODEL_OVERRIDE=("model.name=${BASE_MODEL}")
fi

python scripts/eval_safety_classifier.py \
    --config-name=eval_ood_validation \
    "${MODEL_OVERRIDE[@]}" \
    "finetuned.generation_adapter=${GENERATION_ADAPTER}" \
    "finetuned.classification_adapter=${CLASSIFICATION_ADAPTER}" \
    "experiment.conditions=[${EVAL_CONDITION}]" \
    "paths.output_dir=${OUTPUT_DIR}" \
    "wandb.run_name=${WANDB_RUN_NAME}"
