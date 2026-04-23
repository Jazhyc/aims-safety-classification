#!/bin/bash
#SBATCH --job-name=distill_ood_eval
#SBATCH --time=01:30:00
#SBATCH --mem=32GB
#SBATCH --partition=gpumedium
#SBATCH --gpus-per-node=rtx_pro_6000:1
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/slurm/%x-%j.out

module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.8.0

source .venv/bin/activate

# OOD validation for a single distillation adapter (one condition per job).
# Variables set by submit_distillation_eval.py in ood-val mode:
#   STUDENT_MODEL   — HuggingFace model ID for the student (base model)
#   ADAPTER_PATH    — absolute path to the LoRA adapter
#   EVAL_CONDITION  — finetuned_reasoning_classification, finetuned_reasoning_synthetic_intent,
#                     or finetuned_reasoning_human_intent
#   OUTPUT_DIR      — per-adapter output directory
#   WANDB_RUN_NAME  — W&B run name

python scripts/eval_safety_classifier.py \
    --config-name=eval_ood_validation \
    "model.name=${STUDENT_MODEL}" \
    "finetuned.reasoning_classification_adapter=${ADAPTER_PATH}" \
    "finetuned.reasoning_human_intent_adapter=${ADAPTER_PATH}" \
    "finetuned.reasoning_synthetic_adapter=${ADAPTER_PATH}" \
    "lora.rank=32" \
    "experiment.conditions=[${EVAL_CONDITION}]" \
    "paths.output_dir=${OUTPUT_DIR}" \
    "wandb.run_name=${WANDB_RUN_NAME}"
