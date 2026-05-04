#!/bin/bash
#SBATCH --job-name=scaling-test-eval
#SBATCH --time=04:00:00
#SBATCH --mem=32GB
#SBATCH --partition=gpumedium
#SBATCH --gpus-per-node=rtx_pro_6000:1
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/slurm/%x-%j.out

module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.8.0

source .venv/bin/activate

# Variables (set by submit_scaling.py --mode test):
#   STUDENT_MODEL   — student base model
#   ADAPTER_PATH    — adapter dir for the trained scaling run
#   EVAL_CONDITION  — finetuned_reasoning_synthetic_intent
#   EVAL_CONFIG     — Hydra config name (default: eval_distillation)
#   OUTPUT_DIR      — per-adapter output dir
#   WANDB_RUN_NAME  — W&B run name

python scripts/eval_safety_classifier.py \
    --config-name=${EVAL_CONFIG:-eval_distillation} \
    "model.name=${STUDENT_MODEL}" \
    "finetuned.reasoning_classification_adapter=${ADAPTER_PATH}" \
    "finetuned.reasoning_human_intent_adapter=${ADAPTER_PATH}" \
    "finetuned.reasoning_synthetic_adapter=${ADAPTER_PATH}" \
    "experiment.conditions=[${EVAL_CONDITION}]" \
    "paths.output_dir=${OUTPUT_DIR}" \
    "wandb.run_name=${WANDB_RUN_NAME}"
