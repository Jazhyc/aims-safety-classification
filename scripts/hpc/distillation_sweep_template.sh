#!/bin/bash
#SBATCH --job-name=distill-sweep
#SBATCH --time=04:00:00
#SBATCH --mem=32GB
#SBATCH --gpus-per-node=rtx_pro_6000:1
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/slurm/%x-%j.out

module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.8.0

source .venv/bin/activate

# Environment variables (set via --export by submit_distillation_sweep.py):
#   STUDENT_MODEL      — HuggingFace model ID for the student
#   LEARNING_RATE      — learning rate for this run
#   CONDITION          — "without_intent" or "with_intent"
#   TRACES_TRAIN_PATH  — path to train/parsed_results.json
#   TRACES_VAL_PATH    — path to validation/parsed_results.json
#   RUN_NAME           — unique identifier for filesystem paths
#   WANDB_PROJECT      — W&B project (one per student x teacher combo)
#   WANDB_RUN_NAME     — W&B run name within the project (condition + lr + version)

# Ensure output directories exist before the trainer / PEFT tries to write to them.
mkdir -p "models/distillation-sweep"
mkdir -p "logs/slurm"

python scripts/baselines/train_generator.py \
    --config-name=reasoning_distillation \
    model.name="${STUDENT_MODEL}" \
    training.learning_rate="${LEARNING_RATE}" \
    training.skip_vllm_eval=false \
    data.reasoning_traces_path="${TRACES_TRAIN_PATH}" \
    data.reasoning_traces_val_path="${TRACES_VAL_PATH}" \
    data.reasoning_traces_condition="${CONDITION}" \
    paths.output_dir="data/train_results/distillation-sweep/${RUN_NAME}" \
    paths.logs_dir="logs/distillation-sweep/${RUN_NAME}" \
    paths.model_save_dir="models/distillation-sweep/${RUN_NAME}" \
    paths.predictions_dir="data/predictions/distillation-sweep/${RUN_NAME}" \
    wandb.project="${WANDB_PROJECT}" \
    wandb.run_name="${WANDB_RUN_NAME}"
