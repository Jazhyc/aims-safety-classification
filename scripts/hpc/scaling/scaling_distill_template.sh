#!/bin/bash
#SBATCH --job-name=scaling-distill
#SBATCH --time=24:00:00
#SBATCH --mem=32GB
#SBATCH --gpus-per-node=rtx_pro_6000:1
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/slurm/%x-%j.out

module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.8.0

source .venv/bin/activate

# Variables (set by submit_scaling.py):
#   STUDENT_MODEL      — HuggingFace student ID
#   LEARNING_RATE      — training LR
#   EPOCHS             — number of training epochs (default scaling: 1)
#   CONDITION          — distillation condition (synthetic_intent for scaling)
#   TRACES_TRAIN_PATH  — full WG train traces (~86k records)
#   TRACES_VAL_PATH    — shared v7 validation traces (informational only at this scale)
#   TRACES_TEST_PATH   — shared v7 test traces
#   RUN_NAME           — adapter directory base name (causal.py appends `_adapter`)
#   ADAPTER_BASE_DIR   — e.g. models/distillation-scaling
#   TRAIN_RESULTS_DIR  — e.g. data/train_results/distillation-scaling
#   PREDICTIONS_DIR    — e.g. data/predictions/distillation-scaling
#   WANDB_PROJECT      — single W&B project for the scaling experiment
#   WANDB_RUN_NAME     — distinguishes runs within the project

mkdir -p "${ADAPTER_BASE_DIR}"
mkdir -p "logs/slurm"

python scripts/baselines/train_generator.py \
    --config-name=reasoning_distillation \
    model.name="${STUDENT_MODEL}" \
    training.learning_rate="${LEARNING_RATE}" \
    training.epochs="${EPOCHS}" \
    training.skip_vllm_eval=false \
    data.reasoning_traces_path="${TRACES_TRAIN_PATH}" \
    data.reasoning_traces_val_path="${TRACES_VAL_PATH}" \
    data.reasoning_traces_test_path="${TRACES_TEST_PATH}" \
    data.reasoning_traces_condition="${CONDITION}" \
    paths.output_dir="${TRAIN_RESULTS_DIR}/${RUN_NAME}" \
    paths.logs_dir="logs/distillation-scaling/${RUN_NAME}" \
    paths.model_save_dir="${ADAPTER_BASE_DIR}/${RUN_NAME}" \
    paths.predictions_dir="${PREDICTIONS_DIR}/${RUN_NAME}" \
    wandb.project="${WANDB_PROJECT}" \
    wandb.run_name="${WANDB_RUN_NAME}"
