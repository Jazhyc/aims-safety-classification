#!/bin/bash
#SBATCH --job-name=ablation-distill-mix
#SBATCH --time=04:00:00
#SBATCH --mem=32GB
#SBATCH --gpus-per-node=rtx_pro_6000:1
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/slurm/%x-%j.out

module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.8.0

source .venv/bin/activate

# Variables (set by submit_broader_intent_mix.py):
#   STUDENT_MODEL         — HuggingFace student ID
#   LEARNING_RATE         — fixed at the per-combo best LR
#   EPOCHS                — ceiling (early stopping handles the rest)
#   CONDITIONS_LIST       — pipe-separated conditions (e.g. "human_intent|synthetic_intent");
#                          converted to comma-separated below before the Hydra list
#                          override. Pipe is used because SLURM's `--export` splits its
#                          value on commas, so a literal comma would truncate the var.
#   TRACES_TRAIN_PATH     — merged parsed_results.json (built locally, not via SLURM)
#   TRACES_VAL_PATH       — shared v7 validation traces
#   TRACES_TEST_PATH      — shared v7 test traces
#   RUN_NAME              — unique adapter directory name (without `_adapter` suffix)
#   ADAPTER_BASE_DIR      — e.g. models/distillation-ablations
#   TRAIN_RESULTS_DIR     — e.g. data/train_results/distillation-ablations
#   PREDICTIONS_DIR       — e.g. data/predictions/distillation-ablations
#   WANDB_PROJECT
#   WANDB_RUN_NAME

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
    "data.reasoning_traces_condition=[${CONDITIONS_LIST//|/,}]" \
    paths.output_dir="${TRAIN_RESULTS_DIR}/${RUN_NAME}" \
    paths.logs_dir="logs/distillation-ablations/${RUN_NAME}" \
    paths.model_save_dir="${ADAPTER_BASE_DIR}/${RUN_NAME}" \
    paths.predictions_dir="${PREDICTIONS_DIR}/${RUN_NAME}" \
    wandb.project="${WANDB_PROJECT}" \
    wandb.run_name="${WANDB_RUN_NAME}"
