#!/bin/bash
#SBATCH --job-name=hyperparam_sweep_gemma
#SBATCH --time=04:00:00
#SBATCH --mem=16GB
#SBATCH --partition=gpumedium
#SBATCH --gpus-per-node=rtx_pro_6000:1
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/slurm/%x-%j.out

module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.8.0

source .venv/bin/activate

# Adapter leaf is prefixed with `gemma3-12b_` so the W&B artifact registry
# (keyed on `Path(model_save_dir + "_adapter").name`) does not collide with
# the Llama 3.1 8B sweep, which uses `lr_${LR}_e_${E}_adapter`.
python scripts/baselines/train_generator.py \
    --config-name=llm_sweep \
    model.name="google/gemma-3-12b-it" \
    training.learning_rate="${LEARNING_RATE}" \
    training.epochs="${EPOCHS}" \
    training.adam_beta1="${ADAM_BETA1}" \
    training.adam_beta2="${ADAM_BETA2}" \
    paths.output_dir="data/train_results/hyperparam_sweep_gemma/lr_${LEARNING_RATE}_e_${EPOCHS}" \
    paths.logs_dir="logs/hyperparam_sweep_gemma/lr_${LEARNING_RATE}_e_${EPOCHS}" \
    paths.model_save_dir="models/sft/hyperparam_sweep_gemma/gemma3-12b_lr_${LEARNING_RATE}_e_${EPOCHS}" \
    paths.predictions_dir="data/predictions/hyperparam_sweep_gemma/lr_${LEARNING_RATE}_e_${EPOCHS}" \
    wandb.project="sft-hyperparam-sweep-gemma"
