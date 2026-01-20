#!/bin/bash
#SBATCH --job-name=hyperparam_sweep
#SBATCH --time=01:00:00
#SBATCH --mem=16GB
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=a100:1
#SBATCH --output=logs/slurm/%x-%j.out

module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.8.0

source .venv/bin/activate

# Run training with the specified hyperparameters
python scripts/train_generator.py \
    --config-name=llm_sweep \
    model.name="meta-llama/Llama-3.1-8B-Instruct" \
    training.learning_rate="${LEARNING_RATE}" \
    training.epochs="${EPOCHS}" \
    training.adam_beta1="${ADAM_BETA1}" \
    training.adam_beta2="${ADAM_BETA2}" \
    paths.output_dir="train_results/causal/hyperparam_sweep/lr_${LEARNING_RATE}_e_${EPOCHS}" \
    paths.logs_dir="logs/causal/hyperparam_sweep/lr_${LEARNING_RATE}_e_${EPOCHS}" \
    paths.model_save_dir="trained_models/causal/hyperparam_sweep/lr_${LEARNING_RATE}_e_${EPOCHS}" \
    paths.predictions_dir="data/predictions/causal/hyperparam_sweep/lr_${LEARNING_RATE}_e_${EPOCHS}" \
    wandb.project="intention-jailbreak-hyperparam-sweep"
