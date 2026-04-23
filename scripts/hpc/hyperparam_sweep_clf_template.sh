#!/bin/bash
#SBATCH --job-name=hyperparam_sweep_clf
#SBATCH --time=02:00:00
#SBATCH --mem=16GB
#SBATCH --partition=gpumedium
#SBATCH --gpus-per-node=rtx_pro_6000:1
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/slurm/%x-%j.out

module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.8.0

source .venv/bin/activate

# Run training with the specified hyperparameters (classification-only mode)
python scripts/baselines/train_generator.py \
    --config-name=llm_sweep \
    model.name="meta-llama/Llama-3.1-8B-Instruct" \
    training.learning_rate="${LEARNING_RATE}" \
    training.epochs="${EPOCHS}" \
    training.adam_beta1="${ADAM_BETA1}" \
    training.adam_beta2="${ADAM_BETA2}" \
    data.classification_only=true \
    data.predict_harm=false \
    generation.max_new_tokens=4 \
    paths.output_dir="data/train_results/hyperparam_sweep_clf/lr_${LEARNING_RATE}_e_${EPOCHS}" \
    paths.logs_dir="logs/hyperparam_sweep_clf/lr_${LEARNING_RATE}_e_${EPOCHS}" \
    paths.model_save_dir="models/sft/hyperparam_sweep_clf/lr_${LEARNING_RATE}_e_${EPOCHS}" \
    paths.predictions_dir="data/predictions/hyperparam_sweep_clf/lr_${LEARNING_RATE}_e_${EPOCHS}" \
    wandb.project="sft-hyperparam-sweep"
