#!/bin/bash
#SBATCH --job-name=llm_gen_${MODEL_NAME}
#SBATCH --time=04:00:00
#SBATCH --mem=16GB
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.8.0

source .venv/bin/activate

# Run training with the specified model
python scripts/train_generator.py \
    --config-name=llm_sweep \
    model.name="${MODEL_HF_PATH}" \
    paths.output_dir="train_results/causal/${MODEL_NAME}" \
    paths.logs_dir="logs/causal/${MODEL_NAME}" \
    paths.model_save_dir="trained_models/causal/${MODEL_NAME}" \
    paths.predictions_dir="data/predictions/causal/${MODEL_NAME}" \
    wandb.project="intention-jailbreak-llm-sweep" \
    wandb.name="${MODEL_NAME}-generator"
