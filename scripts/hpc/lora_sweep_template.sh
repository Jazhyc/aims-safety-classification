#!/bin/bash
#SBATCH --job-name=lora_sweep
#SBATCH --time=01:30:00
#SBATCH --mem=32GB
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=a100:1
#SBATCH --output=logs/slurm/%x-%j.out

module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.8.0

source .venv/bin/activate

# Run training with the specified LoRA configuration
# Environment variables set by submit script:
#   - LORA_RANK: LoRA rank (e.g., 8, 16, 32, 64)
#   - LORA_ALPHA: LoRA alpha (e.g., 16, 32, 64, 128)
#   - CONFIG_NAME: Base config name
#   - EXPERIMENT_NAME: Name for output directories

python scripts/baselines/train_generator.py \
    --config-name="${CONFIG_NAME}" \
    peft.lora_rank="${LORA_RANK}" \
    peft.lora_alpha="${LORA_ALPHA}" \
    paths.output_dir="data/train_results/${EXPERIMENT_NAME}" \
    paths.logs_dir="logs/${EXPERIMENT_NAME}" \
    paths.model_save_dir="models/sft/${EXPERIMENT_NAME}" \
    paths.predictions_dir="data/predictions/${EXPERIMENT_NAME}" \
    wandb.name="${EXPERIMENT_NAME}"