#!/bin/bash
#SBATCH --job-name=eval_intent_gen
#SBATCH --time=00:20:00
#SBATCH --mem=16GB
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=a100:1
#SBATCH --output=logs/slurm/%x-%j.out

module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.8.0

source .venv/bin/activate

# Evaluate intent generation from best hyperparameter sweep model
python scripts/eval_intent_generation.py --config-name=eval_intent_generation
