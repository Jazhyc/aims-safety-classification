#!/bin/bash
#SBATCH --job-name=safety_exp
#SBATCH --time=04:00:00
#SBATCH --mem=16GB
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=rtx_pro_6000:1
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/slurm/%x-%j.out

module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.8.0

source .venv/bin/activate

# Run safety experiment with all conditions including baselines from other papers
python scripts/eval_safety_classifier.py --config-name=eval_safety_classifier
