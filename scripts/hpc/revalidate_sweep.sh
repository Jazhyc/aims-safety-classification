#!/bin/bash
#SBATCH --job-name=revalidate-sweep
#SBATCH --time=04:00:00
#SBATCH --mem=64GB
#SBATCH --partition=gpumedium
#SBATCH --gpus-per-node=rtx_pro_6000:1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/slurm/%x-%j.out

module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.8.0

source .venv/bin/activate

mkdir -p "logs/slurm"

python scripts/distillation/revalidate_sweep.py
