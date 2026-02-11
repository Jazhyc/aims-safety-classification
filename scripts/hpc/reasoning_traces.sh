#!/bin/bash
#SBATCH --job-name=reasoning_traces
#SBATCH --time=02:00:00
#SBATCH --mem=16GB
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=a100:1
#SBATCH --output=logs/slurm/%x-%j.out

module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.8.0

source .venv/bin/activate

# Generate reasoning traces from WildGuard train set
python scripts/generate_reasoning_traces.py --config-name=reasoning_traces
