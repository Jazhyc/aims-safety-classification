#!/bin/bash
#SBATCH --job-name=safety_exp_gemma3_12b
#SBATCH --time=04:00:00
#SBATCH --mem=16GB
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=rtx_pro_6000:1
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/slurm/%x-%j.out

module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.8.0

source .venv/bin/activate

# Benchmark google/gemma-3-12b-it on the prompting conditions defined in
# eval_safety_classifier.yaml across all eval datasets.
python scripts/eval_safety_classifier.py --config-name=eval_safety_classifier \
    model.name=google/gemma-3-12b-it
