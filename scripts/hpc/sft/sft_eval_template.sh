#!/bin/bash
#SBATCH --job-name=sft_eval
#SBATCH --time=04:00:00
#SBATCH --mem=32GB
#SBATCH --partition=gpumedium
#SBATCH --gpus-per-node=rtx_pro_6000:1
#SBATCH --output=logs/slurm/%x-%j.out

module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.8.0

source .venv/bin/activate

# Evaluate best SFT adapters from each hyperparam sweep on all test datasets.
# GENERATION_ADAPTER and CLASSIFICATION_ADAPTER are set by submit_sft_eval.py
# based on highest validation harm F1 across sweep runs.
python scripts/eval_safety_classifier.py \
    --config-name=${CONFIG_NAME:-eval_sft_baselines} \
    "finetuned.generation_adapter=${GENERATION_ADAPTER}" \
    "finetuned.classification_adapter=${CLASSIFICATION_ADAPTER}"
