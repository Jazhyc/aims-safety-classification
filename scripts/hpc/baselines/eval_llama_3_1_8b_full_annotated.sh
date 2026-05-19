#!/bin/bash
#SBATCH --job-name=llama318b_full_annotated
#SBATCH --time=02:00:00
#SBATCH --mem=32GB
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=rtx_pro_6000:1
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/slurm/%x-%j.out

module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.8.0

source .venv/bin/activate

mkdir -p "logs/slurm"

# Synthetic harm labeler with Llama-3.1-8B-Instruct over the FULL annotated
# set (n=1,724) so we can add Human–Llama and Llama–WildGuardMix rows to
# tab:agreement_kappa at matched sample size with the existing gpt-oss-120b
# row. vanilla_generation only — captures both generated_intent (cosine) and
# predicted_harm (κ).
python scripts/eval_safety_classifier.py \
    --config-name=eval_safety_classifier_full_annotated_llama
