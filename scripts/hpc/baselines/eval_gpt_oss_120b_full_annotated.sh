#!/bin/bash
#SBATCH --job-name=gptoss120b_full_annotated
#SBATCH --time=04:00:00
#SBATCH --mem=32GB
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=rtx_pro_6000:1
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/slurm/%x-%j.out

module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.8.0

source .venv/bin/activate

mkdir -p "logs/slurm"

# Re-run synthetic harm labeler with gpt-oss-120b over the FULL annotated set
# (n=1,724) so the human–model row in tab:agreement_kappa matches the
# human–WildGuardMix sample size. vanilla_generation only — captures both
# generated_intent (for cosine sim) and predicted_harm (for Cohen's κ).
python scripts/eval_safety_classifier.py \
    --config-name=eval_safety_classifier_full_annotated
