#!/bin/bash
#SBATCH --job-name=gptoss120b_baselines
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

# Evaluate GPT-OSS-120B on the baseline prompting conditions across all test datasets.
python scripts/eval_safety_classifier.py \
    --config-name=eval_safety_classifier \
    "model.name=openai/gpt-oss-120b" \
    "experiment.conditions=[vanilla_classification,vanilla_generation,vanilla_with_human_intent,zeroshot_cot_classification,zeroshot_cot_generation,zeroshot_cot_classification_with_intent]" \
    "wandb.run_name=gpt-oss-120b-baselines-test"
