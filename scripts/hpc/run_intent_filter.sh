#!/bin/bash
#SBATCH --job-name=intent-filter
#SBATCH --time=01:00:00
#SBATCH --mem=32GB
#SBATCH --partition=gpushort
#SBATCH --gpus-per-node=a100:1
#SBATCH --output=logs/slurm/%x-%j.out

module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.8.0

source .venv/bin/activate
source ~/.wandb_secrets

JUDGE_MODEL="${JUDGE_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-data/dpo_pairs/train_t0.8_intent_filtered}"

echo "Judge model : ${JUDGE_MODEL}"
echo "Output dir  : ${OUTPUT_DIR}"

python scripts/generate_dpo_pairs.py \
    --from-samples data/dpo_pairs/train_t0.8/parsed_samples.jsonl \
    --adapter-path trained_models/causal/hyperparam_sweep/lr_5e-05_e_5_adapter \
    --base-model   meta-llama/Llama-3.1-8B-Instruct \
    --judge-model  "${JUDGE_MODEL}" \
    --intent-filter \
    --output-dir   "${OUTPUT_DIR}"
