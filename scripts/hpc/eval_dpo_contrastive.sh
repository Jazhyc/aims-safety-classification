#!/bin/bash
#SBATCH --job-name=eval-dpo-ctr
#SBATCH --time=01:00:00
#SBATCH --mem=4GB
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=a100:1
#SBATCH --output=logs/slurm/%x-%j.out

module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.8.0

source .venv/bin/activate

BASE_MODEL="meta-llama/Llama-3.1-8B-Instruct"

echo "=== Evaluating DPO adapter ==="
python scripts/eval_sft_baseline.py \
    --adapter-path trained_models/causal/llama-dpo_adapter \
    --base-model   "${BASE_MODEL}" \
    --output-dir   trained_models/causal/llama-dpo/predictions

echo ""
echo "=== Evaluating Contrastive adapter (best checkpoint) ==="
python scripts/eval_sft_baseline.py \
    --adapter-path trained_models/causal/llama-contrastive_adapter_best \
    --base-model   "${BASE_MODEL}" \
    --output-dir   trained_models/causal/llama-contrastive/predictions

echo ""
echo "Done. Prediction files:"
echo "  trained_models/causal/llama-dpo/predictions/test_predictions.jsonl"
echo "  trained_models/causal/llama-contrastive/predictions/test_predictions.jsonl"
