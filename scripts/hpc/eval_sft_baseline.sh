#!/bin/bash
#SBATCH --job-name=eval-sft
#SBATCH --time=01:00:00
#SBATCH --mem=4GB
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=a100:1
#SBATCH --output=logs/slurm/%x-%j.out

module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.8.0

source .venv/bin/activate

python scripts/eval_sft_baseline.py \
    --adapter-path trained_models/causal/hyperparam_sweep/lr_0.0005_e_5_adapter \
    --base-model   meta-llama/Llama-3.1-8B-Instruct \
    --output-dir   data/predictions/sft_baseline \
    --test-size    0.1 \
    --val-size     0.1 \
    --seed         22

echo "SFT baseline predictions saved to data/predictions/sft_baseline/test_predictions.jsonl"
