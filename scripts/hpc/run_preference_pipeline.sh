#!/bin/bash
#SBATCH --job-name=preference-pipeline
#SBATCH --time=5:00:00
#SBATCH --mem=32GB
#SBATCH --partition=gpumedium
#SBATCH --gpus-per-node=a100:1
#SBATCH --output=logs/slurm/%x-%j.out

module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.8.0

source .venv/bin/activate
source ~/.wandb_secrets

echo "======================================================================"
echo " Preference-learning pipeline (DPO only)"
echo "  Steps: generate pairs → train DPO"
echo "         → eval SFT/DPO → compare"
echo "  Use --force-from=N to re-run from a specific step"
echo "======================================================================"

python scripts/run_preference_pipeline.py \
    --adapter-path           trained_models/causal/hyperparam_sweep/lr_5e-05_e_5_adapter \
    --base-model             meta-llama/Llama-3.1-8B-Instruct \
    --pairs-dir              data/dpo_pairs/train_t0.8 \
    --dpo-output-dir         trained_models/causal/llama-dpo \
    --contrastive-output-dir trained_models/causal/llama-contrastive \
    --sft-pred-dir           data/predictions/sft_baseline \
    --epochs                 1 \
    --learning-rate          5e-5 \
    --batch-size             2 \
    --grad-accum             8 \
    --dpo-beta               0.5 \
    --kl-beta                0.5 \
    --temperature            0.8 \
    --k-samples              5 \
    --seed                   22 \
    --wandb-project          dpo-contrastive-pipeline \
    --force-from             5 \
    --skip-steps             4,7

echo ""
echo "======================================================================"
echo " Done. Key outputs:"
echo "   trained_models/causal/llama-dpo_adapter/"
echo "   data/predictions/sft_baseline/test_predictions.jsonl"
echo "   trained_models/causal/llama-dpo/predictions/test_predictions.jsonl"
echo "   data/comparison/comparison_summary.json"
echo "======================================================================"
