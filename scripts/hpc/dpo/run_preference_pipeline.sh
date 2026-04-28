#!/bin/bash
#SBATCH --job-name=preference-pipeline
#SBATCH --time=5:00:00
#SBATCH --mem=32GB
#SBATCH --partition=gpumedium
#SBATCH --gpus-per-node=a100:1
#SBATCH --output=logs/slurm/%x-%j.out

module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/13.2.0

source .venv/bin/activate
source ~/.wandb_secrets

ATTN_IMPL="${ATTN_IMPL:-sdpa}"
WANDB_RUN="${WANDB_RUN:-hard-dpo-beta0.5-seed22}"

echo "======================================================================"
echo " Preference-learning pipeline (DPO only)"
echo "  Steps: generate pairs → balance pairs → train DPO → eval DPO"
echo "  Use --force-from=N to re-run from a specific step"
echo "======================================================================"

python scripts/dpo/run_preference_pipeline.py \
    --adapter-path           Jazhyc/llama-3.1-8b-sft-generation \
    --base-model             meta-llama/Llama-3.1-8B-Instruct \
    --pairs-dir              data/dpo_pairs/train_t0.8 \
    --dpo-output-dir         trained_models/causal/llama-dpo \
    --epochs                 1 \
    --learning-rate          5e-5 \
    --batch-size             2 \
    --grad-accum             8 \
    --dpo-beta               0.5 \
    --attn-implementation    "${ATTN_IMPL}" \
    --temperature            0.8 \
    --k-samples              5 \
    --seed                   22 \
    --wandb-project          dpo-pipeline \
    --wandb-run              "${WANDB_RUN}" \
    --force-from             3 \
    --skip-steps             4

echo ""
echo "======================================================================"
echo " Done. Key outputs:"
echo "   trained_models/causal/llama-dpo_adapter/"
echo "   trained_models/causal/llama-dpo/predictions/"
echo "======================================================================"
