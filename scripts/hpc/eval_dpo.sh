#!/bin/bash
#SBATCH --job-name=eval-dpo
#SBATCH --time=01:00:00
#SBATCH --mem=32GB
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=a100:1
#SBATCH --output=logs/slurm/%x-%j.out

module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.8.0

source .venv/bin/activate

echo "======================================================================"
echo " Evaluate DPO adapter + compare with SFT baseline"
echo "======================================================================"

python scripts/run_preference_pipeline.py \
    --adapter-path           trained_models/causal/hyperparam_sweep/lr_0.0005_e_5_adapter \
    --base-model             meta-llama/Llama-3.1-8B-Instruct \
    --pairs-dir              data/dpo_pairs/train_t0.8 \
    --dpo-output-dir         trained_models/causal/llama-dpo \
    --contrastive-output-dir trained_models/causal/llama-contrastive \
    --sft-pred-dir           data/predictions/sft_baseline \
    --seed                   22 \
    --skip-steps             1,2,3,4,6 \
    --force-from             5 \
    --wandb-project          intention-jailbreak

echo ""
echo "======================================================================"
echo " Done. Key outputs:"
echo "   trained_models/causal/llama-dpo/predictions/test_predictions.jsonl"
echo "   data/comparison/comparison_summary.json"
echo "======================================================================"
