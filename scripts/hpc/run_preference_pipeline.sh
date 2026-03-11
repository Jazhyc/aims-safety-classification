#!/bin/bash
#SBATCH --job-name=preference-pipeline
#SBATCH --time=16:00:00
#SBATCH --mem=48GB
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=a100:1
#SBATCH --output=logs/slurm/%x-%j.out

module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.8.0

source .venv/bin/activate

echo "======================================================================"
echo " Preference-learning pipeline"
echo "  Steps: generate pairs → train DPO → train contrastive"
echo "         → eval SFT/DPO/contrastive → compare"
echo "  Use --force-from=N to re-run from a specific step"
echo "======================================================================"

python scripts/run_preference_pipeline.py \
    --adapter-path           trained_models/causal/hyperparam_sweep/lr_0.0005_e_5_adapter \
    --base-model             meta-llama/Llama-3.1-8B-Instruct \
    --pairs-dir              data/dpo_pairs/train_t0.8 \
    --dpo-output-dir         trained_models/causal/llama-dpo \
    --contrastive-output-dir trained_models/causal/llama-contrastive \
    --sft-pred-dir           data/predictions/sft_baseline \
    --epochs                 3 \
    --learning-rate          5e-5 \
    --batch-size             2 \
    --grad-accum             8 \
    --dpo-beta               0.1 \
    --kl-beta                0.1 \
    --temperature            0.8 \
    --k-samples              5 \
    --seed                   22 \
    --wandb-project          intention-jailbreak \
    --force-from             3

echo ""
echo "======================================================================"
echo " Done. Key outputs:"
echo "   trained_models/causal/llama-dpo_adapter/"
echo "   trained_models/causal/llama-contrastive_adapter_best/"
echo "   data/predictions/sft_baseline/test_predictions.jsonl"
echo "   trained_models/causal/llama-dpo/predictions/test_predictions.jsonl"
echo "   trained_models/causal/llama-contrastive/predictions/test_predictions.jsonl"
echo "   data/comparison/comparison_summary.json"
echo "======================================================================"
