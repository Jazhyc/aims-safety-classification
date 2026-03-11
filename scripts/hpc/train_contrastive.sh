#!/bin/bash
#SBATCH --job-name=train-contrastive
#SBATCH --time=06:00:00
#SBATCH --mem=48GB
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=a100:1
#SBATCH --output=logs/slurm/%x-%j.out

module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.8.0

source .venv/bin/activate

ADAPTER_PATH="trained_models/causal/hyperparam_sweep/lr_0.0005_e_5_adapter"
BASE_MODEL="meta-llama/Llama-3.1-8B-Instruct"
PAIRS_PATH="data/dpo_pairs/train_t0.8/contrastive_pairs.jsonl"
OUTPUT_DIR="trained_models/causal/llama-contrastive"

echo "======================================================================"
echo " Contrastive (InfoNCE + KL) fine-tuning"
echo "  Pairs    : ${PAIRS_PATH}"
echo "  SFT init : ${ADAPTER_PATH}"
echo "  Output   : ${OUTPUT_DIR}"
echo "======================================================================"

python scripts/train_contrastive.py \
    --pairs-path     "${PAIRS_PATH}" \
    --adapter-path   "${ADAPTER_PATH}" \
    --base-model     "${BASE_MODEL}" \
    --output-dir     "${OUTPUT_DIR}" \
    --epochs         3 \
    --learning-rate  5e-5 \
    --grad-accum     8 \
    --max-length     512 \
    --kl-beta        0.1 \
    --seed           22 \
    --wandb-project  intention-jailbreak \
    --wandb-run      contrastive-kl-beta0.1

echo ""
echo "======================================================================"
echo " Done. Outputs:"
echo "   ${OUTPUT_DIR}_adapter/           <- final contrastive LoRA adapter"
echo "   ${OUTPUT_DIR}_adapter_best/      <- best checkpoint by val loss"
echo "   ${OUTPUT_DIR}/predictions/       <- test predictions"
echo "   ${OUTPUT_DIR}/training_summary.json"
echo "======================================================================"
