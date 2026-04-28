#!/bin/bash
#SBATCH --job-name=gen-dpo-pairs
#SBATCH --time=04:00:00
#SBATCH --mem=32GB
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=a100:1
#SBATCH --output=logs/slurm/%x-%j.out

module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/13.2.0

source .venv/bin/activate

ADAPTER_PATH="Jazhyc/llama-3.1-8b-sft-generation"
BASE_MODEL="meta-llama/Llama-3.1-8B-Instruct"
OUTPUT_DIR="data/dpo_pairs/train_t0.8"

echo "======================================================================"
echo " Generating DPO preference pairs from the TRAIN split"
echo "  Adapter   : ${ADAPTER_PATH}"
echo "  Base model: ${BASE_MODEL}"
echo "  Output    : ${OUTPUT_DIR}"
echo "======================================================================"

python scripts/dpo/generate_dpo_pairs.py \
    --adapter-path   "${ADAPTER_PATH}" \
    --base-model     "${BASE_MODEL}" \
    --output-dir     "${OUTPUT_DIR}" \
    --num-samples    5 \
    --temperature    0.8 \
    --top-p          0.95 \
    --max-new-tokens 64

echo ""
echo "======================================================================"
echo " Done. Outputs:"
echo "   ${OUTPUT_DIR}/parsed_samples.jsonl    <- raw k samples per prompt"
echo "   ${OUTPUT_DIR}/dpo_pairs.jsonl         <- one (chosen, rejected) per prompt"
echo "   ${OUTPUT_DIR}/summary.json            <- run statistics"
echo "======================================================================"
