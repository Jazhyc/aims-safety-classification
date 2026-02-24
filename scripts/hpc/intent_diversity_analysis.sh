#!/bin/bash
#SBATCH --job-name=intent_diversity
#SBATCH --time=04:00:00
#SBATCH --mem=32GB
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=a100:1
#SBATCH --output=logs/slurm/%x-%j.out

module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.8.0

source .venv/bin/activate

# ---------------------------------------------------------------------------
# Intent Diversity Analysis — Temperature Sweep
#
# Runs the diversity analysis 5 times with increasing temperatures to
# characterise the diversity-quality tradeoff for DPO pair construction.
#
# Temperature intuition:
#   0.3  → near-greedy; expect very low diversity, almost no DPO pairs
#   0.6  → moderate diversity; outputs still clean and structured
#   0.8  → standard DPO sampling point from the literature
#   1.0  → high diversity; watch parse_failures in summary.json
#   1.2  → stress test; some outputs may lose the Intent/Harm format
# ---------------------------------------------------------------------------

ADAPTER_PATH="trained_models/causal/hyperparam_sweep/lr_0.0005_e_5_adapter"
BASE_MODEL="meta-llama/Llama-3.1-8B-Instruct"
TEMPERATURES=(0.3 0.6 0.8 1.0 1.2)

for TEMP in "${TEMPERATURES[@]}"; do
    echo ""
    echo "=========================================="
    echo " Running temperature = ${TEMP}"
    echo "=========================================="

    python scripts/intent_diversity_analysis.py \
        --adapter-path "${ADAPTER_PATH}" \
        --base-model "${BASE_MODEL}" \
        --output-dir "data/diversity_analysis/t${TEMP}" \
        --num-samples 5 \
        --temperature "${TEMP}" \
        --top-p 0.95 \
        --max-new-tokens 64
done

echo ""
echo "=========================================="
echo " All 5 temperature runs complete."
echo " Results in: data/diversity_analysis/"
echo " Next: open notebooks/diversity_analysis.ipynb"
echo "=========================================="