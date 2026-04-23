#!/bin/bash
#SBATCH --job-name=intent_diversity
#SBATCH --time=06:00:00
#SBATCH --mem=32GB
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=a100:1
#SBATCH --output=logs/slurm/%x-%j.out

module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.8.0

source .venv/bin/activate

# ---------------------------------------------------------------------------
# Intent Diversity Analysis — Temperature Sweep + Harm-Source Comparison
#
# Three passes over the same 5 temperatures:
#
#   Pass 1 (generated): Original mode — harm label parsed from the same
#     LLM generation as the intent.  Also writes parsed_samples.jsonl so
#     subsequent passes can skip the expensive generation step.
#
#   Pass 2 (llm_t0): Loads parsed_samples.jsonl from Pass 1 and reclassifies
#     each intent using the same LLM at temperature=0 (greedy / deterministic).
#     Decouples intent diversity from harm-label noise.
#
#   Pass 3 (modernbert): Loads parsed_samples.jsonl from Pass 1 and classifies
#     each intent with the trained ModernBERT intent classifier.  vLLM is NOT
#     loaded for this pass.
#
# ---------------------------------------------------------------------------

ADAPTER_PATH="trained_models/causal/hyperparam_sweep/lr_5e-05_e_5_adapter"
BASE_MODEL="meta-llama/Llama-3.1-8B-Instruct"
MODERNBERT_PATH="trained_models/bert_intent/ModernBERT-base"
TEMPERATURES=(0.0 0.3 0.6 0.8 1.0 1.2)

# ===========================================================================
# Pass 1: Generate k intents at each temperature (harm label from generation)
#         Saves parsed_samples.jsonl for reuse in passes 2 and 3.
# ===========================================================================
echo ""
echo "######################################################################"
echo "# Pass 1: generation (harm label from LLM output)"
echo "######################################################################"

for TEMP in "${TEMPERATURES[@]}"; do
    echo ""
    echo "=========================================="
    echo " [Pass 1] temperature = ${TEMP}"
    echo "=========================================="

    # t=0 is greedy/deterministic — k=1 is sufficient, extra samples are identical
    if [ "${TEMP}" = "0.0" ]; then
        K=1
        TOP_P=1.0
    else
        K=5
        TOP_P=0.95
    fi

    python scripts/dpo/intent_diversity_analysis.py \
        --adapter-path "${ADAPTER_PATH}" \
        --base-model "${BASE_MODEL}" \
        --output-dir "data/diversity_analysis/t${TEMP}" \
        --num-samples "${K}" \
        --temperature "${TEMP}" \
        --top-p "${TOP_P}" \
        --max-new-tokens 64 \
        --harm-source generated
done

# ===========================================================================
# Pass 2: Reclassify with LLM at temperature=0
#         Loads parsed_samples.jsonl
# ===========================================================================
echo ""
echo "######################################################################"
echo "# Pass 2: llm_t0 (harm label from LLM at temperature=0)"
echo "######################################################################"

for TEMP in "${TEMPERATURES[@]}"; do
    echo ""
    echo "=========================================="
    echo " [Pass 2] temperature = ${TEMP}"
    echo "=========================================="

    python scripts/dpo/intent_diversity_analysis.py \
        --adapter-path "${ADAPTER_PATH}" \
        --base-model "${BASE_MODEL}" \
        --from-samples "data/diversity_analysis/t${TEMP}/parsed_samples.jsonl" \
        --harm-source llm_t0 \
        --output-dir "data/diversity_analysis_llm_t0/t${TEMP}" \
        --temperature "${TEMP}"
done

# ===========================================================================
# Pass 3: Reclassify with ModernBERT intent classifier
#         Loads parsed_samples.jsonl
# ===========================================================================
echo ""
echo "######################################################################"
echo "# Pass 3: modernbert (harm label from trained ModernBERT classifier)"
echo "######################################################################"

for TEMP in "${TEMPERATURES[@]}"; do
    echo ""
    echo "=========================================="
    echo " [Pass 3] temperature = ${TEMP}"
    echo "=========================================="

    python scripts/dpo/intent_diversity_analysis.py \
        --from-samples "data/diversity_analysis/t${TEMP}/parsed_samples.jsonl" \
        --harm-source modernbert \
        --modernbert-path "${MODERNBERT_PATH}" \
        --output-dir "data/diversity_analysis_modernbert/t${TEMP}" \
        --temperature "${TEMP}"
done

echo ""
echo "######################################################################"
echo "# All passes complete."
echo "#"
echo "# Results:"
echo "#   data/diversity_analysis/            <- Pass 1 (generated)"
echo "#   data/diversity_analysis_llm_t0/     <- Pass 2 (llm_t0)"
echo "#   data/diversity_analysis_modernbert/ <- Pass 3 (modernbert)"
echo "#"
echo "######################################################################"
