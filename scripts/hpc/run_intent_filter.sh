#!/bin/bash
#SBATCH --job-name=intent-filter
#SBATCH --time=01:00:00
#SBATCH --mem=32GB
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=a100:1
#SBATCH --output=logs/slurm/%x-%j.out

module purge
module load CUDA/12.8.0
module load Python/3.12.3-GCCcore-13.3.0

cd "$HOME/repos/intent-gen"
source .venv/bin/activate

PROJECTS_DIR="${PROJECTS_DIR:-/scratch/s4351495}"
export HF_HOME="${HF_HOME:-${PROJECTS_DIR}/huggingface_cache}"
# Do NOT set HF_HUB_OFFLINE here — the judge model may need to be downloaded
# if it is not already in the local cache.
export VLLM_USE_V1=0

# Override these via environment when submitting, e.g.:
#   JUDGE_MODEL="RedHatAI/gemma-3-27b-it-quantized.w4a16" \
#   OUTPUT_DIR="data/dpo_pairs/train_t0.8_filtered_gemma27b" \
#   sbatch scripts/hpc/run_intent_filter.sh
JUDGE_MODEL="${JUDGE_MODEL:-RedHatAI/gemma-3-27b-it-quantized.w4a16}"
OUTPUT_DIR="${OUTPUT_DIR:-data/dpo_pairs/train_t0.8_intent_filtered}"

echo "Judge model : ${JUDGE_MODEL}"
echo "Output dir  : ${OUTPUT_DIR}"

python scripts/generate_dpo_pairs.py \
    --from-samples data/dpo_pairs/train_t0.8/parsed_samples.jsonl \
    --adapter-path "${PROJECTS_DIR}/trained_models/causal/hyperparam_sweep/lr_5e-05_e_5_adapter" \
    --base-model   meta-llama/Llama-3.1-8B-Instruct \
    --judge-model  "${JUDGE_MODEL}" \
    --intent-filter \
    --output-dir   "${OUTPUT_DIR}"
