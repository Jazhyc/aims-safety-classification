#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

# ── config ────────────────────────────────────────────────────────────────────

EXPERIMENT_NAME="grpo_n16_b16_temp12_ep10"
SYSTEM_PROMPT_FILE="$SCRIPT_DIR/system_prompt_reasoning_long.txt"
STEPS=(10 20 30 40 50 60 70 80 90 100)

mkdir -p "$SCRIPT_DIR/logs_evaluate"

# ── evaluate ──────────────────────────────────────────────────────────────────

for STEP in "${STEPS[@]}"; do
    MODEL_PATH="${CKPT_ROOT_PATH}/${EXPERIMENT_NAME}/huggingface_merged_step${STEP}"
    OUTPUT="${CKPT_ROOT_PATH}/${EXPERIMENT_NAME}/results_step${STEP}"

    sbatch -p $SLURM_PARTITION_TRAIN \
        --account $SLURM_ACCOUNT \
        --mem=100G \
        --cpus-per-task 64 \
        --gres gpu:1 \
        --nodes=1 \
        --ntasks-per-node=1 \
        --time=24:00:00 \
        --job-name=Evaluate \
        --output="$SCRIPT_DIR/logs_evaluate/evaluate_%j_%N.log" \
        --wrap="apptainer exec --nv \
            -B ${NLP_DIR}:${NLP_DIR} \
            --env HF_HOME=${HF_HOME} \
            --env TORCH_COMPILE_CACHE_DIR=${TORCH_COMPILE_CACHE_DIR} \
            ${SIF_PATH} python3 $SCRIPT_DIR/evaluate_safety_multi_vllm.py \
            --model_path '$MODEL_PATH' \
            --output_path '$OUTPUT' \
            --max_new_tokens 512 \
            --apply_chat_template \
            --system_prompt_file '$SYSTEM_PROMPT_FILE'"
done
