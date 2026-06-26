#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

# ── config ────────────────────────────────────────────────────────────────────

EXPERIMENT_NAME="grpo_n16_b16_temp12_ep10"
BASE_DIR="${CKPT_ROOT_PATH}/${EXPERIMENT_NAME}"

# Checkpoints to merge — must match the global_step_* directories saved during training
# (determined by trainer.save_freq and the number of steps per epoch)
STEPS=(10 20 30 40 50 60 70 80 90 100)

mkdir -p "$SCRIPT_DIR/logs_merge"

# ── merge ─────────────────────────────────────────────────────────────────────

for STEP in "${STEPS[@]}"; do
    CKPT_PATH="${BASE_DIR}/global_step_${STEP}/actor"
    OUTPUT_PATH="${BASE_DIR}/huggingface_merged_step${STEP}"
    mkdir -p "$OUTPUT_PATH"

    sbatch -p $SLURM_PARTITION_TRAIN \
        --account $SLURM_ACCOUNT \
        --mem=100G \
        --cpus-per-task 64 \
        --gres gpu:1 \
        --job-name=Merge \
        --output="$SCRIPT_DIR/logs_merge/merge_%j_%N.log" \
        --wrap="apptainer exec --nv \
            -B ${NLP_DIR}:${NLP_DIR} \
            --env HF_HOME=${HF_HOME} \
            ${SIF_PATH} python3 -m verl.model_merger merge \
            --backend fsdp \
            --local_dir '$CKPT_PATH' \
            --target_dir '$OUTPUT_PATH'"
done
