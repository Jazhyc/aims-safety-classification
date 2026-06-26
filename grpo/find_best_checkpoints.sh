#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

# ── config ────────────────────────────────────────────────────────────────────

EXPERIMENT_NAME="grpo_n16_b16_temp12_ep10"
RESULTS_DIR="${CKPT_ROOT_PATH}/${EXPERIMENT_NAME}"

mkdir -p "$SCRIPT_DIR/logs_find_best"

# ── find best ─────────────────────────────────────────────────────────────────

sbatch -p $SLURM_PARTITION_DATA \
    --account $SLURM_ACCOUNT \
    --mem=16G \
    --cpus-per-task 4 \
    --gres gpu:0 \
    --time=1:00:00 \
    --job-name=FindBest \
    --output="$SCRIPT_DIR/logs_find_best/find_%j_%N.log" \
    --wrap="apptainer exec \
        -B ${NLP_DIR}:${NLP_DIR} \
        ${SIF_PATH} python3 $SCRIPT_DIR/find_best_checkpoints.py \
        --model_folder '$RESULTS_DIR'"
