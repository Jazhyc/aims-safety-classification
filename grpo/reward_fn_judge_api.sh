#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

JUDGE_MODEL="google/gemma-3-27b-it"

mkdir -p "$SCRIPT_DIR/logs_judge"

sbatch -p $SLURM_PARTITION_JUDGE \
    --account $SLURM_ACCOUNT \
    --mem=200G \
    --cpus-per-task 48 \
    --gres gpu:2 \
    --time=24:00:00 \
    --job-name=JudgeService \
    --output="$SCRIPT_DIR/logs_judge/judge_%j_%N.log" \
    --wrap="apptainer exec --nv \
        -B ${NLP_DIR}:${NLP_DIR} \
        --env HF_HOME=${HF_HOME} \
        --env TORCH_COMPILE_CACHE_DIR=${TORCH_COMPILE_CACHE_DIR} \
        ${SIF_PATH} python3 $SCRIPT_DIR/reward_fn_judge_api_async.py \
        --model ${JUDGE_MODEL}"
