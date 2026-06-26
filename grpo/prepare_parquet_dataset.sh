#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

mkdir -p logs

sbatch -p $SLURM_PARTITION_DATA \
    --account $SLURM_ACCOUNT \
    --gres gpu:0 \
    --mem=16G \
    --job-name=Safety \
    --output=logs/prepare_parquet_dataset_%j.log \
    --wrap="apptainer exec --nv \
        -B ${NLP_DIR}:${NLP_DIR} \
        --env HF_HOME=${HF_HOME} \
        --env TORCH_COMPILE_CACHE_DIR=${TORCH_COMPILE_CACHE_DIR} \
        ${SIF_PATH} python3 $SCRIPT_DIR/prepare_parquet_dataset_combined_system.py \
        --system_prompt $SCRIPT_DIR/system_prompt_reasoning_long.txt \
        --output_dir $SCRIPT_DIR/data/aims_safety_intents"
