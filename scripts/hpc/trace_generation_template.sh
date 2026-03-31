#!/bin/bash
#SBATCH --job-name=trace-gen
#SBATCH --time=04:00:00
#SBATCH --mem=32GB
#SBATCH --partition=gpumedium
#SBATCH --gpus-per-node=rtx_pro_6000:1
#SBATCH --output=logs/slurm/%x-%j.out

module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.8.0

source .venv/bin/activate

# Environment variables (set via --export by submit_trace_generation.py):
#   MODEL_NAME     — HuggingFace model ID for the teacher
#   OUTPUT_DIR     — base directory for saving traces (e.g. data/reasoning_traces_v6)
#   THINKING_MODE  — "true" or "false"; set true for models with native thinking tokens

mkdir -p "logs/slurm"

python scripts/distillation/generate_reasoning_traces.py \
    --config-name=reasoning_traces \
    model.name="${MODEL_NAME}" \
    model.backend=vllm \
    thinking_mode="${THINKING_MODE}" \
    paths.output_dir="${OUTPUT_DIR}"
