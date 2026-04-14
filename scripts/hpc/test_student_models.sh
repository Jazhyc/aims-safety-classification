#!/bin/bash
# Quick smoke test for all student models in the distillation sweep.
# Runs 10 training steps per model with W&B disabled and vLLM eval skipped.
# Intended for interactive use on an RTX 6000 Pro:
#
#   srun --gpus-per-node=rtx_pro_6000:1 --mem=32GB --time=01:00:00 --pty bash
#   cd /scratch/s4626451/intention-jailbreak
#   bash scripts/hpc/test_student_models.sh

set -e

module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.8.0
source .venv/bin/activate

# Use a single teacher's traces (no_intent, simplest condition)
TRACES_TRAIN="data/reasoning_traces_v7/RedHatAI-gemma-3-27b-it-quantized.w4a16/train/parsed_results.json"
TRACES_VAL="data/reasoning_traces_v7/RedHatAI-gemma-3-27b-it-quantized.w4a16/validation/parsed_results.json"

STUDENT_MODELS=(
    # "meta-llama/Llama-3.1-8B-Instruct"
    # "mistralai/Ministral-3-14B-Reasoning-2512"
    "google/gemma-4-E4B"
)

PASS=()
FAIL=()

for MODEL in "${STUDENT_MODELS[@]}"; do
    SLUG="${MODEL//\//-}"
    echo ""
    echo "================================================================"
    echo "Testing: ${MODEL}"
    echo "================================================================"

    if python scripts/baselines/train_generator.py \
        --config-name=reasoning_distillation \
        model.name="${MODEL}" \
        +training.max_steps=10 \
        training.skip_vllm_eval=true \
        data.reasoning_traces_path="${TRACES_TRAIN}" \
        data.reasoning_traces_val_path="${TRACES_VAL}" \
        data.reasoning_traces_condition="no_intent" \
        paths.output_dir="data/train_results/smoke-test/${SLUG}" \
        paths.logs_dir="logs/smoke-test/${SLUG}" \
        paths.model_save_dir="models/smoke-test/${SLUG}" \
        paths.predictions_dir="data/predictions/smoke-test/${SLUG}" \
        wandb.enabled=false; then
        echo "  PASS: ${MODEL}"
        PASS+=("${MODEL}")
    else
        echo "  FAIL: ${MODEL}"
        FAIL+=("${MODEL}")
    fi
done

echo ""
echo "================================================================"
echo "Results"
echo "================================================================"
echo "PASS (${#PASS[@]}):"
for m in "${PASS[@]}"; do echo "  + ${m}"; done
echo "FAIL (${#FAIL[@]}):"
for m in "${FAIL[@]}"; do echo "  - ${m}"; done
