#!/bin/bash
# Quick OOM smoke test for all student models in the distillation sweep.
#
# For each student, tries (per_device_batch_size, gradient_accumulation) pairs
# in descending order of per-device batch size — all keep effective batch = 32.
# Stops at the first pair that completes 10 training steps without OOM and
# records it. Skips vLLM eval to isolate the training step.
#
# Usage:
#   srun --gpus-per-node=rtx_pro_6000:1 --mem=32GB --time=01:00:00 --pty bash
#   cd /scratch/s4626451/intention-jailbreak
#   bash scripts/hpc/distillation/test_student_models.sh

set -u

module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.8.0
source .venv/bin/activate

# Use a single teacher's traces (no_intent, simplest condition / shortest output)
TRACES_TRAIN="data/reasoning_traces_v7/RedHatAI-gemma-3-27b-it-quantized.w4a16/train/parsed_results.json"
TRACES_VAL="data/reasoning_traces_v7/RedHatAI-gemma-3-27b-it-quantized.w4a16/validation/parsed_results.json"

STUDENT_MODELS=(
    "meta-llama/Llama-3.1-8B-Instruct"
    "Qwen/Qwen3-8B"
    "google/gemma-3-12b-it"
    "mistralai/Ministral-3-14B-Reasoning-2512"
)

# (per_device_batch_size, gradient_accumulation) pairs — all give effective=32.
# Tried in this order; first one that fits wins.
BATCH_PAIRS=(
    "32 1"
    "16 2"
    "8 4"
    "4 8"
    "2 16"
)

declare -A RESULTS  # model -> "per_device x grad_accum" or "OOM"

for MODEL in "${STUDENT_MODELS[@]}"; do
    SLUG="${MODEL//\//-}"
    echo ""
    echo "================================================================"
    echo "Testing: ${MODEL}"
    echo "================================================================"

    BEST=""
    for PAIR in "${BATCH_PAIRS[@]}"; do
        read -r PER_DEV GRAD_ACC <<< "${PAIR}"
        echo ""
        echo "  -> per_device=${PER_DEV}, grad_accum=${GRAD_ACC}  (effective=$((PER_DEV * GRAD_ACC)))"

        if python scripts/baselines/train_generator.py \
            --config-name=reasoning_distillation \
            model.name="${MODEL}" \
            +training.max_steps=10 \
            training.batch_size="${PER_DEV}" \
            training.gradient_accumulation="${GRAD_ACC}" \
            training.skip_vllm_eval=true \
            data.reasoning_traces_path="${TRACES_TRAIN}" \
            data.reasoning_traces_val_path="${TRACES_VAL}" \
            data.reasoning_traces_condition="no_intent" \
            paths.output_dir="data/train_results/smoke-test/${SLUG}" \
            paths.logs_dir="logs/smoke-test/${SLUG}" \
            paths.model_save_dir="models/smoke-test/${SLUG}" \
            paths.predictions_dir="data/predictions/smoke-test/${SLUG}" \
            wandb.enabled=false; then
            echo "  PASS at per_device=${PER_DEV}, grad_accum=${GRAD_ACC}"
            BEST="${PER_DEV}x${GRAD_ACC}"
            break
        else
            echo "  FAIL at per_device=${PER_DEV}, grad_accum=${GRAD_ACC}"
        fi
    done

    if [[ -z "${BEST}" ]]; then
        RESULTS["${MODEL}"]="OOM at all pairs"
    else
        RESULTS["${MODEL}"]="${BEST}"
    fi
done

echo ""
echo "================================================================"
echo "Largest per-device batch fitting effective=32 (per student)"
echo "================================================================"
for MODEL in "${STUDENT_MODELS[@]}"; do
    printf "  %-50s %s\n" "${MODEL}" "${RESULTS[${MODEL}]}"
done
