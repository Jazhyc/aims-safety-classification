#!/bin/bash
#SBATCH --job-name=sft_eval
#SBATCH --time=04:00:00
#SBATCH --mem=32GB
#SBATCH --partition=gpumedium
#SBATCH --gpus-per-node=rtx_pro_6000:1
#SBATCH --output=logs/slurm/%x-%j.out

module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.8.0

source .venv/bin/activate

# Evaluate best SFT adapters from each hyperparam sweep on all test datasets.
# GENERATION_ADAPTER and CLASSIFICATION_ADAPTER are set by submit_sft_eval.py
# based on highest validation harm F1 across sweep runs.
#
# Optional overrides (set by submit_sft_eval.py when --base-model is given,
# e.g. for the Gemma 3 12B sweep):
#   BASE_MODEL              — HF model name (e.g. google/gemma-3-12b-it).
#                             Required when adapters were trained on a non-default base
#                             (the eval config defaults to Llama 3.1 8B).
#   EXPERIMENT_CONDITIONS   — comma-separated condition list, e.g.
#                             "finetuned_generation,finetuned_classification".
#                             When the project has only generation adapters
#                             (Gemma sweep, generation-only), submit_sft_eval.py
#                             sets this to "finetuned_generation" so the eval
#                             doesn't try to load a missing classification adapter.

MODEL_OVERRIDE=()
if [ -n "${BASE_MODEL:-}" ]; then
    MODEL_OVERRIDE=("model.name=${BASE_MODEL}")
fi

CONDITIONS_OVERRIDE=()
if [ -n "${EXPERIMENT_CONDITIONS:-}" ]; then
    CONDITIONS_OVERRIDE=("experiment.conditions=[${EXPERIMENT_CONDITIONS}]")
fi

python scripts/eval_safety_classifier.py \
    --config-name=${CONFIG_NAME:-eval_sft_baselines} \
    "${MODEL_OVERRIDE[@]}" \
    "${CONDITIONS_OVERRIDE[@]}" \
    "finetuned.generation_adapter=${GENERATION_ADAPTER}" \
    "finetuned.classification_adapter=${CLASSIFICATION_ADAPTER}"
