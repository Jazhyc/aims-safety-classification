#!/bin/bash
#SBATCH --job-name=grpo_baseline_eval
#SBATCH --time=01:00:00
#SBATCH --mem=32GB
#SBATCH --partition=gpumedium
#SBATCH --gpus-per-node=rtx_pro_6000:1
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/slurm/%x-%j.out

# One-off eval of the GRPO intent-safety classifier
# (iustinsirbu/llama-3.1-8b-grpo-intent-safety) against the standard
# annotated-intents + 5-OOD-set evaluation suite. The condition is wired into
# eval_safety_classifier.py as a prior-work-style classifier with its own
# system prompt (see configs/prompt_templates.yaml :: grpo_system_prompt).
#
# Override at submit time via --export=ALL,GRPO_MODEL=<hf-repo>:
#   GRPO_MODEL — HuggingFace repo or local path to a GRPO-style classifier.

set -euo pipefail

module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.8.0

source .venv/bin/activate

GRPO_MODEL="${GRPO_MODEL:-iustinsirbu/llama-3.1-8b-grpo-intent-safety}"

# Use the same dataset layout as eval_safety_classifier.yaml (annotated-intents
# test split + 5 OOD sets) but restrict to the new GRPO condition. Outputs land
# under data/safety_experiment/<dataset>/ per the standard layout.
python scripts/eval_safety_classifier.py \
    --config-name=eval_safety_classifier \
    "experiment.conditions=[grpo_classification]" \
    "grpo.name=${GRPO_MODEL}" \
    "wandb.enabled=true" \
    "wandb.run_name=grpo-baseline-$(basename ${GRPO_MODEL})"
