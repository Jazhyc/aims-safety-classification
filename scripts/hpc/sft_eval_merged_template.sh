#!/bin/bash
#SBATCH --job-name=sft_eval_merged
#SBATCH --time=02:00:00
#SBATCH --mem=16GB
#SBATCH --partition=gpumedium
#SBATCH --gpus-per-node=rtx_pro_6000:1
#SBATCH --output=logs/slurm/%x-%j.out

module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.8.0

source .venv/bin/activate

# Evaluate a single pre-merged SFT model on all test datasets.
# MERGED_MODEL_PATH and CONDITION are set by submit_sft_eval.py --use-merged.
# Merge adapters first with: python scripts/merge_adapter.py ...
python scripts/eval_safety_classifier.py \
    --config-name=eval_sft_baselines \
    "model.name=${MERGED_MODEL_PATH}" \
    "finetuned.use_merged=true" \
    "experiment.conditions=[${CONDITION}]"
