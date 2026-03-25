#!/bin/bash
#SBATCH --job-name=distill-pipeline
#SBATCH --time=04:00:00
#SBATCH --mem=32GB
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=rtx_pro_6000:1
#SBATCH --output=logs/slurm/%x-%j.out

module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.8.0

source .venv/bin/activate

# Full distillation pipeline: teacher trace generation → student fine-tuning → safety evaluation.
# Configuration is driven by configs/experiments/distillation_pipeline.yaml.
# Stages can be individually disabled in the config (stages.step1_teacher, step2_distill, step3_safety).
python scripts/distillation/run_distillation_pipeline.py
