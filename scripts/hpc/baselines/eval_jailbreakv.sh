#!/bin/bash
#SBATCH --job-name=jailbreakv_eval
#SBATCH --time=04:00:00
#SBATCH --mem=32GB
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=rtx_pro_6000:1
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/slurm/%x-%j.out

# Evaluate the four prior-work guardrails (WildGuard, Nemotron Safety Reasoner 4B,
# GuardReasoner 8B, GPT-OSS-Safeguard 120B) and our released distilled Gemma-3-12B
# classifier on JailbreakV-28K (all three splits). Outputs land under
# data/safety_experiment/jailbreakv/<dataset>/.
#
# All prompts are treated as harmful (the source has no benign labels), so F1
# reduces to 2*recall / (1 + recall). compute_metrics() handles this automatically.

set -euo pipefail

module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.8.0

source .venv/bin/activate

mkdir -p "logs/slurm"

python scripts/eval_safety_classifier.py --config-name=eval_jailbreakv
