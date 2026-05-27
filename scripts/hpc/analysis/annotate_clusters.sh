#!/bin/bash
#SBATCH --job-name=annotate-clusters
#SBATCH --time=01:30:00
#SBATCH --mem=32GB
#SBATCH --partition=gpumedium
#SBATCH --gpus-per-node=rtx_pro_6000:1
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/slurm/%x-%j.out

module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.8.0

source .venv/bin/activate

mkdir -p "logs/slurm"

# Defaults match the LE-DPO cluster export from
# notebooks/analysis/intent_embedding_pca.ipynb. Override per-run with:
#   sbatch --export=ALL,INPUT=...,OUTPUT=...,MODEL=...,N_EXAMPLES=... annotate_clusters.sh
INPUT="${INPUT:-data/cluster_annotations/le_dpo_intent_clusters.jsonl}"
OUTPUT="${OUTPUT:-data/cluster_annotations/le_dpo_cluster_labels.jsonl}"
MODEL="${MODEL:-openai/gpt-oss-120b}"
N_EXAMPLES="${N_EXAMPLES:-10}"
SAMPLING_STRATEGY="${SAMPLING_STRATEGY:-random}"
SEED="${SEED:-42}"

python scripts/analysis/annotate_clusters.py \
    --input "${INPUT}" \
    --output "${OUTPUT}" \
    --model "${MODEL}" \
    --n-examples "${N_EXAMPLES}" \
    --sampling-strategy "${SAMPLING_STRATEGY}" \
    --seed "${SEED}" \
    --tensor-parallel-size 1
