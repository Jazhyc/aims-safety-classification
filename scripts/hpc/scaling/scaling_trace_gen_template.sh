#!/bin/bash
#SBATCH --job-name=scaling-trace-gen
#SBATCH --time=24:00:00
#SBATCH --mem=32GB
#SBATCH --partition=gpumedium
#SBATCH --gpus-per-node=rtx_pro_6000:1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/slurm/%x-%j.out

module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.8.0

source .venv/bin/activate

# Variables (set by submit_scaling.py):
#   MODEL_NAME           — HuggingFace teacher ID
#   OUTPUT_DIR           — base traces dir (writes to <OUTPUT_DIR>/<teacher_slug>/<DATA_CONDITION_DIR>/train/)
#   THINKING_MODE        — "true"/"false"
#   SAMPLES_JSON         — pre-built samples JSON path
#   CONDITION            — distillation condition (only "synthetic_intent" for scaling)
#   DATA_CONDITION_DIR   — subdirectory name under the teacher slug (e.g. "full_wg")

mkdir -p "logs/slurm"

# Clean teacher slug for the path; keep in sync with _TEACHER_SLUG_OVERRIDES
# in run_distillation_pipeline.py / generate_reasoning_traces.py.
TEACHER_SLUG="${MODEL_NAME//\//-}"

# generate_reasoning_traces.py writes to <output_dir>/<teacher_slug>/{train,validation,test}.
# We funnel into an interim dir and move only `train/` into place so a future
# data condition under the same teacher slug doesn't overwrite this one.
INTERIM_OUTPUT_DIR="${OUTPUT_DIR}/_interim_${SLURM_JOB_ID:-$$}_${DATA_CONDITION_DIR}"
mkdir -p "${INTERIM_OUTPUT_DIR}"

python scripts/distillation/generate_reasoning_traces.py \
    --config-name=reasoning_traces \
    model.name="${MODEL_NAME}" \
    model.backend=vllm \
    thinking_mode="${THINKING_MODE}" \
    "conditions=[${CONDITION}]" \
    "dataset.samples_json=${SAMPLES_JSON}" \
    paths.output_dir="${INTERIM_OUTPUT_DIR}"

FINAL_DIR="${OUTPUT_DIR}/${TEACHER_SLUG}/${DATA_CONDITION_DIR}"
mkdir -p "${FINAL_DIR}"

if [ -d "${INTERIM_OUTPUT_DIR}/${TEACHER_SLUG}/train" ]; then
    rm -rf "${FINAL_DIR}/train"
    mv "${INTERIM_OUTPUT_DIR}/${TEACHER_SLUG}/train" "${FINAL_DIR}/train"
    echo "Moved trace output → ${FINAL_DIR}/train"
else
    echo "ERROR: expected ${INTERIM_OUTPUT_DIR}/${TEACHER_SLUG}/train to exist" >&2
    exit 1
fi

rm -rf "${INTERIM_OUTPUT_DIR}"
