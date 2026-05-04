#!/bin/bash
#SBATCH --job-name=ablation-trace-gen
#SBATCH --time=04:00:00
#SBATCH --mem=32GB
#SBATCH --partition=gpumedium
#SBATCH --gpus-per-node=rtx_pro_6000:1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/slurm/%x-%j.out

module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.8.0

source .venv/bin/activate

# Variables (set by submit_label_source_ablation.py):
#   MODEL_NAME    — HuggingFace teacher ID
#   OUTPUT_DIR    — base traces dir; the script writes to <OUTPUT_DIR>/<model-slug>/<DATA_CONDITION_DIR>/train/
#   THINKING_MODE — "true"/"false"
#   SAMPLES_JSON  — path to the prepared samples JSON for this ablation condition
#   CONDITION     — distillation condition (only "synthetic_intent" for the ablation)
#   DATA_CONDITION_DIR — subdirectory name inserted between teacher slug and split,
#                        e.g. "hard_original" / "random_original"

mkdir -p "logs/slurm"

# Clean teacher slug for the path; keep in sync with _TEACHER_SLUG_OVERRIDES
# in run_distillation_pipeline.py / generate_reasoning_traces.py.
TEACHER_SLUG="${MODEL_NAME//\//-}"

# Generate traces, then move the resulting "<teacher_slug>/train" output
# into "<teacher_slug>/<DATA_CONDITION_DIR>/train" so multiple data conditions
# can coexist under the same teacher slug without overwriting one another.
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

# generate_reasoning_traces.py writes to <output_dir>/<teacher_slug>/{train,validation,test}.
# We expect only `train/` to be populated (samples_json sets split="train").
if [ -d "${INTERIM_OUTPUT_DIR}/${TEACHER_SLUG}/train" ]; then
    rm -rf "${FINAL_DIR}/train"
    mv "${INTERIM_OUTPUT_DIR}/${TEACHER_SLUG}/train" "${FINAL_DIR}/train"
    echo "Moved trace output → ${FINAL_DIR}/train"
else
    echo "ERROR: expected ${INTERIM_OUTPUT_DIR}/${TEACHER_SLUG}/train to exist" >&2
    exit 1
fi

rm -rf "${INTERIM_OUTPUT_DIR}"
