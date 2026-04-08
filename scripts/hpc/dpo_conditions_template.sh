#!/bin/bash
#SBATCH --job-name=dpo-cond
#SBATCH --time=10:00:00
#SBATCH --mem=64GB
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=a100:1
#SBATCH --output=logs/slurm/%x-%j.out

set -euo pipefail

module purge
module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.8.0

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${PROJECT_ROOT}"

source .venv/bin/activate
if [ -f "${HOME}/.wandb_secrets" ]; then
  # shellcheck disable=SC1090
  source "${HOME}/.wandb_secrets"
fi

export PYTHONUNBUFFERED=1

STAGE="${STAGE:-}"
if [ -z "${STAGE}" ]; then
  echo "ERROR: STAGE is required."
  echo "Valid STAGE values: hard_dpo | judge_dpo | augment_prep | sft_aug | dpo_aug"
  exit 1
fi

# ---------------------------------------------------------------------------
# Shared defaults (override via sbatch --export)
# ---------------------------------------------------------------------------
BASE_MODEL="${BASE_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
BASE_SFT_ADAPTER="${BASE_SFT_ADAPTER:-trained_models/causal/hyperparam_sweep/lr_5e-05_e_5_adapter}"
JUDGE_MODEL="${JUDGE_MODEL:-google/gemma-3-27b-it}"
# Gemma 27B needs 2× A100-40GB in bfloat16; set to 1 only for small judge models
JUDGE_TENSOR_PARALLEL="${JUDGE_TENSOR_PARALLEL:-2}"

SEED="${SEED:-22}"
TEMPERATURE="${TEMPERATURE:-0.8}"
K_SAMPLES="${K_SAMPLES:-5}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
EPOCHS="${EPOCHS:-3}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
DPO_BETA="${DPO_BETA:-0.1}"
HARMFUL_WEIGHT="${HARMFUL_WEIGHT:-1.0}"
UNBALANCED="${UNBALANCED:-0}"
AUG_EPOCHS="${AUG_EPOCHS:-1}"
FORCE="${FORCE:-0}"
FORCE_FROM="${FORCE_FROM:-}"
FORCE_AUGMENT="${FORCE_AUGMENT:-0}"
ATN_IMPL="${ATN_IMPL:-sdpa}"

WANDB_PROJECT="${WANDB_PROJECT:-intention-jailbreak}"
WANDB_PROJECT_SFT="${WANDB_PROJECT_SFT:-sft-hyperparam-sweep}"

# Condition paths
HARD_PAIRS_DIR="${HARD_PAIRS_DIR:-data/dpo_pairs/hard}"
HARD_BALANCED_DIR="${HARD_BALANCED_DIR:-data/dpo_pairs/hard_balanced}"
HARD_DPO_OUTPUT="${HARD_DPO_OUTPUT:-trained_models/causal/dpo-hard}"
SFT_PRED_DIR_HARD="${SFT_PRED_DIR_HARD:-data/predictions/sft_hard}"

JUDGE_PAIRS_DIR="${JUDGE_PAIRS_DIR:-data/dpo_pairs/judge}"
JUDGE_BALANCED_DIR="${JUDGE_BALANCED_DIR:-data/dpo_pairs/judge_balanced}"
JUDGE_DPO_OUTPUT="${JUDGE_DPO_OUTPUT:-trained_models/causal/dpo-judge}"
SFT_PRED_DIR_JUDGE="${SFT_PRED_DIR_JUDGE:-data/predictions/sft_judge}"
# Override epochs/beta for judge independently of the global EPOCHS/DPO_BETA.
# Falls back to global values when not set or empty.
JUDGE_EPOCHS="${JUDGE_EPOCHS:=${EPOCHS}}"
JUDGE_BETA="${JUDGE_BETA:=${DPO_BETA}}"
# Set to non-empty to skip sample generation and reuse cached parsed_samples.jsonl
JUDGE_FROM_SAMPLES="${JUDGE_FROM_SAMPLES:-}"

ANNOTATED_PAIRS_DIR="${ANNOTATED_PAIRS_DIR:-${HARD_PAIRS_DIR}}"
WILDGUARD_CANDIDATES="${WILDGUARD_CANDIDATES:-data/wildguard_easy/candidates.jsonl}"
WILDGUARD_MODEL_PATH="${WILDGUARD_MODEL_PATH:-models/modernbert-ensemble/final_model}"
WILDGUARD_PAIRS_DIR="${WILDGUARD_PAIRS_DIR:-data/wildguard_easy/dpo_pairs}"
SFT_EXAMPLES_OUTPUT="${SFT_EXAMPLES_OUTPUT:-data/wildguard_easy/sft_examples.jsonl}"
AUGMENTED_MERGED_PAIRS_DIR="${AUGMENTED_MERGED_PAIRS_DIR:-data/dpo_pairs/augmented}"

AUG_SFT_MODEL_SAVE_DIR="${AUG_SFT_MODEL_SAVE_DIR:-models/sft/llm_sweep/augmented_sft}"
AUG_SFT_ADAPTER_PATH="${AUG_SFT_ADAPTER_PATH:-${AUG_SFT_MODEL_SAVE_DIR}_adapter}"
# When NO_SFT_AUG=1, dpo_aug starts from the original SFT adapter instead of
# the augmented one (avoids the imbalanced sft_examples contamination).
NO_SFT_AUG="${NO_SFT_AUG:-0}"
DPO_AUG_INIT_ADAPTER="$( [ "${NO_SFT_AUG}" = "1" ] && echo "${BASE_SFT_ADAPTER}" || echo "${AUG_SFT_ADAPTER_PATH}" )"
AUG_SFT_OUTPUT_DIR="${AUG_SFT_OUTPUT_DIR:-data/train_results/sft_augmented}"
AUG_SFT_LOGS_DIR="${AUG_SFT_LOGS_DIR:-logs/sft_augmented}"
AUG_SFT_PRED_DIR="${AUG_SFT_PRED_DIR:-data/predictions/sft_augmented}"
AUG_SFT_WANDB_NAME="${AUG_SFT_WANDB_NAME:-sft-augmented}"

AUG_BALANCED_DIR="${AUG_BALANCED_DIR:-data/dpo_pairs/augmented_balanced}"
AUG_DPO_OUTPUT="${AUG_DPO_OUTPUT:-trained_models/causal/dpo-augmented}"

echo "======================================================================"
echo " Preference Condition Stage"
echo "  Stage       : ${STAGE}"
echo "  Project root: ${PROJECT_ROOT}"
echo "  Base model  : ${BASE_MODEL}"
echo "======================================================================"

PIPELINE_RESTART_ARGS=()
if [ "${FORCE}" = "1" ]; then
  PIPELINE_RESTART_ARGS+=(--force)
fi
if [ -n "${FORCE_FROM}" ]; then
  PIPELINE_RESTART_ARGS+=(--force-from "${FORCE_FROM}")
fi
if [ "${FORCE_AUGMENT}" = "1" ]; then
  PIPELINE_RESTART_ARGS+=(--force-augment)
fi

case "${STAGE}" in
  hard_dpo)
    HARD_SKIP_STEPS="6"
    HARD_EXTRA_ARGS=()
    if [ "${UNBALANCED}" = "1" ]; then
      # Skip balancing step; use harmful_weight in loss instead
      HARD_SKIP_STEPS="2,6"
      HARD_EXTRA_ARGS+=(--unbalanced --harmful-weight "${HARMFUL_WEIGHT}")
    fi
    python scripts/run_preference_pipeline.py \
      --adapter-path "${BASE_SFT_ADAPTER}" \
      --base-model "${BASE_MODEL}" \
      --pairs-dir "${HARD_PAIRS_DIR}" \
      --balanced-pairs-dir "${HARD_BALANCED_DIR}" \
      --dpo-output-dir "${HARD_DPO_OUTPUT}" \
      --sft-pred-dir "${SFT_PRED_DIR_HARD}" \
      --temperature "${TEMPERATURE}" \
      --k-samples "${K_SAMPLES}" \
      --max-model-len "${MAX_MODEL_LEN}" \
      --epochs "${EPOCHS}" \
      --learning-rate "${LEARNING_RATE}" \
      --batch-size "${BATCH_SIZE}" \
      --grad-accum "${GRAD_ACCUM}" \
      --dpo-beta "${DPO_BETA}" \
      --seed "${SEED}" \
      --wandb-project "${WANDB_PROJECT}" \
      "${PIPELINE_RESTART_ARGS[@]}" \
      "${HARD_EXTRA_ARGS[@]}" \
      --skip-steps "${HARD_SKIP_STEPS}"
    ;;

  judge_dpo)
    # Two-pass judge flow. Pass 1 (sample generation) is skipped when
    # JUDGE_FROM_SAMPLES is set — reuses cached parsed_samples.jsonl.
    if [ -z "${JUDGE_FROM_SAMPLES}" ]; then
      python scripts/generate_dpo_pairs.py \
        --adapter-path "${BASE_SFT_ADAPTER}" \
        --base-model "${BASE_MODEL}" \
        --output-dir "${JUDGE_PAIRS_DIR}" \
        --temperature "${TEMPERATURE}" \
        --num-samples "${K_SAMPLES}" \
        --max-model-len "${MAX_MODEL_LEN}"
    else
      echo "[judge_dpo] Skipping sample generation — reusing ${JUDGE_PAIRS_DIR}/parsed_samples.jsonl"
    fi

    python scripts/generate_dpo_pairs.py \
      --from-samples "${JUDGE_PAIRS_DIR}/parsed_samples.jsonl" \
      --adapter-path "${BASE_SFT_ADAPTER}" \
      --base-model "${BASE_MODEL}" \
      --output-dir "${JUDGE_PAIRS_DIR}" \
      --intent-filter \
      --judge-model "${JUDGE_MODEL}" \
      --judge-tensor-parallel "${JUDGE_TENSOR_PARALLEL}" \
      --max-model-len "${MAX_MODEL_LEN}"

    python scripts/run_preference_pipeline.py \
      --adapter-path "${BASE_SFT_ADAPTER}" \
      --base-model "${BASE_MODEL}" \
      --pairs-dir "${JUDGE_PAIRS_DIR}" \
      --balanced-pairs-dir "${JUDGE_BALANCED_DIR}" \
      --dpo-output-dir "${JUDGE_DPO_OUTPUT}" \
      --sft-pred-dir "${SFT_PRED_DIR_JUDGE}" \
      --epochs "${JUDGE_EPOCHS}" \
      --learning-rate "${LEARNING_RATE}" \
      --batch-size "${BATCH_SIZE}" \
      --grad-accum "${GRAD_ACCUM}" \
      --dpo-beta "${JUDGE_BETA}" \
      --seed "${SEED}" \
      --wandb-project "${WANDB_PROJECT}" \
      "${PIPELINE_RESTART_ARGS[@]}" \
      --skip-steps 1,6
    ;;

  augment_prep)
    # Pre-steps A/B/C only: filter WildGuard, generate easy DPO pairs, merge pairs.
    python scripts/run_preference_pipeline.py \
      --adapter-path "${BASE_SFT_ADAPTER}" \
      --base-model "${BASE_MODEL}" \
      --augment \
      --pairs-dir "${ANNOTATED_PAIRS_DIR}" \
      --wildguard-candidates "${WILDGUARD_CANDIDATES}" \
      --wildguard-model-path "${WILDGUARD_MODEL_PATH}" \
      --wildguard-pairs-dir "${WILDGUARD_PAIRS_DIR}" \
      --augmented-pairs-dir "${AUGMENTED_MERGED_PAIRS_DIR}" \
      --sft-examples-output "${SFT_EXAMPLES_OUTPUT}" \
      --temperature "${TEMPERATURE}" \
      --k-samples "${K_SAMPLES}" \
      --max-model-len "${MAX_MODEL_LEN}" \
      --seed "${SEED}" \
      "${PIPELINE_RESTART_ARGS[@]}" \
      --skip-steps 1,2,3,4,5,6
    ;;

  sft_aug)
    python scripts/baselines/train_generator.py \
      --config-name=llm_sweep \
      model.name="${BASE_MODEL}" \
      model.attn_implementation="${ATN_IMPL}" \
      +data.extra_train_data="${SFT_EXAMPLES_OUTPUT}" \
      paths.model_save_dir="${AUG_SFT_MODEL_SAVE_DIR}" \
      paths.output_dir="${AUG_SFT_OUTPUT_DIR}" \
      paths.logs_dir="${AUG_SFT_LOGS_DIR}" \
      paths.predictions_dir="${AUG_SFT_PRED_DIR}" \
      wandb.project="${WANDB_PROJECT_SFT}" \
      wandb.name="${AUG_SFT_WANDB_NAME}"
    ;;

  dpo_aug)
    # Uses merged pairs from augment_prep.
    # Init adapter: augmented SFT (default) or original SFT (when NO_SFT_AUG=1).
    echo "  DPO-aug init adapter: ${DPO_AUG_INIT_ADAPTER}"
    python scripts/run_preference_pipeline.py \
      --adapter-path "${DPO_AUG_INIT_ADAPTER}" \
      --base-model "${BASE_MODEL}" \
      --augment \
      --pairs-dir "${ANNOTATED_PAIRS_DIR}" \
      --wildguard-candidates "${WILDGUARD_CANDIDATES}" \
      --wildguard-model-path "${WILDGUARD_MODEL_PATH}" \
      --wildguard-pairs-dir "${WILDGUARD_PAIRS_DIR}" \
      --augmented-pairs-dir "${AUGMENTED_MERGED_PAIRS_DIR}" \
      --sft-examples-output "${SFT_EXAMPLES_OUTPUT}" \
      --balanced-pairs-dir "${AUG_BALANCED_DIR}" \
      --dpo-output-dir "${AUG_DPO_OUTPUT}" \
      --sft-pred-dir "${AUG_SFT_PRED_DIR}" \
      --epochs "${AUG_EPOCHS}" \
      --learning-rate "${LEARNING_RATE}" \
      --batch-size "${BATCH_SIZE}" \
      --grad-accum "${GRAD_ACCUM}" \
      --dpo-beta "${DPO_BETA}" \
      --seed "${SEED}" \
      --wandb-project "${WANDB_PROJECT}" \
      "${PIPELINE_RESTART_ARGS[@]}" \
      --skip-steps 1,6
    ;;

  *)
    echo "ERROR: Unknown STAGE='${STAGE}'"
    exit 2
    ;;
esac

echo ""
echo "Stage ${STAGE} completed."
