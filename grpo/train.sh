#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

# ── config ────────────────────────────────────────────────────────────────────

MODEL_PATH="meta-llama/Llama-3.1-8B-Instruct"
DATA_DIR="$SCRIPT_DIR/data/aims_safety_intents"
REWARD_FN="$SCRIPT_DIR/reward_fn_judge_async.py"
REWARD_FN_NAME="fn__with_reasoning__format_label_judge"
WANDB_PROJECT="intent_for_safety"
EXPERIMENT_NAME="grpo_n16_b16_temp12_ep10"
CKPT_DIR="${CKPT_ROOT_PATH}/${EXPERIMENT_NAME}"

mkdir -p "$SCRIPT_DIR/logs_train"
mkdir -p "$CKPT_DIR"

# ── ray / vllm env ────────────────────────────────────────────────────────────

unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES

export RAY_TMPDIR="/tmp/ray_tmp"
mkdir -p $RAY_TMPDIR
export RAY_ADDRESS_TEMP_DIR=$RAY_TMPDIR
export XDG_RUNTIME_DIR=$RAY_TMPDIR
export TMPDIR=$RAY_TMPDIR

export VLLM_RPC_TIMEOUT=600000
export VLLM_ENGINE_ITERATION_TIMEOUT_S=120
export RAY_rpc_server_listen_port=6379
export RAY_num_server_call_threads=128

export MY_EXPERIMENT_NAME=$EXPERIMENT_NAME

# ── launch ────────────────────────────────────────────────────────────────────

sbatch -p $SLURM_PARTITION_TRAIN \
    --account $SLURM_ACCOUNT \
    --mem=1100G \
    --cpus-per-task 64 \
    --gres gpu:8 \
    --nodes=1 \
    --ntasks-per-node=1 \
    --time=24:00:00 \
    --job-name=Safety \
    --output="$SCRIPT_DIR/logs_train/train_%j_%N.log" \
    --wrap="apptainer exec --nv \
        -B ${NLP_DIR}:${NLP_DIR} \
        --env HF_HOME=${HF_HOME} \
        --env TORCH_COMPILE_CACHE_DIR=${TORCH_COMPILE_CACHE_DIR} \
        --env WANDB_API_KEY=${WANDB_API_KEY} \
        --env JUDGE_ADDRESS=${JUDGE_ADDRESS} \
        --env VLLM_RPC_TIMEOUT=${VLLM_RPC_TIMEOUT} \
        --env VLLM_ENGINE_ITERATION_TIMEOUT_S=${VLLM_ENGINE_ITERATION_TIMEOUT_S} \
        --env RAY_rpc_server_listen_port=${RAY_rpc_server_listen_port} \
        --env RAY_num_server_call_threads=${RAY_num_server_call_threads} \
        --env MY_EXPERIMENT_NAME=${MY_EXPERIMENT_NAME} \
        --env RAY_TMPDIR=${RAY_TMPDIR} \
        --env TMPDIR=${RAY_TMPDIR} \
        ${SIF_PATH} python3 $SCRIPT_DIR/launch_verl.py \
        algorithm.adv_estimator=grpo \
        data.train_files=${DATA_DIR}/train.parquet \
        data.val_files=${DATA_DIR}/validation.parquet \
        data.train_batch_size=128 \
        data.max_prompt_length=1024 \
        data.max_response_length=512 \
        data.truncation='right' \
        actor_rollout_ref.model.path=${MODEL_PATH} \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.actor.ppo_mini_batch_size=128 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.enforce_eager=True \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
        actor_rollout_ref.rollout.max_model_len=2048 \
        actor_rollout_ref.rollout.prompt_length=1280 \
        actor_rollout_ref.rollout.response_length=512 \
        actor_rollout_ref.rollout.max_num_seqs=64 \
        actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
        actor_rollout_ref.rollout.n=16 \
        actor_rollout_ref.rollout.free_cache_engine=True \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
        actor_rollout_ref.actor.use_kl_loss=True \
        actor_rollout_ref.actor.kl_loss_coef=0.001 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.actor.fsdp_config.use_torch_compile=False \
        custom_reward_function.path=${REWARD_FN} \
        custom_reward_function.name=${REWARD_FN_NAME} \
        trainer.logger=['wandb'] \
        trainer.project_name=${WANDB_PROJECT} \
        trainer.experiment_name=${EXPERIMENT_NAME} \
        trainer.n_gpus_per_node=8 \
        trainer.nnodes=1 \
        trainer.save_freq=5 \
        trainer.test_freq=5 \
        trainer.total_epochs=10 \
        trainer.default_local_dir=${CKPT_DIR} \
        actor_rollout_ref.rollout.enable_chunked_prefill=False \
        actor_rollout_ref.rollout.enable_prefix_caching=False \
        actor_rollout_ref.rollout.temperature=1.2"
