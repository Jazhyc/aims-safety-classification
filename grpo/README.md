

The GRPO models were trained independently from the rest of the models, using a slurm node with 8 GPUs. The dependencis are solved via a sif container that is provided for starting sbatch jobs on the node. Below are the instructions for replicating GRPO training.


## Step 1

Copy the example config and fill in the values for your cluster:

```bash
cp grpo/config.sh.example grpo/config.sh
```

Edit `grpo/config.sh` and set:

- `SIF_PATH` — absolute path where the `.sif` container will be built and stored
- `SLURM_PARTITION_DATA` / `SLURM_PARTITION_JUDGE` / `SLURM_PARTITION_TRAIN` — SLURM partitions for each job type; set them all to the same value if your cluster uses a single partition
- `SLURM_ACCOUNT` — your SLURM account (remove the `--account` flag from the sbatch calls if your cluster does not require one)
- `NLP_DIR` — directory where large files (models, cache) are stored; this gets bind-mounted into the container
- `HF_HOME` / `TORCH_COMPILE_CACHE_DIR` — cache directories, defaulting to `$NLP_DIR/.cache`; override if your cache lives elsewhere

`config.sh` is gitignored and will never be committed.


## Step 2

Build the container:

```bash
source grpo/config.sh
apptainer build --fakeroot $SIF_PATH grpo/verl_vllm017.def
```

This installs verl and all dependencies needed for GRPO training.


## Step 3

Prepare the dataset:

```bash
bash grpo/prepare_parquet_dataset.sh
```

This downloads `Jazhyc/aims-safety-intents` from HuggingFace, formats it for veRL, and writes `train.parquet` and `validation.parquet` to `grpo/data/aims_safety_intents/`. The validation file combines the original validation and test splits.

The system prompt used is `grpo/system_prompt_reasoning_long.txt`.


## Step 4

Start the judge service:

```bash
bash grpo/reward_fn_judge_api.sh
```

This submits a SLURM job that serves a FastAPI reward model on port 8000. The judge model (`JUDGE_MODEL` in `reward_fn_judge_api.sh`, default `google/gemma-3-27b-it`) is downloaded automatically to `HF_HOME` on first run. The service must be running before launching GRPO training, as the trainer calls it to score each rollout.

Logs are written to `grpo/logs_judge/`.

Once the job is running, find the hostname of the node it was assigned to (e.g. via `squeue`) and set `JUDGE_ADDRESS` in `config.sh` to that value before launching training.


## Step 5

Launch GRPO training:

```bash
bash grpo/train.sh
```

This submits a SLURM job that runs GRPO training via veRL on 8 GPUs. Checkpoints are saved to `CKPT_ROOT_PATH/<experiment_name>/` at the frequency set by `trainer.save_freq`. Training progress is logged to WandB under `WANDB_PROJECT`.

To change the model being trained, the experiment name, or any veRL hyperparameters, edit the variables at the top of `train.sh`.

Logs are written to `grpo/logs_train/`.


## Step 6

Merge FSDP checkpoint shards into HuggingFace model format:

```bash
bash grpo/merge_model_states.sh
```

This submits one SLURM job per checkpoint step, converting the FSDP actor shards saved during training into a standard HuggingFace model directory at `CKPT_ROOT_PATH/<experiment_name>/huggingface_merged_step<N>/`.

Before running, set `EXPERIMENT_NAME` and `STEPS` at the top of `merge_model_states.sh` to match your training run. `STEPS` should list the `global_step_*` directories you want to merge (determined by `trainer.save_freq` in `train.sh`).

Logs are written to `grpo/logs_merge/`.


## Step 7

Evaluate each merged checkpoint across all safety benchmarks:

```bash
bash grpo/evaluate.sh
```

This submits one SLURM job per checkpoint step, running inference with vLLM and computing F1 scores on the following datasets: `aims-safety-intents` (internal), `wildguardmix`, `xstest`, `aegis`, `toxic-chat`, `openai-moderation`, and held-out validation splits. Results are saved to `CKPT_ROOT_PATH/<experiment_name>/results_step<N>/`.

Set `EXPERIMENT_NAME` and `STEPS` at the top of `evaluate.sh` to match your training run.

Logs are written to `grpo/logs_evaluate/`.


## Step 8

Select the best checkpoint:

```bash
bash grpo/find_best_checkpoints.sh
```

This reads the `combined_metrics.json` files produced in Step 7 and writes `best_checkpoints_report.txt` to `CKPT_ROOT_PATH/<experiment_name>/`, identifying the best checkpoint by internal dataset F1 and by validation dataset F1.

Set `EXPERIMENT_NAME` at the top of `find_best_checkpoints.sh` to match your experiment.

Logs are written to `grpo/logs_find_best/`.


