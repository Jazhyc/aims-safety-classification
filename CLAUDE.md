# Intention Jailbreak — Project Guide

## Overview

Research project on jailbreak detection using verbalized intent analysis. The core idea is to train models to reason about *why* a user is asking something (intent) and use that to classify harmfulness, rather than pure text-surface signals.

## Repository Structure

```
intention-jailbreak/
├── configs/                    # Hydra YAML configs for training and generation
│   ├── train_config.yaml
│   ├── prompt_templates.yaml
│   └── experiments/            # Per-script experiment configs (one YAML per script)
├── data/                       # (gitignored) datasets, predictions, reasoning traces, etc.
│   └── train_results/          # (gitignored) HF Trainer checkpoints per run
│       └── distillation/       # Distillation training checkpoints
├── logs/                       # (gitignored) training logs
├── models/                     # (gitignored) trained LoRA adapters and model weights
│   ├── distillation/           # Reasoning-distillation adapters (set via distilled_models_dir)
│   ├── sft/                    # SFT/hyperparam-sweep adapters (set via sft_models_dir)
│   ├── bert_harm/              # BERT harm classifier checkpoints
│   └── bert_intent/            # BERT intent classifier checkpoints
├── notebooks/                  # Jupyter analysis notebooks — see categories below
│   ├── baselines/              # Baseline intent generation/classification analysis
│   ├── distillation/           # Teacher distillation sweep + reasoning trace analysis
│   ├── modernbert/             # ModernBERT classifier analysis
│   └── dataset_analysis/       # One-off dataset distribution notebooks
├── notes/                      # Markdown project notes and experiment logs
├── scripts/                    # Training, evaluation, generation scripts — see categories below
│   ├── baselines/              # Intent classification & generation baselines
│   ├── distillation/           # Teacher trace generation & student distillation
│   ├── dataset_analysis/       # One-off dataset analysis scripts
│   ├── dataset_generation/     # ModernBERT classifier training
│   ├── hpc/                    # SLURM job submission scripts + shell wrappers
│   └── report/                 # One-off scripts for report figures
├── src/intention_jailbreak/    # Library source code
│   ├── comparison/             # Synthetic intent generation + clustering comparison
│   ├── dataset/                # WildGuardMix loading + stratified splits
│   ├── embeddings/             # BERTopic wrapper
│   ├── ensemble/               # Deep ensemble classifier
│   ├── model_generation/       # LLM generation utils, prompt templates, preprocessing
│   └── training/               # Training utils, metrics, data prep
└── tests/
```

## Script Categories

| Subdirectory | Purpose | Scripts |
|---|---|---|
| `scripts/baselines/` | Intent classification & generation baselines | `eval_sft_baseline.py`, `eval_intent_generation.py`, `compare_models.py`, `train_generator.py` |
| `scripts/distillation/` | Teacher reasoning trace generation & student distillation | `generate_reasoning_traces.py`, `run_distillation_pipeline.py`, `check_trace_leakage.py` |
| `scripts/` (root) | Multi-condition safety/harm classification + misc | `eval_safety_classifier.py`, `intent_diversity_analysis.py` |
| `scripts/` (root, **do not modify**) | DPO + contrastive preference learning | `train_dpo.py`, `train_contrastive.py`, `generate_dpo_pairs.py`, `balance_dpo_pairs.py`, `compare_dpo_sweep.py`, `run_preference_pipeline.py` |
| `scripts/dataset_generation/` | ModernBERT classifier used to generate the annotation set via uncertainty filtering | `train.py`, `evaluate_test.py`, `plot_calibration.py` |
| `scripts/dataset_analysis/` | One-off scripts for analysing the annotated dataset | `generate_harm_labels.py`, `synthetic_comparison.py`, `evaluate_harm_predictions.py` |
| `scripts/hpc/` | SLURM submission + shell wrappers (keep as-is) | `submit_*.py`, `slurm_utils.py`, `*.sh` |
| `scripts/report/` | One-off figure generation for the paper | `plot_gemma_prompt_format.py`, `plot_qwen32b_reasoning_mode.py` |

## Notebook Categories

| Subdirectory | Purpose | Notebooks |
|---|---|---|
| `notebooks/baselines/` | Baseline intent generation/classification analysis | `eval_results_analysis.ipynb`, `wandb_llm_sweep_analysis.ipynb`, `safety_experiment_results.ipynb` |
| `notebooks/distillation/` | Teacher distillation sweep + reasoning trace analysis | `distillation_teacher_sweep.ipynb`, `reasoning_traces_analysis_llama.ipynb` |
| `notebooks/` (root, **do not modify**) | Preference learning results | `preference_learning_results.ipynb`, `diversity_analysis.ipynb` |
| `notebooks/modernbert/` | ModernBERT classifier analysis | `intent_classifier_analysis.ipynb`, `test_results_analysis.ipynb` |
| `notebooks/dataset_analysis/` | Old one-off notebooks for analysing the annotated dataset and its distribution | `annotation_analysis.ipynb`, `disagreement_analysis.ipynb`, `harm_labels_comparison.ipynb`, `harm_label_eval_4class_vs_binary.ipynb`, `wildguardmix_analysis.ipynb`, `clustering_analysis.ipynb`, `bert_harm_labels.ipynb` |

## Primary Dataset: `Jazhyc/wildguard-annotated-intents`

Human-annotated intents for ~1.7K WildGuardMix prompts. Loaded via HuggingFace `datasets`.

**Key properties:**
- Three splits on HuggingFace: `train`, `validation`, `test`
- Harm labels: 5-level (`Completely Harmful`, `Uncertain Harmful`, `Uncertain Safe`, `Completely Safe`, `Flag for Removal`)

**Split policy (8-1-1):**
- All entries sharing a prompt with any duplicate (detected by prompt text, not `Duplicate ID` which only tags the 2nd+ occurrence) go into **train only**
- Val and test are drawn only from entries with unique prompts, stratified on `Annotator Harm`
- Rationale: prevents cross-split leakage where the same prompt appears in both train and eval

**Dataset loading entry point:** `src/intention_jailbreak/model_generation/preprocessing.py` — `preprocess_data(split="train"|"validation"|"test")`. All consumers should call this with the appropriate split; do not call `train_val_test_split()` on this dataset anymore.

**Scripts that consume this dataset and the split they load:**
| Script | Split |
|---|---|
| `scripts/baselines/eval_sft_baseline.py` | `test` |
| `scripts/baselines/eval_intent_generation.py` (via config) | `test` |
| `scripts/distillation/generate_reasoning_traces.py` | all three (single vLLM pass) |
| `scripts/eval_safety_classifier.py` (via config) | `test` |
| `src/intention_jailbreak/model_generation/causal.py` | `train`/`validation`/`test` |
| `src/intention_jailbreak/model_generation/seq2seq.py` | `train`/`validation`/`test` |
| `src/intention_jailbreak/model_generation/bert_harm.py` | `train`/`validation`/`test` |
| `scripts/generate_dpo_pairs.py` *(do not touch)* | still uses manual split |

## Reasoning Traces

Generated by `scripts/distillation/generate_reasoning_traces.py` using a teacher LLM. All three Hub splits are processed in a single vLLM pass and saved to split-specific subdirectories:

```
data/reasoning_traces/<model-slug>/
  train/
    raw_outputs.json
    parsed_results.json      ← consumed by train_generator.py for SFT
  validation/
    raw_outputs.json
    parsed_results.json      ← consumed by train_generator.py as validation set
  test/
    raw_outputs.json
    parsed_results.json      ← consumed by eval_safety_classifier.py (future)
```

`scripts/distillation/run_distillation_pipeline.py` passes both `train/parsed_results.json` and `validation/parsed_results.json` to `scripts/baselines/train_generator.py` → `causal.py` for clean train/val separation.

### Reasoning Trace Conditions

Three conditions control how reasoning traces are generated and how students are trained on them:

| Condition | Preamble | Teacher Prompt | Student Output | Training Intent Source |
|---|---|---|---|---|
| `no_intent` | No mention of intent | Harm label only | Reasoning + Harm | N/A |
| `synthetic_intent` | Asks to produce intent | Harm label only | Reasoning + Intent + Harm | **Model-generated** intent from `predicted.prompt_intent` |
| `human_intent` | Asks to produce intent | Intent + harm label | Reasoning + Intent + Harm | **Human-annotated** intent from `ground_truth.intent` |

- **no_intent**: Baseline classification task. Preamble does not mention intent; model reasons about harm only. Teacher provides harm label as ground truth.
- **synthetic_intent**: Teacher generates intent itself (no ground truth intent provided). Student learns to mimic the teacher's reasoning process including the generated intent. Useful for studying model behavior on intent inference.
- **human_intent**: Teacher has access to human-annotated intent. Student learns on the ground truth intent. Strongest signal for distillation.

Trace files contain a `condition` field that filters which records are used during training (`causal.py:load_reasoning_traces_dataset`). Each trace record includes:
- `ground_truth.intent`: Human-annotated intent (used by `human_intent`, ignored by others)
- `predicted.prompt_intent`: Model-generated intent from teacher (used by `synthetic_intent`, ignored by others)
- `predicted.prompt_harm`: Teacher's predicted harm (used for optional `filter_teacher_disagreements`)

For eval, conditions map to:
- `no_intent` → `finetuned_reasoning_classification`
- `synthetic_intent` → `finetuned_reasoning_synthetic_intent`
- `human_intent` → `finetuned_reasoning_human_intent`

## Secondary Dataset: WildGuardMix

Used for ModernBERT classifier training. Loaded via `src/intention_jailbreak/dataset/wildguardmix.py`.
Stratified split on `prompt_harm_label`, `adversarial`, `subcategory`. English-only filtering via `language_filter.py`.

## Model Artifact Registry

LoRA adapters can be synced to a single W&B registry project (`intention-jailbreak-models`) via `src/intention_jailbreak/model_generation/artifacts.py`. This allows adapters trained on one device to be retrieved on another without needing to know which experiment project produced them.

Three functions:
- `artifact_exists(name, registry_project)` — call **before training starts**; raises if an adapter with the same name already exists, preventing accidental overwrites from parallel runs.
- `upload_adapter(adapter_path, registry_project)` — uploads root-level adapter files only (skips `checkpoint-N/` subdirs); called automatically after `trainer.save_model()` in `causal.py` when enabled.
- `download_adapter(name, registry_project, target_dir)` — downloads to `target_dir/name`; skips if directory already has files.

Artifact name = adapter directory name (e.g. `reasoning-distillation-gemma-3-27b-no-intent_adapter`), which is unique across pipelines. Enable per-config via `artifacts.enabled: true` in `llm_sweep.yaml` / `reasoning_distillation.yaml`.

## Experiment Status

| Stage | Status | Notes |
|---|---|---|
| Prompting baselines (`eval_safety_classifier.py`) | ✅ Complete | W&B project: `Baselines`; results artifact: `safety-experiment-baselines` |
| SFT hyperparam sweep (`submit_hyperparam_sweep.py`) | 🔄 In progress | Llama 3.1 8B; W&B project: `sft-hyperparam-sweep`; model selected by validation harm F1 (not cosine similarity) |
| Distillation teacher traces | ⬜ Pending | |
| Distillation student SFT | ⬜ Pending | |

**Sweep model selection policy:** rank sweep runs by **validation harm F1** (computed via `compare_models.py`), not cosine similarity. Cosine similarity measures annotation similarity, not task performance.

## Experimental Version: v7

Current distillation and trace generation experiments use version **v7**. The version tag is used throughout the codebase to avoid collision between experimental runs.

**Where v7 is referenced:**
- `scripts/hpc/submit_trace_generation.py`: `TRACES_VERSION = "v7"` → creates traces in `data/reasoning_traces_v7/`
- `scripts/hpc/submit_distillation_sweep.py`: `TRACES_VERSION = "v7"`
- `scripts/hpc/submit_distillation_eval.py`: `DEFAULT_TRACES_VERSION = "v7"`
- `scripts/distillation/score_tom_reasoning.py`: `DEFAULT_VERSION = "v7"`
- `scripts/distillation/reparse_traces.py`: default src/dst args `v6`/`v7`
- `configs/experiments/distillation_pipeline.yaml`: `run_suffix: "v7"`, W&B projects, cache dir

**v7 Teacher Models (no thinking mode):**
- `cyankiwi/Mistral-Small-4-119B-2603-AWQ-4bit`
- `cyankiwi/Qwen3.5-122B-A10B-AWQ-4bit`
- `cyankiwi/gemma-4-31B-it-AWQ-4bit`

To bump to v8, update the version constant in all submission scripts and the pipeline config above. The version is appended to output directories and W&B projects for traceability and to prevent collisions.

## Key Conventions

- **Config management**: Hydra (`configs/`) for training; YAML files under `configs/experiments/` for generation/training/eval scripts.
- **Experiment tracking**: Weights & Biases (wandb).
- **Model artifacts**: W&B artifact registry via `model_generation/artifacts.py`; registry project `intention-jailbreak-models` (opt-in per config).
- **HPC**: SLURM via scripts in `scripts/hpc/`. Most training jobs are submitted via Python submission scripts.
- **Inference**: vLLM used for all large LLM inference (generation, harm labeling, distillation).
- **LoRA**: All LLM fine-tuning uses LoRA/QLoRA adapters with QLoRA (4-bit NF4).
- **Attention**: `flash_attention_2` throughout (`flash-attn 2.8.3`, cu128/torch2.10/cp312 wheel). Sequence packing (`padding_free: true`) enabled in all training configs.
- **Early stopping**: Applied in all training runs via `EarlyStoppingCallback(patience=1)` in `causal.py`. All configs use `epochs: 5` as the ceiling; early stopping finds the actual stopping point.
- **Reproducibility**: Seeds set via `training/utils.py:set_all_seeds()`.

## Known Pitfalls

### vLLM LoRA key mismatch for multimodal students (Gemma3, Mistral3, Qwen3.5)

`_load_causal_lm` in `causal.py` extracts the language tower from `*ForConditionalGeneration` into a `CausalLM` wrapper for training. PEFT then saves adapter keys relative to that wrapper:
```
base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight
```
But vLLM serves the full `*ForConditionalGeneration` model, where the same module is at `language_model.model.layers.0...`. After stripping `base_model.model.`, vLLM looks for `model.layers.0...`, finds nothing, and **silently runs the base model** — no error is raised, but all adapter weights are ignored and outputs are identical across different adapters.

**Fix**: `_load_causal_lm` sets `model._vllm_lora_prefix = "language_model."` on affected models, and `_fix_lora_keys_for_vllm()` (called automatically after `trainer.save_model()`) renames the keys so vLLM finds them. For existing adapters, run `scripts/distillation/fix_adapter_keys.py`.

**Do NOT** set `enable_tower_connector_lora=True` in vLLM as a workaround — that flag is for applying LoRA to vision tower weights (which our text-only adapters don't have) and it crashes when combined with `limit_mm_per_prompt={"image": 0}`.

**Affected students**: `google/gemma-3-12b-it`, `mistralai/Ministral-3-14B-Reasoning-2512`. Llama 3.1 8B and `Qwen/Qwen3-8B` are pure CausalLMs and are unaffected.
