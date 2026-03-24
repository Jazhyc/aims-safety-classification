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

**Do not touch DPO/contrastive scripts** — those are owned by another team member.

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

## Secondary Dataset: WildGuardMix

Used for ModernBERT classifier training. Loaded via `src/intention_jailbreak/dataset/wildguardmix.py`.
Stratified split on `prompt_harm_label`, `adversarial`, `subcategory`. English-only filtering via `language_filter.py`.

## Key Conventions

- **Config management**: Hydra (`configs/`) for training; YAML files under `configs/experiments/` for generation/training/eval scripts.
- **Experiment tracking**: Weights & Biases (wandb).
- **HPC**: SLURM via scripts in `scripts/hpc/`. Most training jobs are submitted via Python submission scripts.
- **Inference**: vLLM used for all large LLM inference (generation, harm labeling, distillation).
- **LoRA**: All LLM fine-tuning uses LoRA/QLoRA adapters.
- **Reproducibility**: Seeds set via `training/utils.py:set_all_seeds()`.
