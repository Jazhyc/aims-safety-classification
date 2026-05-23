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
│   └── report/                 # Scripts that generate report figures/tables (see "Report Artefacts" below)
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
| `scripts/` (root) | Multi-condition safety/harm classification + misc | `eval_safety_classifier.py`, `backfill_eval_timing.py` |
| `scripts/` (root, **do not modify**) | DPO + contrastive preference learning | `train_dpo.py`, `train_contrastive.py`, `generate_dpo_pairs.py`, `balance_dpo_pairs.py`, `compare_dpo_sweep.py`, `run_preference_pipeline.py` |
| `scripts/dataset_generation/` | ModernBERT classifier used to generate the annotation set via uncertainty filtering | `train.py`, `evaluate_test.py`, `plot_calibration.py` |
| `scripts/dataset_analysis/` | One-off scripts for analysing the annotated dataset | `generate_harm_labels.py`, `synthetic_comparison.py`, `evaluate_harm_predictions.py` |
| `scripts/hpc/` | SLURM submission + shell wrappers (keep as-is) | `submit_*.py`, `slurm_utils.py`, `*.sh` |
| `scripts/hpc/ablations/` | Ablation experiments (label-source, etc.) | `submit_label_source_ablation.py`, `prepare_ablation_samples.py` |
| `scripts/hpc/scaling/` | Data-scaling experiment (full WildGuardMix → gemma-3-12b distillation) | `submit_scaling.py`, `prepare_scaling_samples.py`, `verify_packing.py` |
| `scripts/report/` | Figure/table generation for the paper (see "Report Artefacts" below) | `generate_benchmark_table.py`, `plot_gemma_prompt_format.py`, `plot_qwen32b_reasoning_mode.py` |

## Notebook Categories

| Subdirectory | Purpose | Notebooks |
|---|---|---|
| `notebooks/baselines/` | Baseline intent generation/classification analysis | `eval_results_analysis.ipynb`, `wandb_llm_sweep_analysis.ipynb`, `safety_experiment_results.ipynb`, `eval_suite_timing.ipynb` |
| `notebooks/distillation/` | Teacher distillation sweep + reasoning trace analysis | `distillation_teacher_sweep.ipynb`, `reasoning_traces_analysis_llama.ipynb` |
| `notebooks/` (root, **do not modify**) | Preference learning results | `preference_learning_results.ipynb`, `diversity_analysis.ipynb` |
| `notebooks/modernbert/` | ModernBERT classifier analysis | `intent_classifier_analysis.ipynb`, `test_results_analysis.ipynb` |
| `notebooks/dataset_analysis/` | Old one-off notebooks for analysing the annotated dataset and its distribution | `annotation_analysis.ipynb`, `disagreement_analysis.ipynb`, `harm_labels_comparison.ipynb`, `harm_label_eval_4class_vs_binary.ipynb`, `wildguardmix_analysis.ipynb`, `clustering_analysis.ipynb`, `bert_harm_labels.ipynb` |

## Primary Dataset: `Jazhyc/wildguard-annotated-intents`

Human-annotated intents for ~1.7K WildGuardMix prompts. Loaded via HuggingFace `datasets`.

**Key properties:**
- Three splits on HuggingFace: `train`, `validation`, `test`
- Harm labels: 5-level (`Completely Harmful`, `Uncertain Harmful`, `Uncertain Safe`, `Completely Safe`, `Flag for Removal`)
- The `Dataset Harm` column ("Harmful"/"Safe") is the **original WildGuardMix** `prompt_harm_label` for that prompt — useful when you want the un-relabeled binary label without joining back to `allenai/wildguardmix`.

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
| `scripts/dpo/generate_dpo_pairs.py` | still uses manual split |

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

**Custom sample sources:** `generate_reasoning_traces.py` accepts an optional `dataset.samples_json` override that bypasses Hub loading. Records must include `prompt`, `prompt_harm_label`, and `split` keys; results are written to the split named in each record (typically all "train"). Used by the label-source ablation to feed deduped/random subsets through the same teacher pipeline.

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

## OOD Validation Pipeline

Model selection uses a two-stage eval workflow to avoid leaking test-set signal into adapter selection:

**Stage 1 — OOD validation** (`--mode ood-val`, default): trained adapters are evaluated on two held-out splits never seen during training:
- `lmsys/toxic-chat` subset `toxicchat0124`, **train** split
- `nvidia/Aegis-AI-Content-Safety-Dataset-2.0`, **validation** split

- **SFT**: every trained adapter is submitted.
- **Distillation**: only the best LR per (teacher × student × condition) by W&B `val_harm_f1` is submitted (compresses 192 → 48 jobs).

Config: `configs/experiments/eval_ood_validation.yaml`. Output: `data/safety_experiment/ood_validation/{sft|distillation}/...`

**Stage 2 — Test eval** (`--mode test`): reads OOD val JSONL files and submits test eval jobs based on OOD val F1.
- **SFT**: picks the best adapter per combination (falls back to W&B `val_harm_f1` if no OOD results).
- **Distillation**: picks the **single** adapter with the highest mean OOD val F1 across all (teacher × student × condition) combinations and submits one test job.

```bash
# Run OOD val for all trained adapters
python scripts/hpc/submit_sft_eval.py
python scripts/hpc/submit_distillation_eval.py

# After jobs finish — submit best adapter for test set eval
python scripts/hpc/submit_sft_eval.py --mode test
python scripts/hpc/submit_distillation_eval.py --mode test
```

**SFT OOD output layout:** `ood_validation/sft/{generation|classification}/{adapter_name}/{toxic_chat,aegis}/`
**Distillation OOD output layout:** `ood_validation/distillation/{teacher_slug}/{student_slug}/{condition}/{adapter_name}/{toxic_chat,aegis}/`

**Adapter deduplication:** if the same LR+epochs config was submitted more than once, only the most recently created W&B run per adapter path is kept (its weights are the ones actually on disk).

## Experiment Status

| Stage | Status | Notes |
|---|---|---|
| Prompting baselines (`eval_safety_classifier.py`) | ✅ Complete | W&B project: `Baselines`; results artifact: `safety-experiment-baselines` |
| SFT hyperparam sweep (`submit_hyperparam_sweep.py`) | 🔄 In progress | Llama 3.1 8B; W&B project: `sft-hyperparam-sweep`; model selected by OOD val F1 |
| Distillation teacher traces | ⬜ Pending | |
| Distillation student SFT | ⬜ Pending | |
| Label-source ablation (gpt-oss-120b → gemma-3-12b synthetic_intent) | ⬜ Pending | Compares `hard_human` (existing) vs `hard_original` (WG labels on hard subset) vs `random_original` (random WG sample). Config: `configs/experiments/ablations/label_source_ablation.yaml` |
| Data-scaling experiment (gpt-oss-120b → gemma-3-12b synthetic_intent on full WildGuardMix ~86.7k) | ⬜ Pending | Motivated by label-source ablation result that random_original (n=932) beat hard_human, suggesting the student is undertrained. 1 epoch on the full corpus, same per-combo best LR (2e-5), no HP sweep / val-set selection. Config: `configs/experiments/scaling/data_scaling.yaml` |
| Broader-intent-mix ablation (gpt-oss-120b → gemma-3-12b, human_intent + synthetic_intent) | ⬜ Pending | Tests whether adding an equal harm-stratified sample of broader-pool `synthetic_intent` traces (drawn from the scaling pool, annotated prompts excluded) to the annotated `human_intent` train set improves OOD generalisation. n=2,756 (1,378+1,378), same per-combo best LR (2e-5), standard 5-epoch ceiling with early stopping. Trace loader accepts a list-valued `reasoning_traces_condition`; per-record condition picks the intent source. Config: `configs/experiments/ablations/broader_intent_mix.yaml` |
| Re-run synthetic harm labeler with gpt-oss-120b for paper Table `tab:agreement_kappa` | ✅ Complete | Ran `vanilla_generation` over the full annotated set (n=1{,}724) via `scripts/hpc/baselines/eval_gpt_oss_120b_full_annotated.sh` → `data/safety_experiment/annotated-intents-full/openai_gpt-oss-120b_vanilla_generation.jsonl`. `tab:agreement_kappa` now has 3 rows ordered by κ: Human-Human (0.55/0.62), Human-GPT-OSS-120B (0.50/0.58), Human-WildGuardMix (0.45/—). Notebook also computes the GPT-OSS-120B-WildGuardMix row (κ=0.50) as a sanity reference, but it was dropped from the table as a confusing tangent. Numbers regenerated by `notebooks/dataset_analysis/agreement_table.ipynb`. Parse-fail policy: `predicted_harm=None` → `safe` so all non-pairwise rows share n=1{,}724. Appendix `app:synthetic-comparison` (Qwen-based) is now orphaned and should either be removed or rebuilt with the gpt-oss data — body text no longer references it. |
| External GRPO baseline (`iustinsirbu/llama-3.1-8b-grpo-intent-safety`) eval | ✅ Complete | New prior-work-style condition `grpo_classification` wired through `eval_safety_classifier.py` (system prompt in `configs/prompt_templates.yaml::safety_classifier.grpo_system_prompt`, function `run_grpo_classification` in `safety_classifier.py`, SLURM wrapper `scripts/hpc/baselines/eval_grpo_baseline.sh`). Output format: `<reasoning>...</reasoning>\nIntent: ...; Harm: ...`. Suite F1 = 0.836 (paper override), highest in the latency-vs-F1 comparison. |
| LE-DPO one-off eval (`Niclas-J-M/llama31-8b-k10-e-hard-b03-e3-s22-dpo-adapter`) | ✅ Complete | Adapter downloaded to `models/dpo/llama31-8b-k10-e-hard-b03-e3-s22_adapter/`, evaluated via `scripts/hpc/dpo/eval_oneoff_dpo_adapter.sh` against `configs/experiments/dpo/eval_dpo_condition.yaml` (5 OOD sets + annotated-intents train/val/test). Suite F1 = 0.812 (paper override). The SLURM script is reusable: `--export=ALL,ADAPTER_PATH=<local path>,OUTPUT_NAME=<slug>`. |

**Sweep model selection policy:** rank by **OOD val F1** (average across ToxicChat train + Aegis val) via the two-stage eval pipeline above. W&B `val_harm_f1` (training val) is only a fallback.

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
- `Qwen/Qwen3-32B`
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
- **Student prompt format**: Uses native chat templates (`tokenizer.apply_chat_template`). Instructions go in the **system** message; the prompt-to-classify goes in the **user** message. `build_student_messages(user_prompt, condition)` in `prompt_templates.py` builds the `[system, user]` list; callers apply `add_generation_prompt=True`. The teacher prompt in `generate_reasoning_traces.py` is intentionally kept as raw text (wraps everything in a user turn).
- **Where prompt text lives**: All prompt strings (preamble, taxonomy, output formats, plus all `safety_classifier.*` SFT/CoT prompts and `dpo.judge_intent_system_prompt`) are in `configs/prompt_templates.yaml`. `src/intention_jailbreak/model_generation/prompt_templates.py` loads them at import and exposes them as constants (`CLASSIFICATION_SYSTEM_PROMPT`, `GENERATION_SYSTEM_PROMPT`, `COT_CLASSIFICATION_SYSTEM_PROMPT`, `COT_GENERATION_SYSTEM_PROMPT`, `PREAMBLE`, `OUTPUT_FORMAT_WITH_INTENT`, …). When the paper needs the exact text of a prompt, read the YAML — do not re-derive from the code.
- **Validation set**: Training uses **val + test combined** (346 examples) as the early-stopping signal for both SFT and distillation paths. Neither split is held out during training; use external evaluation for unbiased final numbers.
- **Local metrics**: After every `run_causal_flow` call, `val_metrics.json` is written alongside the adapter (keys: `val_harm_f1`, `val_harm_precision`, `val_harm_recall`, `val_semantic_sim`, `val_eval_loss`, `model`, `condition`, `learning_rate`). Use this to compare runs without W&B.
- **Eval metrics on disk**: `eval_safety_classifier.py` writes a `<model_slug>_<condition>.metrics.json` sidecar next to every prediction `.jsonl` it produces (per-dataset accuracy/F1/`elapsed_s`/`total_tokens`/`tokens_per_second`), and a per-`(model_slug, condition)` `<model_slug>_<condition>_suite_summary.json` at `paths.output_dir` root with per-dataset entries plus a `suite` aggregate (`elapsed_s`, `total_tokens`, `total_examples`, `tokens_per_second`). Downstream notebooks read these and don't need W&B. For older runs that pre-date this change, `scripts/backfill_eval_timing.py` pulls the same numbers from W&B into the same file layout — extend its `MODELS` list to add another model.
- **`max_length_causal`**: Controls total token budget (prompt + completion) for SFT training truncation. The system message alone is ~300–400 tokens; 512 is too small for most examples.
- **Ablation adapters**: trained adapters from `scripts/hpc/ablations/` go to `models/distillation-ablations/` (NOT `models/distillation-sweep/`) so the main distillation eval submitter doesn't pick them up. Each ablation has its own `submit_*.py` orchestrator with `samples | traces | train | ood-val | test` modes.
- **Mixed-condition training**: `data.reasoning_traces_condition` accepts either a single string or a list (e.g. `[human_intent, synthetic_intent]`). With a list, `load_reasoning_traces_dataset` keeps records whose `condition` is in the set and picks the intent source per-record (`predicted.prompt_intent` for `synthetic_intent`, `ground_truth.intent` otherwise). All listed conditions must share the same template family (intent-producing vs `no_intent`) — the loader raises otherwise. The broader-intent-mix ablation uses this to combine annotated `human_intent` traces with broader-pool `synthetic_intent` traces in a single training set.
- **Scaling adapters**: same pattern — `scripts/hpc/scaling/` writes to `models/distillation-scaling/` and traces to `data/reasoning_traces_scaling/`. Modes: `samples | traces | train | test` (no `ood-val` since there's only one adapter — nothing to select between). Single-shot experiment: 1 epoch, no HP sweep, val set is informational only. The same-named `verify_packing.py` script runs an interactive comparison of `padding_free=True` with/without TRL `packing=True` to gate whether to enable the packing flag for the scaling run.

## Report Artefacts

Tables and figures in `report/latex/` are produced by scripts in `scripts/report/` — **do not edit the `.tex` files directly** unless the artefact has no generator. Edit the generator and re-run it so the numbers can be regenerated when results change.

| Artefact (`report/latex/...`) | Generator | Notes |
|---|---|---|
| `benchmark_table.tex` | `scripts/report/generate_benchmark_table.py` | Edit numbers/groups in `get_hardcoded_groups()` and re-run. W&B fetch is stubbed (TODO). |
| `img/gemma_prompt_format.pdf` | `scripts/report/plot_gemma_prompt_format.py` | |
| `img/qwen32b_reasoning_mode.pdf` | `scripts/report/plot_qwen32b_reasoning_mode.py` | |
| `img/pareto_f1_latency.pdf` | `notebooks/baselines/eval_suite_timing.ipynb` (scatter cell) | Mean Test F1 vs per-prompt latency scatter with Pareto frontier. Symmetric save bbox centres the axes for a LaTeX caption. |
| `pareto_latency_table.tex` | `notebooks/baselines/eval_suite_timing.ipynb` (table cell) | 12-row latency / tokens / mean-F1 table sorted by ms/prompt, includes LlamaGuard 4 and ShieldGemma 27B. |

Other `.tex` files (`acl_latex.tex`, `dpo_results.tex`, `dpo_validation.tex`, `GRPO_results.tex`) are hand-written prose/tables with no generator and should be edited directly.

### Eval-suite latency & Pareto analysis (`notebooks/baselines/eval_suite_timing.ipynb`)

Headline figure for the paper. Reads `<model_slug>_<condition>_suite_summary.json` files under `data/safety_experiment/` and emits the Pareto scatter + table above.

- **Models covered**: WildGuard 7B, Nemotron Safety 4B, GuardReasoner 8B, GPT OSS Safeguard 120B, LlamaGuard 4 12B, ShieldGemma 27B (prior-work), Llama-3.1-8B / Gemma-3-12B SFT (generation), Llama-3.1-8B LE-DPO, Llama-3.1-8B GRPO (= `iustinsirbu/llama-3.1-8b-grpo-intent-safety`), Llama-3.1-8B / Gemma-3-12B Distill (gpt-oss-120b teacher, synthetic_intent / human_intent respectively). LlamaGuard + ShieldGemma are excluded from the scatter to keep axes tight; both are in the LaTeX table.
- **Pareto frontier (latency vs OOD F1)**: Llama-3.1-8B SFT (4.66 ms, F1 0.791) → LE-DPO (5.52 ms, 0.812) → Llama Distill (20.19 ms, 0.820) → GRPO (25.28 ms, 0.836). Every prior-work classifier is dominated.
- **`paper_mean_ood_f1` override**: optional per-model field in the notebook's `MODELS` dict — set it when the paper reports a multi-seed mean that diverges from this notebook's single-eval number. Currently set for LE-DPO (0.812) and GRPO (0.836). All other models use the locally computed mean.
- **F1 fallback**: if a `_suite_summary.json` entry has `f1 == 0` (the W&B parser-failure case — see Known Pitfalls), the notebook recomputes F1 from the prediction `.jsonl` so the row still plots correctly (e.g. Gemma-3-12B SFT).
- **`ms/prompt` is the headline latency metric** — not `tokens/second`. tok/s is misleading for short-output models (prefill dominates wall-clock, so a fast classifier like LlamaGuard 4 looks "slow" on tok/s). All latency claims in the paper should use ms/prompt or examples/sec.
- **Hardware**: all rows pinned to a single `rtx_pro_6000` GPU partition under vLLM continuous batching, so the latency comparison is fair within ~hardware noise (~±0.005 F1 between repeats).
- **Output formats per prior-work classifier** (useful for paper text — token-count facts):
  - LlamaGuard 4 12B: bimodal — `safe` (3 tok) or `unsafe\nS<n>` (6 tok). Mean ~3.7 reflects safe/unsafe class balance.
  - WildGuard 7B: constant 23-tok 3-line `Harmful request / Response refusal / Harmful response` template; we only parse line 1.
  - Nemotron 4B: short reasoning + verdict (~70 tok).
  - gpt-oss-safeguard 120B: reasoning + 0/1 verdict in harmony `final` channel (~137 tok).
  - GuardReasoner 8B: long CoT (~286 tok).
  - ShieldGemma 27B: single token (Yes/No), but 27B base is memory-bound on prefill → still the slowest at 53 ms/prompt.

## Tests

Run with: `.venv/bin/python3.12 -m pytest tests/ -v` (activating the venv via `source` does not persist across bash commands — use the full path).

`tests/test_chat_template_prompts.py` covers: `build_student_messages` structure/content for all conditions, mock-tokenizer chat template integration, and round-trip tests for both assistant-turn parsers (`parse_reasoning_output` in `causal.py` and `_parse_reasoning_output` in `safety_classifier.py`).

## Known Pitfalls

### W&B `f1 = 0` for full-model evals (Gemma SFT generation)

`eval_safety_classifier.py` logs per-(dataset, condition) metrics including `accuracy`, `f1`, `precision`, `recall`, `total` to W&B at the end of each dataset. For at least one historical Gemma-3-12B SFT generation run, every metric logged as `0` while `elapsed_s` and `total_tokens` were correct — a log-time parser failure rather than a model failure (the per-example predictions in the `.jsonl` are intact and re-parse correctly).

**Symptom**: a suite summary with `f1: 0.0` across all datasets but realistic `elapsed_s`/`total_tokens` values; predicted_harm populated in the local `.jsonl`.

**Fix**: `notebooks/baselines/eval_suite_timing.ipynb` falls back to recomputing F1 from the `.jsonl` (`true_harm_binary` vs `predicted_harm`) when the summary's F1 is 0, so the row still appears on the Pareto scatter. The `scripts/backfill_eval_timing.py` workflow inherits the same predictions and benefits from the same fallback in the notebook. If a fresh re-eval is needed (e.g. to repopulate W&B), re-run `eval_safety_classifier.py` for the affected `model.name` + condition.

### vLLM LoRA key mismatch for multimodal students (Gemma3, Mistral3, Qwen3.5)

`_load_causal_lm` in `causal.py` extracts the language tower from `*ForConditionalGeneration` into a `CausalLM` wrapper for training. PEFT then saves adapter keys relative to that wrapper:
```
base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight
```
But vLLM serves the full `*ForConditionalGeneration` model, where the same module is at `language_model.model.layers.0...`. After stripping `base_model.model.`, vLLM looks for `model.layers.0...`, finds nothing, and **silently runs the base model** — no error is raised, but all adapter weights are ignored and outputs are identical across different adapters.

**Fix**: `_load_causal_lm` sets `model._vllm_lora_prefix = "language_model."` on affected models, and `_fix_lora_keys_for_vllm()` (called automatically after `trainer.save_model()`) renames the keys so vLLM finds them. For existing adapters, run `scripts/distillation/fix_adapter_keys.py`.

**Do NOT** set `enable_tower_connector_lora=True` in vLLM as a workaround — that flag is for applying LoRA to vision tower weights (which our text-only adapters don't have) and it crashes when combined with `limit_mm_per_prompt={"image": 0}`.

**Affected students**: `google/gemma-3-12b-it`, `mistralai/Ministral-3-14B-Instruct-2512-BF16`. Llama 3.1 8B and `Qwen/Qwen3-8B` are pure CausalLMs and are unaffected.

### TRL sequence packing breaks for tower-extracted multimodal students

`training.packing=true` in `causal.py` (passed through to TRL's `SFTConfig`) causes a ~27% spike in eval_loss on the gemma-3-12b student vs the unpacked baseline at identical effective batch and data — verified by `scripts/hpc/scaling/verify_packing.py`. Symptom is consistent with cross-example attention contamination: the trainer is concatenating examples without resetting attention/position boundaries via flash-attn varlen, so packed examples cross-attend within the same sequence. Likely root cause is the same wrapping that motivates the vLLM LoRA-key fix — `_load_causal_lm` extracts the language tower from `*ForConditionalGeneration` into a `CausalLM` shell, and the resulting model probably doesn't expose the cumulative-seqlen attention path TRL needs.

**Rule**: do NOT enable `training.packing=true` for any of the multimodal-tower students (gemma-3-12b, ministral-3-14b). For pure-CausalLM students (Llama 3.1 8B, Qwen3-8B), re-run `verify_packing.py` first to confirm the loss matches the unpacked baseline before relying on the speedup. Note also that on small step counts the wall-clock comparison is uninformative (dominated by setup); compare on at least a few hundred steps.

Separately, `packing=true` blows up GPU memory because each packed sequence fills `max_length` tokens and the cross_entropy logits tensor (`batch × seq × vocab × 4B`) balloons. `verify_packing.py` works around this by lowering `per_device_train_batch_size` to 2 and bumping `gradient_accumulation` to 16 to keep the effective batch at 32.

### `map_harm_to_binary` substring trap on WildGuardMix labels

WildGuardMix's `prompt_harm_label` is `harmful` / `unharmful`. A naïve substring check (`if 'harmful' in harm_lower`) silently misclassifies `unharmful` as `harmful` because `'harmful'` is a substring of `'unharmful'`. Both `data_utils.py:map_harm_to_binary` and `safety_classifier.py:map_harm_to_binary` MUST handle the WG binary tokens with explicit equality checks BEFORE falling back to substring matching for the 4-category annotated-intents labels.

Symptom in the distillation pipeline: with `filter_teacher_disagreements=True`, the trainer reports something like `Filtered (harm disagreement): 491 / 932` for traces whose ground-truth labels are WG binary — every `unharmful` record is dropped, the surviving train set is all-harmful with `harm_label='harmful'`, and the student degenerates to predicting `harmful` for everything. Step counts in the SLURM log will roughly match the harmful-class count rather than the full train size.

The 4-category labels (`Completely Safe`, `Uncertain Harmful`, etc.) do not trigger this because none contain `'unharmful'` as a substring; the bug only surfaces when consumers feed raw WG labels through (e.g. the label-source ablation, or any future ablation that uses `Dataset Harm` / `prompt_harm_label` directly).

### Reasoning-trace parser path (do not reintroduce schema parser)

`parse_model_output` in `scripts/distillation/generate_reasoning_traces.py` must always route to `_parse_content_fields` (LAST-match per regex, `re.IGNORECASE | re.DOTALL`). Do NOT add back a `tokenizer.parse_response`-based path: HF's schema parser matches FIRST occurrence and does not apply `IGNORECASE`, so the reasoning regex's `(?:Prompt )?intent:` lookahead silently fails to match `Intent:` (capital I) and the reasoning capture extends to `Prompt harm:`, sweeping the polished `Intent:` text into reasoning. The intent regex then captures FIRST `Intent:`, which lands inside any residual thinking content (gpt-oss harmony emits `Intent:` markers in its `analysis`/`assistantfinal` thinking section).

Symptom: `reasoning` strings contain `Intent:` substrings, `predicted.prompt_intent` values are >400 chars and look like the teacher's CoT thinking instead of the polished intent. Quick check: `grep -c '"Intent:' parsed_results.json` should be 0.

If a regression slips through, raw_outputs.json is sufficient to re-parse in place — see the inline reparser pattern used in the label-source ablation (`scripts/hpc/ablations/submit_label_source_ablation.py` history). `scripts/distillation/reparse_traces.py` is stale (imports `parse_response` / `_extract_fields`, neither of which exist anymore) — fix or replace it before relying on it.

### SLURM `--export` truncates values containing commas

`scripts/hpc/slurm_utils.submit_sbatch` builds the export string as `K1=V1,K2=V2,…`. SLURM's `--export` splits that value on commas to delimit variables, so any `V` containing a literal comma silently truncates: `--export=CONDITIONS_LIST=human_intent,synthetic_intent,…` makes SLURM set `CONDITIONS_LIST=human_intent` and treat `synthetic_intent=` as a separate (empty) var. The downstream job sees only the prefix and runs on the wrong data with no error.

Symptom (broader-intent-mix bug): training reported `Loaded 1374 examples (conditions=['human_intent'])` and ran 215 steps instead of the expected 2752 examples / 430 steps. The adapter trained as a human_intent-only run, indistinguishable from the existing v7 human_intent adapter.

**Rule**: never pass commas through `submit_sbatch` export values. Use a different separator (e.g. `|`) and decode in the bash template via `${VAR//|/,}`. As of this fix, `submit_sbatch` raises `ValueError` on comma-containing values so the bug fails loudly instead of silently.
