<div align="center">
  <h1>Paved with True Intents</h1>
  <p><strong>Intent-Aware Training Improves LLM Safety Classification Across Training Regimes</strong></p>
  <p>
    <a href="https://arxiv.org/abs/2606.27210"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2606.27210-b31b1b?logo=arxiv&logoColor=white"></a>
    <a href="https://jazhyc.github.io/aims-safety/"><img alt="Project page" src="https://img.shields.io/badge/Project-Page-2f80ed"></a>
    <a href="https://huggingface.co/collections/Jazhyc/aims-intent-aware-safety-classification"><img alt="Hugging Face collection" src="https://img.shields.io/badge/Hugging_Face-AIMS_Collection-ffd21e?logo=huggingface&logoColor=black"></a>
  </p>
  <p>Official repository · Accepted to <strong>Findings of EMNLP 2026</strong></p>
</div>

## Overview

Safety classifiers often rely on surface features of a prompt. This project instead treats the user's underlying intent as an explicit supervision signal between the prompt and its harm label.

We introduce **AIMS**, a dataset of 1,724 difficult safety prompts with human-written intent descriptions and harm annotations, and study intent-aware learning across four training regimes:

- supervised fine-tuning (SFT);
- label-error-driven direct preference optimization (LE-DPO);
- reasoning distillation; and
- group relative policy optimization (GRPO).

Across five external safety benchmarks, intent-aware models improve the accuracy-efficiency trade-off over the evaluated prior-work classifiers. See the paper for the complete experimental setup, results, and limitations.

## Released resources

| Resource | Description |
|---|---|
| [AIMS dataset](https://huggingface.co/datasets/Jazhyc/aims-safety-intents) | Human-written intents and harm annotations for 1,724 difficult WildGuardMix prompts |
| [SFT model](https://huggingface.co/Jazhyc/Llama-3.1-8B-aims-sft-generation) | Llama 3.1 8B trained to generate intent and harm labels |
| [LE-DPO model](https://huggingface.co/Jazhyc/Llama-3.1-8B-aims-le-dpo) | Preference-trained model targeting label-flipping intent errors |
| [Distilled model](https://huggingface.co/Jazhyc/Llama-3.1-8B-aims-distill-synthetic-intent) | Llama 3.1 8B distilled from synthetic-intent reasoning traces |
| [GRPO model](https://huggingface.co/Jazhyc/Llama-3.1-8B-aims-grpo) | Model trained with an explicit intent-faithfulness reward |

The Hugging Face [AIMS collection](https://huggingface.co/collections/Jazhyc/aims-intent-aware-safety-classification) groups the dataset and all released checkpoints.

## Dataset

Load the three official AIMS splits directly from Hugging Face:

```python
from datasets import load_dataset

dataset = load_dataset("Jazhyc/aims-safety-intents")
train = dataset["train"]
validation = dataset["validation"]
test = dataset["test"]
```

Prompts with duplicate text are restricted to the training split to prevent cross-split leakage. Validation and test contain only unique prompts and are stratified by the human harm annotation. Repository code should load these published splits through `preprocess_data()` rather than creating a new random split.

AIMS is deliberately enriched for ambiguous, adversarial, and borderline prompts. It is useful for studying intent-aware safety classification, but it is English-only and is not representative of organic production traffic. See the dataset card for its full schema, provenance, license, and limitations.

## Installation

The project uses Python 3.12 and [uv](https://docs.astral.sh/uv/) for dependency management:

```bash
git clone https://github.com/Jazhyc/aims-safety-classification.git
cd aims-safety-classification
uv sync
source .venv/bin/activate
```

The dependency set includes a Linux CUDA 12.8 / PyTorch 2.10 FlashAttention wheel and is intended primarily for GPU research environments. Model downloads can be redirected to scratch storage by setting `HF_HOME` before running a command:

```bash
export HF_HOME=/scratch/$USER/huggingface
```

## Repository structure

```text
aims-safety-classification/
├── configs/experiments/       # Training, generation, and evaluation configurations
├── notebooks/                 # Analysis notebooks grouped by experiment family
├── scripts/
│   ├── baselines/             # SFT and prompting baselines
│   ├── distillation/          # Trace generation and student distillation
│   ├── dpo/                   # Preference-data preparation and DPO training
│   ├── dataset_generation/    # ModernBERT annotation-set classifier
│   ├── dataset_analysis/      # Dataset analysis utilities
│   ├── hpc/                   # SLURM submission and experiment orchestration
│   └── report/                # Paper figure and table generators
├── src/intention_jailbreak/   # Reusable dataset, training, inference, and analysis code
├── tests/                     # Regression tests
└── AGENTS.md                  # Detailed contributor and experiment guide
```

Generated data, model weights, checkpoints, and logs are intentionally gitignored.

## Core workflows

Experiment behavior is configured with Hydra YAML files under `configs/experiments/`. Common entry points include:

```bash
# Evaluate prompting, prior-work, or fine-tuned safety classifiers
.venv/bin/python scripts/eval_safety_classifier.py \
  --config-name=eval_safety_classifier

# Train an SFT or reasoning-distilled generator
.venv/bin/python scripts/baselines/train_generator.py \
  --config-name=llm_sweep

# Generate teacher reasoning traces
.venv/bin/python scripts/distillation/generate_reasoning_traces.py \
  --config-name=reasoning_traces
```

Large sweeps and evaluations are orchestrated through the scripts under `scripts/hpc/`. Review the selected YAML and shell submission script before launching a run: many workflows require multiple GPUs, external model access, W&B credentials, or paid API inference.

### GRPO training with veRL

The original GRPO training pipeline is maintained on the [`grpo` branch](https://github.com/Jazhyc/aims-safety-classification/tree/grpo). It uses [veRL](https://github.com/verl-project/verl), Apptainer, SLURM, eight GPUs, and a separately served judge model. It remains separate from the main training stack because it has its own container and cluster configuration.

The `main` branch contains the GRPO model's evaluation integration and the analyses used in the paper.

## Evaluation policy

Adapter selection uses external validation data from ToxicChat and Aegis rather than test-suite performance. Final evaluation covers five external safety benchmarks. Evaluation runs write prediction JSONL files, per-dataset metric sidecars, and suite summaries so downstream notebooks do not depend on W&B state.

The repository contains research pipelines and prompt-level safety classifiers. Their outputs should not be treated as the sole authority for high-stakes moderation decisions.

## Contributing

See [AGENTS.md](AGENTS.md) for the detailed repository map, experiment conventions, dataset split policy, artifact layout, validation workflow, and known pitfalls. Several subdirectories contain generated or independently versioned research artefacts, so check the relevant repository boundary before committing changes.

## Citation

Please cite the arXiv paper:

```bibtex
@misc{ferrao2026pavedtrueintentsintentaware,
  title         = {Paved with True Intents: Intent-Aware Training Improves LLM Safety Classification Across Training Regimes},
  author        = {Jeremias Ferrao and Niclas Müller-Hof and Iustin Sîrbu and Traian Rebedea and Yftah Ziser},
  year          = {2026},
  eprint        = {2606.27210},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2606.27210}
}
```

The paper has been accepted to Findings of EMNLP 2026; the arXiv record remains the citation source until proceedings metadata is available.
