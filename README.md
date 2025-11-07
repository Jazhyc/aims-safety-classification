# Intention Jailbreak

Research project for using verbalized intention analysis to improve jailbreak mitigation in large language models.

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for Python package management.

1. Run the following command to create a virtual environment and install the project along with all dependencies:

```bash
uv sync
```

This will install the `intention_jailbreak` package in editable mode, so changes to the source code are immediately reflected.

2. Activate the virtual environment:

```bash
source .venv/bin/activate
```

Alternatively, you can run commands directly with uv without activating the environment:

```bash
uv run python your_script.py
```

For faster training, it is strongly recommended to install flash attention when using ModernBERT. The pre-installed wheels can be found [here](https://github.com/Dao-AILab/flash-attention/releases/tag/v2.8.3).

## Project Structure

```
intention-jailbreak/
├── configs/                      # Hydra configuration
│   ├── train_config.yaml
│   └── sweep_config.yaml
│
├── data/                         # Data (gitignored)
│   ├── annotations/
│   ├── embeddings/
│   ├── cache/
│   └── test_predictions/
│
├── logs/                         # Logs (gitignored)
│   ├── hydra/
│   └── wandb/
│
├── models/                       # Trained models (gitignored)
│   ├── modernbert-classifier/
│   └── modernbert-ensemble/
│
├── notebooks/                    # Analysis notebooks
│   ├── clustering_analysis.ipynb
│   ├── annotation_analysis.ipynb
│   ├── test_results_analysis.ipynb
│   └── wildguardmix_analysis.ipynb
│
├── notes/                        # Documentation
├── scripts/                      # Training and evaluation
├── src/intention_jailbreak/      # Main library
└── tests/                        # Test scripts
```

## Workflow

This project follows a multi-stage pipeline for training and analyzing intent classification models:

### 1. Training

Train the model using the main training script:

```bash
python scripts/train.py

# Override config parameters as needed
python scripts/train.py training.per_device_train_batch_size=128
```

**Important Configuration Options** (edit in `configs/train_config.yaml`):
- `ensemble.enabled`: Enable/disable ensemble mode (multiple models)
- `ensemble.num_models`: Number of models in the ensemble
- `model.context_window`: Model context window size
- `training.learning_rate`: Learning rate
- `training.num_epochs`: Number of training epochs

**Logging:**
- Training and validation results are logged to Weights & Biases (W&B)
- Configure W&B settings in `configs/train_config.yaml` to run in offline mode if desired
- Results are saved to `logs/` and `models/` directories

**Note:** You might get graph breaks when using torch.compile. These can be safely ignored.

### 2. Test Set Evaluation

Evaluate the trained model on the test set using the test script and analysis notebook:

- **Script:** `scripts/evaluate_test.py`
- **Notebook:** `notebooks/test_results_analysis.ipynb`
- **Functionality:**
  - Generates predictions on the test set
  - Produces statistics on the annotation set and examines uncertain samples for annotation by sweeping the model's confidence intervals
  - Exports results as a JSON file with the desired confidence interval

**Model Calibration:**
- The ensemble of 3 models was evaluated on the annotation set using a calibration curve
- The model was found to be mostly **underconfident** (predicted probabilities lower than actual accuracy)
- Calibration error: **0.27** (refer to `scripts/plot_calibration.py` for visualization)

### 3. Annotation with Label Studio

Use the JSON output from step 2 to annotate uncertain predictions:

1. Install and host a [Label Studio](https://labelstud.io/) instance
2. Import the JSON file from step 2 into Label Studio
3. Annotate the samples following the labeling guidelines (see annotation protocol in `/notes`)
4. Export the annotated data and save to `data/annotations/`

### 4. Annotation Analysis

Process and analyze the annotated data using the annotation analysis notebook:

- **Notebook:** `notebooks/annotation_analysis.ipynb`
- **Functionality:**
  - Converts Label Studio JSON output to parquet format
  - Applies data filtering by removing flagged samples
  - Generates statistics about annotated intents
  - Outputs a cleaned parquet file

**Note:** A processed parquet file is available at [Jazhyc/wildguard-annotated-intents](https://huggingface.co/datasets/Jazhyc/wildguard-annotated-intents) on Hugging Face.

### 5. Intent Clustering

Analyze how intents are clustered using semantic embeddings:

- **Notebook:** `notebooks/clustering_analysis.ipynb`
- **Functionality:**
  - Uses BERTopic for automatic topic discovery
  - Provides an interactive interface for manual topic labeling
  - Caches embeddings for efficient recomputation
  - Produces a topic-label mapping saved as JSON

### 6. Inter-Annotator Disagreement Analysis

Analyze how consistently human annotators label and describe intents:

- **Notebook:** `notebooks/disagreement_analysis.ipynb`
- **Functionality:**
  - Measures annotator agreement using label entropy, majority fraction, and score range
  - Computes intent paraphrase similarity using TF-IDF and cosine similarity
  - Identifies four types of prompts based on agreement patterns:
    - **High similarity, low disagreement:** Clear and uncontroversial examples
    - **High similarity, high disagreement:** Agreed intent, different harm assessments
    - **Low similarity, high disagreement:** Ambiguous prompts with different interpretations
    - **Low similarity, low disagreement:** Diverse phrasings with consistent harm judgment
  - Provides concrete examples of each disagreement pattern

### 7. Synthetic Intent Generation and Comparison to Human Annotators

- **Script:** `scripts/synthetic_comparison.py` (more details in corresponding markdown file in `notes/`)
- **Functionality:**
  - Generate synthetic intents using an open source LLM
  - Generate synthetic harm labels (completely harmful, uncertain harmful, uncertain safe, completely safe) using an open source LLM
  - Apply BERTopic for automatic topic discovery on both the synthetic and human intents
  - Visualize and compare human and synthetic intents
  - Compute confusion matrix of synthethic harm labels w.r.t. human labels.

### 8. Additional Analysis

For detailed findings and insights from each step, refer to the `notes/` folder which documents:
- Dataset analysis and statistics
- Training and Hyperparameter setup
- Intent Annotation Protocol
- Analysis on the annotated intents
- Comparison of synthetic intents to human annotated intents
- And more detailed technical explorations
