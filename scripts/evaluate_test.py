"""
Evaluate trained model on test set and save predictions with probabilities.

This script:
1. Loads the trained model
2. Evaluates on the held-out test set
3. Saves predictions with harmful class probabilities
4. Logs results to wandb (classifier-annotation-set project)

Usage:
    python scripts/evaluate_test.py
"""

import os

from intention_jailbreak.ensemble.deepensembleclassifier import DeepEnsembleClassifier
from intention_jailbreak.utils.calibration import classifier_calibration_curve, classifier_calibration_error, plot_calibration_curve
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DIR"] = "logs/wandb"

from pathlib import Path
import pandas as pd
import torch
import wandb
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from intention_jailbreak.dataset import wildguardmix
from intention_jailbreak.training import compute_classification_metrics, print_gpu_info, tokenize_dataset
from intention_jailbreak.common import LABEL_COLUMN, TEXT_COLUMN, POSITIVE_LABEL


# Configuration Constants
MODEL_PATH = "models/modernbert-ensemble/final_model"
USE_ENSEMBLE = True
NUM_ENSEMBLE_MODELS = 3
BATCH_SIZE = 512
MAX_LENGTH = 2048
TEST_SIZE = 0.9  # 90% annotation set split
VAL_SIZE = 0.1  # 10% of training split for validation
TEST_SIZE_FROM_TRAIN = 0.1  # 10% of training split for test
RANDOM_STATE = 42
FILTER_ENGLISH = True
LANGUAGE_CACHE_DIR = "data/cache"
OUTPUT_DIR = Path("data/test_predictions")
WANDB_PROJECT = "classifier-annotation-set"
WANDB_ENTITY = "intention-analysis"
WANDB_RUN_NAME = "test-set-evaluation-ensemble"
NUM_CALIBRATION_BINS = 20
EVAL_THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]
NUM_WORKERS = 8  # Number of workers for data loading
PIN_MEMORY = True  # Pin memory for faster GPU transfer


def main():
    """Evaluate model on held-out test set and annotation set."""
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Initialize wandb
    wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=WANDB_RUN_NAME,
        dir="logs/wandb",
        config={
            "model_path": MODEL_PATH,
            "use_ensemble": USE_ENSEMBLE,
            "num_ensemble_models": NUM_ENSEMBLE_MODELS if USE_ENSEMBLE else 1,
            "batch_size": BATCH_SIZE,
            "max_length": MAX_LENGTH,
            "test_size": TEST_SIZE,
            "val_size": VAL_SIZE,
            "test_size_from_train": TEST_SIZE_FROM_TRAIN,
            "filter_english": FILTER_ENGLISH,
            "num_workers": NUM_WORKERS,
            "pin_memory": PIN_MEMORY,
            "dataset": "wildguardmix",
        }
    )
    
    print_gpu_info()
    
    # Load held-out test set (from training split)
    print("\n" + "="*80)
    print("STEP 1: Evaluating on held-out test set (from training split)")
    print("="*80)
    test_set_path = OUTPUT_DIR / "held_out_test_set.parquet"
    if not test_set_path.exists():
        raise FileNotFoundError(
            f"Held-out test set not found at {test_set_path}. "
            "Please run the training script first to create this file."
        )
    
    test_df = pd.read_parquet(test_set_path)
    print(f"\nHeld-out test set size: {len(test_df)}")
    
    # Load model and tokenizer
    print(f"\nLoading model from: {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    if USE_ENSEMBLE:
        model = DeepEnsembleClassifier.from_pretrained(
            MODEL_PATH,
            model_class=AutoModelForSequenceClassification,
            dtype=torch.bfloat16
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_PATH,
            dtype=torch.bfloat16
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model = torch.compile(model)
    model.eval()
    
    # Prepare labels
    test_df['label'] = (test_df[LABEL_COLUMN] == POSITIVE_LABEL).astype(int)
    
    # Create dataset
    print("\nPreparing dataset...")
    test_dataset = Dataset.from_pandas(test_df[[TEXT_COLUMN, 'label']])
    
    # Tokenize using the same function as training script
    print("Tokenizing...")
    test_dataset = tokenize_dataset(
        test_dataset,
        tokenizer,
        MAX_LENGTH,
        TEXT_COLUMN,
        num_proc=NUM_WORKERS
    )
    test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    
    # Create DataLoader with multiple workers
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=False
    )
    
    # Get predictions in batches
    print("\nGenerating predictions...")
    all_probs = []
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Evaluating"):
            # Move to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label']
            
            # Get predictions
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Handle ensemble vs single model outputs
            if hasattr(outputs, 'probs'):  # Ensemble model
                probs = outputs.probs
            else:  # Single model
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
            
            harmful_probs = probs[:, 1].float().cpu().numpy()  # Probability of harmful class
            preds = torch.argmax(probs, dim=-1).cpu().numpy()
            
            all_probs.extend(harmful_probs)
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    # Add predictions to dataframe
    test_df['harmful_probability'] = all_probs
    test_df['predicted_label'] = all_preds
    
    # Compute metrics
    print("\nComputing metrics...")
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', zero_division=0
    )
    
    metrics = {
        "test/accuracy": accuracy,
        "test/precision": precision,
        "test/recall": recall,
        "test/f1": f1,
        "test/samples": len(test_df)
    }
    
    print("\nTest Set Results:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    
    # Log to wandb
    wandb.log(metrics)
    
    # Log confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    wandb.log({
        "test/confusion_matrix": wandb.plot.confusion_matrix(
            probs=None,
            y_true=all_labels,
            preds=all_preds,
            class_names=["unharmful", "harmful"]
        )
    })
    
    # Log probability distribution
    wandb.log({
        "test/harmful_probability_dist": wandb.Histogram(all_probs),
        "test/harmful_prob_by_true_label": wandb.plot.histogram(
            wandb.Table(
                data=[[p, l] for p, l in zip(all_probs, all_labels)],
                columns=["harmful_probability", "true_label"]
            ),
            "harmful_probability",
            title="Harmful Probability Distribution by True Label"
        )
    })
    
    # Remove response column if it exists
    if 'response' in test_df.columns:
        test_df = test_df.drop(columns=['response'])
        print("\nRemoved 'response' column from dataset")
    
    # Save predictions
    output_file = OUTPUT_DIR / "test_predictions.parquet"
    test_df.to_parquet(output_file, index=False)
    print(f"\nPredictions saved to: {output_file}")
    
    # Log artifact to wandb
    artifact = wandb.Artifact(
        name="test-predictions",
        type="dataset",
        description="Test set with model predictions and harmful class probabilities"
    )
    artifact.add_file(output_file)
    wandb.log_artifact(artifact)
    
    # Create summary table
    summary_data = []
    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
        preds_at_threshold = (np.array(all_probs) >= threshold).astype(int)
        acc = accuracy_score(all_labels, preds_at_threshold)
        prec, rec, f1_t, _ = precision_recall_fscore_support(
            all_labels, preds_at_threshold, average='binary', zero_division=0
        )
        summary_data.append([threshold, acc, prec, rec, f1_t])
    
    summary_table = wandb.Table(
        data=summary_data,
        columns=["threshold", "accuracy", "precision", "recall", "f1"]
    )
    wandb.log({"test/metrics_by_threshold": summary_table})

    # Compute classifier calibration curve
    conf_curve, acc_curve = classifier_calibration_curve(
        y_pred=np.array(all_preds),
        y_true=np.array(all_labels),
        y_confidences=np.array(all_probs),
        num_bins=NUM_CALIBRATION_BINS
    )

    # Compute calibration error
    cal_error = classifier_calibration_error(
        y_pred=np.array(all_preds),
        y_true=np.array(all_labels),
        y_confidences=np.array(all_probs),
        num_bins=NUM_CALIBRATION_BINS,
    )

    cal_curve_path = OUTPUT_DIR / "classifier_calibration_curve.png"

    plot_calibration_curve(
        conf=conf_curve,
        acc=acc_curve,
        title="Classifier Calibration Curve",
        y_label="Empirical Accuracy",
        relative_save_path=cal_curve_path
    )

    print(f"Classifier Calibration Error: {cal_error:.4f}")
    wandb.log({
        "test/calibration_error_weighted": cal_error
    })

    artifact_cal = wandb.Artifact(
        name="classifier_calibration_curve",
        type="figure",
        description="Classifier calibration curve (probabilities vs empirical accuracy)"
    )
    artifact_cal.add_file(cal_curve_path)
    wandb.log_artifact(artifact_cal)
    
    print("\n" + "="*80)
    print("STEP 2: Computing probabilities for annotation set")
    print("="*80)
    
    # Load annotation set
    print("\nLoading annotation set...")
    _, annotation_df = wildguardmix.load_and_split(
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        filter_english=FILTER_ENGLISH,
        text_column='prompt',
        language_cache_dir=LANGUAGE_CACHE_DIR
    )
    print(f"Annotation set size: {len(annotation_df)}")
    
    # Prepare labels
    annotation_df['label'] = (annotation_df[LABEL_COLUMN] == POSITIVE_LABEL).astype(int)
    
    # Create dataset
    print("\nPreparing annotation dataset...")
    annotation_dataset = Dataset.from_pandas(annotation_df[[TEXT_COLUMN, 'label']])
    
    # Tokenize
    print("Tokenizing...")
    annotation_dataset = tokenize_dataset(
        annotation_dataset,
        tokenizer,
        MAX_LENGTH,
        TEXT_COLUMN,
        num_proc=NUM_WORKERS
    )
    annotation_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    
    # Create DataLoader
    annotation_dataloader = DataLoader(
        annotation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=False
    )
    
    # Get predictions in batches
    print("\nGenerating predictions for annotation set...")
    annotation_probs = []
    annotation_preds = []
    annotation_labels = []
    
    with torch.no_grad():
        for batch in tqdm(annotation_dataloader, desc="Evaluating annotation set"):
            # Move to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label']
            
            # Get predictions
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Handle ensemble vs single model outputs
            if hasattr(outputs, 'probs'):  # Ensemble model
                probs = outputs.probs
            else:  # Single model
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
            
            harmful_probs = probs[:, 1].float().cpu().numpy()
            preds = torch.argmax(probs, dim=-1).cpu().numpy()
            
            annotation_probs.extend(harmful_probs)
            annotation_preds.extend(preds)
            annotation_labels.extend(labels.cpu().numpy())
    
    # Add predictions to dataframe
    annotation_df['harmful_probability'] = annotation_probs
    annotation_df['predicted_label'] = annotation_preds
    
    # Compute metrics for annotation set
    print("\nComputing metrics for annotation set...")
    annotation_accuracy = accuracy_score(annotation_labels, annotation_preds)
    annotation_precision, annotation_recall, annotation_f1, _ = precision_recall_fscore_support(
        annotation_labels, annotation_preds, average='binary', zero_division=0
    )
    
    annotation_metrics = {
        "annotation/accuracy": annotation_accuracy,
        "annotation/precision": annotation_precision,
        "annotation/recall": annotation_recall,
        "annotation/f1": annotation_f1,
        "annotation/samples": len(annotation_df)
    }
    
    print("\nAnnotation Set Results:")
    print(f"  Accuracy:  {annotation_accuracy:.4f}")
    print(f"  Precision: {annotation_precision:.4f}")
    print(f"  Recall:    {annotation_recall:.4f}")
    print(f"  F1 Score:  {annotation_f1:.4f}")
    
    # Log to wandb
    wandb.log(annotation_metrics)
    
    # Log confusion matrix for annotation set
    annotation_cm = confusion_matrix(annotation_labels, annotation_preds)
    wandb.log({
        "annotation/confusion_matrix": wandb.plot.confusion_matrix(
            probs=None,
            y_true=annotation_labels,
            preds=annotation_preds,
            class_names=["unharmful", "harmful"]
        )
    })
    
    # Log probability distribution for annotation set
    wandb.log({
        "annotation/harmful_probability_dist": wandb.Histogram(annotation_probs),
        "annotation/harmful_prob_by_true_label": wandb.plot.histogram(
            wandb.Table(
                data=[[p, l] for p, l in zip(annotation_probs, annotation_labels)],
                columns=["harmful_probability", "true_label"]
            ),
            "harmful_probability",
            title="Annotation Set - Harmful Probability Distribution by True Label"
        )
    })
    
    # Remove response column if it exists
    if 'response' in annotation_df.columns:
        annotation_df = annotation_df.drop(columns=['response'])
        print("\nRemoved 'response' column from annotation dataset")
    
    # Save annotation set predictions
    annotation_output_file = OUTPUT_DIR / "annotation_predictions.parquet"
    annotation_df.to_parquet(annotation_output_file, index=False)
    print(f"\nAnnotation predictions saved to: {annotation_output_file}")
    
    # Log artifact to wandb
    annotation_artifact = wandb.Artifact(
        name="annotation-predictions",
        type="dataset",
        description="Annotation set with model predictions and harmful class probabilities"
    )
    annotation_artifact.add_file(annotation_output_file)
    wandb.log_artifact(annotation_artifact)
    
    # Create summary table for annotation set
    annotation_summary_data = []
    for threshold in EVAL_THRESHOLDS:
        preds_at_threshold = (np.array(annotation_probs) >= threshold).astype(int)
        acc = accuracy_score(annotation_labels, preds_at_threshold)
        prec, rec, f1_t, _ = precision_recall_fscore_support(
            annotation_labels, preds_at_threshold, average='binary', zero_division=0
        )
        annotation_summary_data.append([threshold, acc, prec, rec, f1_t])
    
    annotation_summary_table = wandb.Table(
        data=annotation_summary_data,
        columns=["threshold", "accuracy", "precision", "recall", "f1"]
    )
    wandb.log({"annotation/metrics_by_threshold": annotation_summary_table})
    
    # Compute calibration for annotation set
    annotation_conf_curve, annotation_acc_curve = classifier_calibration_curve(
        y_pred=np.array(annotation_preds),
        y_true=np.array(annotation_labels),
        y_confidences=np.array(annotation_probs),
        num_bins=NUM_CALIBRATION_BINS
    )
    
    annotation_cal_error = classifier_calibration_error(
        y_pred=np.array(annotation_preds),
        y_true=np.array(annotation_labels),
        y_confidences=np.array(annotation_probs),
        num_bins=NUM_CALIBRATION_BINS,
    )
    
    annotation_cal_curve_path = OUTPUT_DIR / "annotation_calibration_curve.png"
    
    plot_calibration_curve(
        conf=annotation_conf_curve,
        acc=annotation_acc_curve,
        title="Annotation Set - Classifier Calibration Curve",
        y_label="Empirical Accuracy",
        relative_save_path=annotation_cal_curve_path
    )
    
    print(f"Annotation Set Calibration Error: {annotation_cal_error:.4f}")
    wandb.log({
        "annotation/calibration_error_weighted": annotation_cal_error
    })
    
    annotation_artifact_cal = wandb.Artifact(
        name="annotation_calibration_curve",
        type="figure",
        description="Annotation set calibration curve (probabilities vs empirical accuracy)"
    )
    annotation_artifact_cal.add_file(annotation_cal_curve_path)
    wandb.log_artifact(annotation_artifact_cal)
    
    wandb.finish()
    
    print("\n" + "="*80)
    print("Evaluation complete!")
    print("="*80)
    print(f"\nHeld-out test set: {len(test_df)} samples")
    print(f"  Saved to: {output_file}")
    print(f"\nAnnotation set: {len(annotation_df)} samples")
    print(f"  Saved to: {annotation_output_file}")
    print(f"\nHarmful probability column added: 'harmful_probability'")
    print(f"  (Probability that the prompt is classified as harmful)")


if __name__ == "__main__":
    main()
