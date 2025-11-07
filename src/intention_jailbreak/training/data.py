"""Data preparation utilities for classification tasks."""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from datasets import Dataset

from ..common import STRATIFICATION_COLUMNS


def prepare_classification_data(
    train_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame] = None,
    val_size: float = 0.1,
    test_size: float = 0.0,
    label_column: str = 'prompt_harm_label',
    text_column: str = 'prompt',
    positive_label: str = 'harmful',
    random_state: int = 42
):
    """
    Prepare data for binary classification with train, validation, and optional test splits.
    
    Uses stratified split on STRATIFICATION_COLUMNS to maintain the same 
    distribution in train, validation, and test sets.
    
    Args:
        train_df: Training dataframe
        test_df: Test dataframe (deprecated, kept for compatibility)
        val_size: Fraction of training data to use for validation
        test_size: Fraction of training data to use for test (default 0.0 for no test split)
        label_column: Name of the column containing labels
        text_column: Name of the column containing text
        positive_label: Label value that should be encoded as 1 (positive class)
        random_state: Random state for reproducibility
    
    Returns:
        If test_size > 0: Tuple of (train_dataset, val_dataset, test_dataset, train_df, val_df, test_df)
        If test_size == 0: Tuple of (train_dataset, val_dataset, train_df, val_df)
        where datasets are HuggingFace Datasets and dataframes are pandas DataFrames with all original columns plus 'label'
    """
    # Create combined stratification key using shared columns
    stratify_key = train_df[STRATIFICATION_COLUMNS].astype(str).agg('_'.join, axis=1)
    
    # First split off test set if requested
    if test_size > 0:
        train_val_df, test_df = train_test_split(
            train_df,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_key
        )
        # Update stratification key for remaining data
        stratify_key = train_val_df[STRATIFICATION_COLUMNS].astype(str).agg('_'.join, axis=1)
    else:
        train_val_df = train_df
        test_df = None
    
    # Split train/val from remaining data
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size,
        random_state=random_state,
        stratify=stratify_key
    )
    
    print(f"Train: {len(train_df)} examples")
    print(f"Validation: {len(val_df)} examples")
    if test_df is not None:
        print(f"Test: {len(test_df)} examples")
    
    def add_binary_labels(df):
        df = df.copy()
        df['label'] = (df[label_column] == positive_label).astype(int)
        return df
    
    train_df = add_binary_labels(train_df)
    val_df = add_binary_labels(val_df)
    
    if test_df is not None:
        test_df = add_binary_labels(test_df)
        print(f"Positive class ({positive_label}): Train {train_df['label'].mean()*100:.1f}%, Val {val_df['label'].mean()*100:.1f}%, Test {test_df['label'].mean()*100:.1f}%")
    else:
        print(f"Positive class ({positive_label}): Train {train_df['label'].mean()*100:.1f}%, Val {val_df['label'].mean()*100:.1f}%")
    
    # Keep only necessary columns for dataset (text and label)
    # Note: We'll add 'weight' column later if class weighting is enabled
    train_dataset = Dataset.from_pandas(train_df[[text_column, 'label']].reset_index(drop=True))
    val_dataset = Dataset.from_pandas(val_df[[text_column, 'label']].reset_index(drop=True))
    
    if test_df is not None:
        test_dataset = Dataset.from_pandas(test_df[[text_column, 'label']].reset_index(drop=True))
        return train_dataset, val_dataset, test_dataset, train_df, val_df, test_df
    else:
        return train_dataset, val_dataset, train_df, val_df


def tokenize_dataset(dataset: Dataset, tokenizer, max_length: int, text_column: str = 'prompt', num_proc: int = 1):
    """
    Tokenize a dataset.
    
    Args:
        dataset: HuggingFace Dataset to tokenize
        tokenizer: Tokenizer to use
        max_length: Maximum sequence length
        text_column: Name of the text column to tokenize
        num_proc: Number of processes to use for parallel tokenization
    
    Returns:
        Tokenized dataset
    """
    def tokenize_function(examples):
        return tokenizer(
            examples[text_column],
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
    
    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=[text_column],
        num_proc=num_proc,
        desc="Tokenizing"
    )
    
    return tokenized


def compute_sample_weights(
    df: pd.DataFrame,
    label_column: str = 'label',
    weight_column: Optional[str] = None,
    use_label_weights: bool = True,
    use_subcategory_weights: bool = False
) -> np.ndarray:
    """
    Compute sample weights based on label frequency and/or subcategory frequency.
    
    Uses the formula: n_samples / (n_classes * np.bincount(y))
    
    Args:
        df: DataFrame with label and optionally subcategory columns
        label_column: Name of binary label column (0/1)
        weight_column: Column to compute subcategory weights from (e.g., 'subcategory')
        use_label_weights: Whether to weight by label (harmful vs unharmful) frequency
        use_subcategory_weights: Whether to weight by subcategory frequency
    
    Returns:
        Array of sample weights (one per sample in df)
    """
    n_samples = len(df)
    sample_weights = np.ones(n_samples)
    
    # Compute weights based on label (harmful/unharmful) if enabled
    if use_label_weights:
        labels = df[label_column].values
        unique_labels, encoded_labels = np.unique(labels, return_inverse=True)
        n_label_classes = len(unique_labels)
        label_counts = np.bincount(encoded_labels)
        label_class_weights = n_samples / (n_label_classes * label_counts)
        label_weights = label_class_weights[encoded_labels]
        sample_weights = sample_weights * label_weights
        
        print(f"Label weighting enabled:")
        print(f"  Label classes: {n_label_classes} (counts: {label_counts})")
        print(f"  Label weights: {label_class_weights}")
    
    # Compute weights based on subcategory if enabled
    if use_subcategory_weights:
        if weight_column is None:
            raise ValueError("weight_column must be specified when use_subcategory_weights=True")
        
        subcategories = df[weight_column].values
        unique_subcats, encoded_subcats = np.unique(subcategories, return_inverse=True)
        n_subcat_classes = len(unique_subcats)
        subcat_counts = np.bincount(encoded_subcats)
        subcat_class_weights = n_samples / (n_subcat_classes * subcat_counts)
        subcat_weights = subcat_class_weights[encoded_subcats]
        sample_weights = sample_weights * subcat_weights
        
        print(f"Subcategory weighting enabled ('{weight_column}'):")
        print(f"  Subcategories: {n_subcat_classes}")
        # Show per-subcategory statistics (first 5)
        for i, subcat in enumerate(unique_subcats[:5]):
            count = subcat_counts[i]
            weight = subcat_class_weights[i]
            print(f"    {subcat}: {count} samples, weight={weight:.3f}")
        if len(unique_subcats) > 5:
            print(f"    ... and {len(unique_subcats) - 5} more subcategories")
    
    # Normalize so mean weight = 1.0
    sample_weights = sample_weights / sample_weights.mean()
    
    print(f"Final sample weights:")
    print(f"  Weight range: [{sample_weights.min():.3f}, {sample_weights.max():.3f}]")
    print(f"  Mean weight: {sample_weights.mean():.3f}")
    
    return sample_weights
