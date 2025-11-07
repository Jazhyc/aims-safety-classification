"""
Convenience functions for loading and preparing the WildGuardMix dataset.

Example usage:
    from intention_jailbreak.dataset import wildguardmix
    
    # Load and split the dataset
    train_df, test_df = wildguardmix.load_and_split()
    
    # Or load without splitting
    dataset = wildguardmix.load()
    df = dataset['train'].to_pandas()
"""

from typing import Tuple
import pandas as pd
from . import load_wildguardmix, create_stratified_split


def load(subset: str = 'wildguardtrain') -> pd.DataFrame:
    """
    Load the WildGuardMix dataset and convert to pandas DataFrame.
    
    Args:
        subset: The subset to load (default: 'wildguardtrain').
    
    Returns:
        pandas DataFrame with the dataset.
    """
    dataset = load_wildguardmix(subset)
    
    # Get the train split (or first available split)
    split_name = 'train' if 'train' in dataset else list(dataset.keys())[0]
    df = dataset[split_name].to_pandas()
    
    return df


def load_and_split(
    subset: str = 'wildguardtrain',
    test_size: float = 0.2,
    random_state: int = 42,
    filter_english: bool = False,
    text_column: str = 'prompt',
    language_cache_dir: str = 'data/cache'
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the WildGuardMix dataset and create a stratified train/test split.
    
    Args:
        subset: The subset to load (default: 'wildguardtrain').
        test_size: Proportion for test set (default: 0.4 for 60:40 split).
        random_state: Random state for reproducibility (default: 42).
        filter_english: Whether to filter for English texts only (default: False).
        text_column: Name of the text column for language filtering (default: 'prompt').
        language_cache_dir: Directory for language detection cache (default: 'data/cache').
    
    Returns:
        Tuple of (train_df, test_df).
    """
    df = load(subset)
    
    # Filter for English texts if requested (before splitting)
    if filter_english:
        from .language_filter import filter_english_texts
        print(f"\n=== Filtering dataset for English texts (before split) ===")
        print(f"Original dataset size: {len(df)}")
        df, _ = filter_english_texts(
            df,
            text_column=text_column,
            use_cache=True,
            cache_dir=language_cache_dir,
            show_progress=True
        )
        print(f"Filtered dataset size: {len(df)}")
        print("=== English filtering complete ===\n")
    
    train_df, test_df = create_stratified_split(df, test_size, random_state)
    
    return train_df, test_df
