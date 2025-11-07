"""Language filtering utilities for dataset preprocessing."""

import gc
import hashlib
import pickle
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from fast_langdetect import detect_language
from tqdm import tqdm


def get_cache_path(df: pd.DataFrame, text_column: str, cache_dir: str = "data/cache") -> Path:
    """
    Generate a cache file path based on the dataset hash.
    
    Args:
        df: DataFrame to hash
        text_column: Name of the text column
        cache_dir: Directory to store cache files
    
    Returns:
        Path to the cache file
    """
    # Create a hash of the dataframe content for cache key
    # Use the text column content and index to create a unique identifier
    content_str = "".join(df[text_column].astype(str).values[:100])  # Sample first 100 for speed
    content_str += f"_len_{len(df)}_cols_{','.join(sorted(df.columns))}"
    
    hash_obj = hashlib.md5(content_str.encode())
    cache_hash = hash_obj.hexdigest()
    
    cache_path = Path(cache_dir) / f"lang_filter_{cache_hash}.pkl"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    
    return cache_path


def detect_english_texts(
    df: pd.DataFrame,
    text_column: str = "prompt",
    use_cache: bool = True,
    cache_dir: str = "data/cache",
    show_progress: bool = True
) -> pd.Series:
    """
    Detect which texts are in English using fast-langdetect.
    
    Args:
        df: DataFrame containing texts
        text_column: Name of the column containing text to check
        use_cache: Whether to use cached results if available
        cache_dir: Directory to store/load cache files
        show_progress: Whether to show progress bar
    
    Returns:
        Boolean Series indicating which rows are English
    """
    cache_path = get_cache_path(df, text_column, cache_dir)
    
    # Try to load from cache
    if use_cache and cache_path.exists():
        print(f"Loading language detection results from cache: {cache_path}")
        try:
            with open(cache_path, 'rb') as f:
                cached_results = pickle.load(f)
                if len(cached_results) == len(df):
                    print(f"✓ Loaded {len(cached_results)} cached language detection results")
                    return pd.Series(cached_results, index=df.index)
                else:
                    print(f"⚠ Cache size mismatch ({len(cached_results)} vs {len(df)}), recomputing...")
        except Exception as e:
            print(f"⚠ Failed to load cache: {e}, recomputing...")
    
    # Detect language for each text
    print(f"Detecting language for {len(df)} texts using fast-langdetect...")
    is_english = []
    
    texts = df[text_column].astype(str).values
    iterator = tqdm(texts, desc="Language detection") if show_progress else texts
    
    for text in iterator:
        try:
            # Truncate to 80 characters to avoid warning spam and improve speed
            # fast-langdetect works best with shorter texts anyway
            text_truncated = text[:80] if len(text) > 80 else text
            
            # fast-langdetect returns a language code string (uppercase)
            lang = detect_language(text_truncated)
            # Compare with uppercase 'EN'
            is_english.append(lang.upper() == 'EN')
        except Exception as e:
            # If detection fails, assume it's not English
            # print(f"Detection failed for text: {text[:50]}... Error: {e}")
            is_english.append(False)
    
    is_english_series = pd.Series(is_english, index=df.index)
    
    # Save to cache
    if use_cache:
        print(f"Saving language detection results to cache: {cache_path}")
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(is_english, f)
            print(f"✓ Cached {len(is_english)} language detection results")
        except Exception as e:
            print(f"⚠ Failed to save cache: {e}")
    
    # Print statistics
    n_english = is_english_series.sum()
    pct_english = 100 * n_english / len(df)
    print(f"Language detection results: {n_english}/{len(df)} ({pct_english:.1f}%) texts are English")
    
    return is_english_series


def filter_english_texts(
    df: pd.DataFrame,
    text_column: str = "prompt",
    use_cache: bool = True,
    cache_dir: str = "data/cache",
    show_progress: bool = True
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Filter DataFrame to keep only English texts.
    
    Args:
        df: DataFrame containing texts
        text_column: Name of the column containing text to check
        use_cache: Whether to use cached results if available
        cache_dir: Directory to store/load cache files
        show_progress: Whether to show progress bar
    
    Returns:
        Tuple of (filtered_df, is_english_mask)
    """
    is_english = detect_english_texts(
        df=df,
        text_column=text_column,
        use_cache=use_cache,
        cache_dir=cache_dir,
        show_progress=show_progress
    )
    
    filtered_df = df[is_english].copy().reset_index(drop=True)
    
    n_removed = len(df) - len(filtered_df)
    if n_removed > 0:
        print(f"Filtered out {n_removed} non-English texts ({100 * n_removed / len(df):.1f}%)")
    
    # Explicitly clean up to free memory
    del is_english
    gc.collect()
    
    return filtered_df, None
