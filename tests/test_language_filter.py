"""Test language filtering functionality."""

import pandas as pd
from intention_jailbreak.dataset import filter_english_texts, detect_english_texts


def test_language_detection():
    """Test that language detection works correctly."""
    # Create a test dataframe with mixed languages
    df = pd.DataFrame({
        'prompt': [
            "Hello, how are you?",  # English
            "Bonjour, comment allez-vous?",  # French
            "¿Cómo estás?",  # Spanish
            "This is a test sentence in English.",  # English
            "Guten Tag, wie geht es Ihnen?",  # German
            "What is the meaning of life?",  # English
            "こんにちは",  # Japanese
            "Write me a poem about nature.",  # English
        ],
        'label': [0, 1, 0, 1, 0, 1, 0, 1]
    })
    
    print("Testing language detection...")
    print(f"Original dataset size: {len(df)}")
    
    # Test detection
    is_english = detect_english_texts(df, text_column='prompt', use_cache=False)
    print(f"\nDetected {is_english.sum()} English texts out of {len(df)}")
    
    # Test filtering
    filtered_df, _ = filter_english_texts(df, text_column='prompt', use_cache=False)
    print(f"\nFiltered dataset size: {len(filtered_df)}")
    
    # Verify English texts
    print("\nEnglish texts found:")
    for idx, row in filtered_df.iterrows():
        print(f"  - {row['prompt']}")
    
    # Expected: should keep approximately 4 English texts
    assert len(filtered_df) >= 3, f"Expected at least 3 English texts, got {len(filtered_df)}"
    assert len(filtered_df) <= len(df), "Filtered dataset should not be larger than original"
    
    print("\n✓ Language detection test passed!")


def test_caching():
    """Test that caching works correctly."""
    df = pd.DataFrame({
        'prompt': [
            "This is a test.",
            "Another test sentence.",
            "One more for good measure."
        ]
    })
    
    print("\nTesting caching functionality...")
    
    # First call - should compute and cache
    print("First call (should compute):")
    is_english_1 = detect_english_texts(df, text_column='prompt', use_cache=True, cache_dir='data/cache/test')
    
    # Second call - should load from cache
    print("\nSecond call (should load from cache):")
    is_english_2 = detect_english_texts(df, text_column='prompt', use_cache=True, cache_dir='data/cache/test')
    
    # Results should be identical
    assert (is_english_1 == is_english_2).all(), "Cached results don't match"
    
    print("\n✓ Caching test passed!")


if __name__ == "__main__":
    test_language_detection()
    test_caching()
    print("\n✓ All language filtering tests passed!")
