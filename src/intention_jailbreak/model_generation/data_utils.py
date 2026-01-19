from typing import Tuple
import matplotlib.pyplot as plt
from collections import Counter


def align_tokenizer_with_model(tokenizer, model):
    """
    Align tokenizer special tokens with model config.
    
    Transformers sometimes doesn't handle this automatically when loading
    models and tokenizers separately. This ensures consistency and prevents
    warnings about mismatched special tokens.
    
    Args:
        tokenizer: The tokenizer instance
        model: The model instance with config
    """
    # Align pad token - model config takes precedence
    if model.config.pad_token_id is not None:
        tokenizer.pad_token_id = model.config.pad_token_id
    elif tokenizer.pad_token is None:
        # Fallback: use eos_token as pad_token
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id
    else:
        # Tokenizer has pad_token but model doesn't - update model
        model.config.pad_token_id = tokenizer.pad_token_id
    
    # Align BOS token - model config takes precedence
    # Handle both single token ID and list of IDs
    if model.config.bos_token_id is not None:
        bos_id = model.config.bos_token_id
        if isinstance(bos_id, list):
            bos_id = bos_id[0]
        tokenizer.bos_token_id = bos_id
    elif tokenizer.bos_token_id is not None:
        model.config.bos_token_id = tokenizer.bos_token_id
    
    # Align EOS token - model config takes precedence
    # Handle both single token ID and list of IDs (e.g., Gemma-3)
    if model.config.eos_token_id is not None:
        # If model has multiple EOS tokens, use the first one
        eos_id = model.config.eos_token_id
        if isinstance(eos_id, list):
            eos_id = eos_id[0]
        tokenizer.eos_token_id = eos_id
    elif tokenizer.eos_token_id is not None:
        model.config.eos_token_id = tokenizer.eos_token_id


def get_lengths(dataset, plot: bool = False) -> Tuple[int, int]:
    prompt_lengths = [len(item["prompt"].split()) for item in dataset]
    intent_lengths = [len(item["intent"].split()) for item in dataset]
    prompt_max = max(prompt_lengths)
    intent_max = max(intent_lengths)
    print("Maximum prompt length:", prompt_max)
    print("Maximum intent length:", intent_max)

    if plot:
        plt.hist(prompt_lengths, bins=30, alpha=0.5, label="Prompt Lengths")
        plt.hist(intent_lengths, bins=30, alpha=0.5, label="Intent Lengths")
        plt.legend(loc="upper right")
        plt.show()

    return prompt_max, intent_max


def map_harm_to_binary(harm_value):
    """
    Map 4-category harm labels to binary (Safe/Harmful).
    
    Args:
        harm_value: One of 'completely harmful', 'uncertain harmful', 'uncertain safe', 'completely safe'
    
    Returns:
        'harmful' or 'safe'
    """
    if harm_value is None:
        return None
    
    harm_lower = str(harm_value).lower().strip()
    
    if 'harmful' in harm_lower:
        return 'harmful'
    elif 'safe' in harm_lower:
        return 'safe'
    else:
        return None


def apply_binary_harm_mapping(dataset, binary_harm_mapping=True):
    """
    Apply binary harm mapping to dataset if enabled.
    
    Args:
        dataset: Dataset with 'Annotator Harm' column
        binary_harm_mapping: Whether to apply binary mapping (default: True)
    
    Returns:
        Dataset with mapped harm values (or unchanged if not enabled)
    """
    if not binary_harm_mapping or "Annotator Harm" not in dataset.column_names:
        return dataset
    
    print("\nApplying binary harm mapping (4 categories -> 2 categories)")
    
    # Print distribution before mapping
    harm_values_before = [ex for ex in dataset["Annotator Harm"] if ex is not None]
    dist_before = Counter(harm_values_before)
    print("Distribution before mapping:")
    for label, count in sorted(dist_before.items()):
        print(f"  {label}: {count}")
    
    # Apply mapping
    def apply_binary_harm(example):
        if "Annotator Harm" in example:
            example["Annotator Harm"] = map_harm_to_binary(example["Annotator Harm"])
        return example
    
    dataset = dataset.map(apply_binary_harm)
    
    # Print distribution after mapping
    harm_values_after = [ex for ex in dataset["Annotator Harm"] if ex is not None]
    dist_after = Counter(harm_values_after)
    print("Distribution after mapping:")
    for label, count in sorted(dist_after.items()):
        print(f"  {label}: {count}")
    print()
    
    return dataset


def train_val_test_split(dataset, config):
    data_cfg = config.get("data", {})
    test_size = data_cfg.get("test_size", 0.1)
    val_size = data_cfg.get("val_size", 0.1)
    seed = data_cfg.get("seed", 22)

    train_test_split = dataset.train_test_split(test_size=test_size, seed=seed)
    test_dataset = train_test_split["test"]
    true_val_size = val_size / (1 - test_size)
    val_dataset_split = train_test_split["train"].train_test_split(
        test_size=true_val_size, seed=seed
    )
    train_dataset = val_dataset_split["train"]
    val_dataset = val_dataset_split["test"]
    return train_dataset, val_dataset, test_dataset
