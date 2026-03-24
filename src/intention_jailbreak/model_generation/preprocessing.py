from datasets import Dataset, load_dataset


def load_raw_parquet(split: str = "train") -> Dataset:
    print(f"Loading dataset from Hugging Face Hub: Jazhyc/wildguard-annotated-intents (split={split})")
    ds = load_dataset("Jazhyc/wildguard-annotated-intents", split=split)
    print("Raw dataset size:", len(ds))
    print("Columns:", ds.column_names)
    return ds


def basic_cleaning(ds: Dataset) -> Dataset:
    rename_map = {
        "ID": "id",
        "Prompt": "prompt",
        "Intent": "intent",
    }

    existing_renames = {k: v for k, v in rename_map.items() if k in ds.column_names}
    ds = ds.rename_columns(existing_renames)

    keep_cols = [
        "id",
        "prompt",
        "intent",
        "Wildguard ID",
        "Duplicate ID",
        "Annotator Harm",
        "Dataset Harm",
        "Adversarial",
    ]
    existing_cols = [c for c in keep_cols if c in ds.column_names]
    ds = ds.select_columns(existing_cols)

    # Filter out rows with empty / missing prompt or intent
    def is_valid(example):
        p = example["prompt"]
        i = example["intent"]
        return (
            p is not None and isinstance(p, str) and p.strip() != "" and
            i is not None and isinstance(i, str) and i.strip() != ""
        )

    ds = ds.filter(is_valid)
    print("After basic cleaning, dataset size:", len(ds))
    return ds


def preprocess_data(split: str = "train") -> Dataset:
    """
    Load and clean the annotated intents dataset.

    Args:
        split: HuggingFace split to load — "train", "validation", or "test".
               Use the canonical HF splits rather than re-splitting locally.
    """
    ds = load_raw_parquet(split)
    ds = basic_cleaning(ds)
    return ds
