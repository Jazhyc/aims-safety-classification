"""Load intent generations from prediction `.jsonl` files into a DataFrame."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd

# Test eval datasets used by `eval_safety_classifier.py`. `annotated-intents-val`
# is the validation split and is excluded here.
DEFAULT_TEST_DATASETS: tuple[str, ...] = (
    "aegis",
    "annotated-intents",
    "openai-moderation",
    "toxic-chat",
    "wildguardmix",
    "xstest",
)


def load_intent_predictions(
    predictions_dir: Path | str,
    *,
    datasets: Iterable[str] = DEFAULT_TEST_DATASETS,
    prediction_filename: str | None = None,
    intent_field: str = "generated_intent",
) -> pd.DataFrame:
    """Load intent generations from `predictions_dir/<dataset>/*.jsonl`.

    Args:
        predictions_dir: Root directory containing one subdir per dataset.
        datasets: Dataset subdirectory names to load.
        prediction_filename: Optional exact filename (e.g. `meta-llama_..._generation.jsonl`).
            If None, the single `*.jsonl` in each dataset dir is used; if multiple
            exist, all are concatenated.
        intent_field: JSON field holding the generated intent.

    Returns DataFrame with columns
        `dataset, id, prompt, intent, true_intent, predicted_harm, true_harm_binary`.
    Rows with empty / missing `intent_field` are dropped.
    """
    predictions_dir = Path(predictions_dir)
    rows: list[dict] = []

    for dataset in datasets:
        dataset_dir = predictions_dir / dataset
        if not dataset_dir.exists():
            raise FileNotFoundError(f"Dataset dir not found: {dataset_dir}")

        if prediction_filename is not None:
            jsonl_paths = [dataset_dir / prediction_filename]
        else:
            jsonl_paths = sorted(dataset_dir.glob("*.jsonl"))
            if not jsonl_paths:
                raise FileNotFoundError(f"No .jsonl files in {dataset_dir}")

        for jsonl_path in jsonl_paths:
            with jsonl_path.open() as f:
                for line in f:
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    intent = rec.get(intent_field)
                    if not intent:
                        continue
                    rows.append(
                        {
                            "dataset": dataset,
                            "id": rec.get("id"),
                            "prompt": rec.get("prompt"),
                            "intent": intent,
                            "true_intent": rec.get("true_intent"),
                            "predicted_harm": rec.get("predicted_harm"),
                            "true_harm_binary": rec.get("true_harm_binary"),
                        }
                    )

    return pd.DataFrame(rows)
