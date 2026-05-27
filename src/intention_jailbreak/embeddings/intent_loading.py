"""Load intent generations from prediction `.jsonl` files into a DataFrame."""

from __future__ import annotations

import json
from collections import Counter
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


def dedupe_by_intent(
    df: pd.DataFrame,
    *,
    intent_col: str = "intent",
) -> pd.DataFrame:
    """Collapse to one row per unique intent string.

    Use this when the analysis is about intents themselves (clustering, topic
    modelling, embedding visualisation) rather than per-prompt behaviour —
    encoding the same intent string 144 times is wasted work and gives UMAP /
    HDBSCAN a misleading view of density.

    Aggregations per group:
      - `n_prompts` (int): number of original rows with this intent.
      - `prompts` (list[str]): unique prompts that produced this intent.
      - `dataset`, `predicted_harm`, `true_harm_binary`: mode (most common value).
      - `dataset_breakdown` (str): hover-friendly tally, e.g.
        `"toxic-chat: 143, aegis: 1"`. Surfaces cross-dataset intents at a glance.
      - `true_intent`: first non-null.
      - `prompt`: hover-ready summary — for solo intents it's just the prompt,
        otherwise `"[N prompts, K unique] first_prompt"`. Keeps the existing
        plot helper (which expects a string) working unchanged.
    """

    def _mode_or_first(series: pd.Series):
        modes = series.mode(dropna=True)
        return modes.iloc[0] if not modes.empty else None

    def _breakdown(series: pd.Series) -> str:
        counts = Counter(v for v in series if v is not None and not pd.isna(v))
        return ", ".join(f"{k}: {v}" for k, v in counts.most_common())

    def _first_non_null(series: pd.Series):
        for v in series:
            if v is None:
                continue
            if isinstance(v, float) and pd.isna(v):
                continue
            return v
        return None

    grouped = (
        df.groupby(intent_col, sort=False, dropna=False)
        .agg(
            n_prompts=("prompt", "size"),
            prompts=("prompt", lambda s: list(dict.fromkeys(v for v in s if v))),
            dataset=("dataset", _mode_or_first),
            dataset_breakdown=("dataset", _breakdown),
            predicted_harm=("predicted_harm", _mode_or_first),
            true_harm_binary=("true_harm_binary", _mode_or_first),
            true_intent=("true_intent", _first_non_null),
        )
        .reset_index()
    )

    def _prompt_summary(row) -> str:
        prompts = row["prompts"]
        if not prompts:
            return ""
        if row["n_prompts"] == 1:
            return prompts[0]
        return f"[{row['n_prompts']} prompts, {len(prompts)} unique] {prompts[0]}"

    grouped["prompt"] = grouped.apply(_prompt_summary, axis=1)
    return grouped
