"""HDBSCAN clustering on sentence embeddings."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import hdbscan as _hdbscan_t  # noqa: F401


def hdbscan_cluster(
    embeddings: np.ndarray,
    *,
    min_cluster_size: int = 30,
    min_samples: int | None = None,
    metric: str = "cosine",
    cluster_selection_method: str = "eom",
    verbose: bool = True,
) -> tuple[np.ndarray, "_hdbscan_t.HDBSCAN"]:
    """Cluster `embeddings` with HDBSCAN.

    Args:
        embeddings: `(n, d)` array.
        min_cluster_size: smallest cluster the algorithm will keep. Bigger →
            fewer, broader clusters. ~1% of the dataset is a sane starting point.
        min_samples: density floor (lower → more aggressive cluster splitting,
            fewer points labelled noise). Defaults to `min_cluster_size`.
        metric: `"cosine"` is the right choice for sentence embeddings.
            HDBSCAN doesn't accept `cosine` directly, so the function L2-normalises
            the input and runs HDBSCAN with euclidean — this produces the same
            cluster structure as cosine on the unit sphere.
        cluster_selection_method: HDBSCAN's `eom` (default) or `leaf`.
            `leaf` returns finer-grained clusters.

    Returns `(labels, fitted_model)`. `-1` in `labels` marks noise points.
    """
    import hdbscan  # lazy

    if metric == "cosine":
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        X = embeddings / norms
        hdbscan_metric = "euclidean"
    else:
        X = embeddings
        hdbscan_metric = metric

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=hdbscan_metric,
        cluster_selection_method=cluster_selection_method,
    )
    if verbose:
        print(
            f"Fitting HDBSCAN on {X.shape[0]} × {X.shape[1]} "
            f"(min_cluster_size={min_cluster_size}, metric={metric})..."
        )
    t0 = time.perf_counter()
    labels = clusterer.fit_predict(X.astype(np.float32))
    if verbose:
        print(f"HDBSCAN fit complete in {time.perf_counter() - t0:.1f}s")
    return labels, clusterer


def export_clusters_jsonl(
    df: pd.DataFrame,
    output_path: Path | str,
    *,
    columns: Sequence[str] = (
        "cluster",
        "intent",
        "n_prompts",
        "prompts",
        "dataset",
        "dataset_breakdown",
        "predicted_harm",
        "true_harm_binary",
        "umap1",
        "umap2",
    ),
    drop_noise: bool = False,
) -> Path:
    """Write `df` to JSONL with one record per row, sorted by cluster id.

    JSONL is the right format for downstream LLM annotation: each line is a
    self-contained record, list-valued columns (e.g. `prompts`) survive
    unflattened, and the file is trivial to iterate per-cluster (`groupby` on
    the load side, or `jq 'select(.cluster == N)'`).

    Args:
        df: DataFrame containing the columns in `columns`.
        output_path: destination `.jsonl` file.
        columns: ordered list of columns to include in each record. Missing
            columns are silently skipped (so callers can pass a superset).
        drop_noise: when True, omit rows where `cluster == -1`.

    Returns the resolved `Path` written.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    present_cols = [c for c in columns if c in df.columns]
    work = df[present_cols].copy()
    if drop_noise and "cluster" in work.columns:
        work = work[work["cluster"] != -1]

    sort_keys = [c for c in ("cluster", "n_prompts") if c in work.columns]
    if sort_keys:
        ascending = [True, False][: len(sort_keys)]
        work = work.sort_values(sort_keys, ascending=ascending)

    n_written = 0
    with output_path.open("w", encoding="utf-8") as f:
        for _, row in work.iterrows():
            record = {}
            for col in present_cols:
                value = row[col]
                # numpy scalars → native Python so json doesn't choke
                if isinstance(value, np.generic):
                    value = value.item()
                elif isinstance(value, np.ndarray):
                    value = value.tolist()
                record[col] = value
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_written += 1

    print(f"Wrote {n_written} rows → {output_path}")
    return output_path


def summarise_clusters(labels: np.ndarray) -> dict:
    """Return a small summary dict: cluster counts, noise share, top sizes."""
    unique, counts = np.unique(labels, return_counts=True)
    n_total = int(labels.size)
    n_noise = int(counts[unique == -1].sum()) if -1 in unique else 0
    cluster_pairs = sorted(
        ((int(u), int(c)) for u, c in zip(unique, counts) if u != -1),
        key=lambda x: -x[1],
    )
    return {
        "n_points": n_total,
        "n_clusters": len(cluster_pairs),
        "n_noise": n_noise,
        "noise_share": n_noise / n_total if n_total else 0.0,
        "top_cluster_sizes": cluster_pairs[:10],
    }
