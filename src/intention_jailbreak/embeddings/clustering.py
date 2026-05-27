"""HDBSCAN clustering on sentence embeddings."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np

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
