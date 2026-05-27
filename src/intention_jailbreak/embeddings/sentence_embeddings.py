"""Sentence-transformer embedding + PCA helpers with disk caching."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from tqdm.auto import tqdm

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _content_hash(texts: Sequence[str], model_name: str) -> str:
    h = hashlib.sha256()
    h.update(model_name.encode("utf-8"))
    h.update(b"\x00")
    for t in texts:
        h.update(t.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


def embed_texts(
    texts: Sequence[str],
    cache_dir: Path | str,
    *,
    model_name: str = DEFAULT_MODEL,
    cache_key: str = "embeddings",
    batch_size: int = 64,
    use_cache: bool = True,
) -> np.ndarray:
    """Encode `texts` with a sentence-transformer, caching the result on disk.

    The cache filename embeds a hash of `(model_name, texts)` so a stale cache
    under the same `cache_key` is never silently returned.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    content_hash = _content_hash(texts, model_name)
    cache_path = cache_dir / f"{cache_key}_{content_hash}.npy"

    if use_cache and cache_path.exists():
        return np.load(cache_path)

    model = SentenceTransformer(model_name)
    embeddings: list[np.ndarray] = []
    for start in tqdm(
        range(0, len(texts), batch_size),
        desc=f"Embedding {cache_key}",
        unit="batch",
    ):
        batch = list(texts[start : start + batch_size])
        emb = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        embeddings.append(emb)

    stacked = np.vstack(embeddings)
    np.save(cache_path, stacked)
    return stacked


def pca_project(
    embeddings: np.ndarray,
    n_components: int = 2,
    *,
    random_state: int = 42,
) -> tuple[np.ndarray, PCA]:
    """Fit PCA on `embeddings` and return `(projection, fitted_pca)`."""
    pca = PCA(n_components=n_components, random_state=random_state)
    projection = pca.fit_transform(embeddings)
    return projection, pca


def umap_project(
    embeddings: np.ndarray,
    n_components: int = 2,
    *,
    n_neighbors: int = 30,
    min_dist: float = 0.1,
    metric: str = "cosine",
    random_state: int = 42,
):
    """Fit UMAP on `embeddings` and return `(projection, fitted_reducer)`.

    `cosine` metric is the right default for sentence-transformer embeddings
    (they're trained with cosine similarity). `n_neighbors=30` favours global
    structure over very tight local clusters; lower it to ~15 for the opposite.
    """
    import umap  # lazy: heavy import, optional dep

    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )
    projection = reducer.fit_transform(embeddings)
    return projection, reducer
