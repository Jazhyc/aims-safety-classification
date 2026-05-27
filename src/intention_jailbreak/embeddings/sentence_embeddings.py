"""Sentence-transformer embedding + PCA helpers with disk caching."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from tqdm.auto import tqdm

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _stable_json(obj: Any) -> str:
    """JSON dump that's deterministic across dict orderings and rejects unhashable junk."""

    def fallback(x: Any):
        # torch.dtype, torch.device etc → repr is stable per-process
        return repr(x)

    return json.dumps(obj, sort_keys=True, default=fallback)


def _content_hash(
    texts: Sequence[str],
    model_name: str,
    model_init_kwargs: dict | None,
    encode_kwargs: dict | None,
    deduplicate: bool,
) -> str:
    h = hashlib.sha256()
    h.update(model_name.encode("utf-8"))
    h.update(b"\x00")
    h.update(_stable_json(model_init_kwargs or {}).encode("utf-8"))
    h.update(b"\x00")
    h.update(_stable_json(encode_kwargs or {}).encode("utf-8"))
    h.update(b"\x00")
    h.update(b"dedup" if deduplicate else b"raw")
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
    model_init_kwargs: dict | None = None,
    encode_kwargs: dict | None = None,
    deduplicate: bool = True,
) -> np.ndarray:
    """Encode `texts` with a sentence-transformer, caching the result on disk.

    Args:
        texts: input strings to embed.
        cache_dir: directory for cached `.npy` files.
        model_name: HF model id (default: `sentence-transformers/all-MiniLM-L6-v2`).
        cache_key: human-readable prefix for the cache file.
        batch_size: rows per `encode()` call.
        use_cache: skip the encode and return the cached array if it exists.
        model_init_kwargs: passed to `SentenceTransformer(...)` (e.g.
            `{"trust_remote_code": True, "model_kwargs": {"dtype": torch.bfloat16}}`).
        encode_kwargs: passed to `model.encode(...)` (e.g. `{"task": "clustering"}`
            for Jina v5; `{"prompt_name": "query"}` for asymmetric retrieval).
        deduplicate: encode each unique string only once and broadcast the result
            back to the original row order. Faster when many duplicates exist and
            sidesteps batched-matmul nondeterminism in low-precision dtypes
            (bf16/fp16) where the same string at different batch positions can
            otherwise produce microscopically different vectors. Output shape is
            unchanged: `(len(texts), embedding_dim)`.

    The cache filename embeds a hash of (model_name, model_init_kwargs,
    encode_kwargs, deduplicate, texts), so changing any of them busts the cache.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    content_hash = _content_hash(
        texts, model_name, model_init_kwargs, encode_kwargs, deduplicate
    )
    cache_path = cache_dir / f"{cache_key}_{content_hash}.npy"

    if use_cache and cache_path.exists():
        return np.load(cache_path)

    model = SentenceTransformer(model_name, **(model_init_kwargs or {}))
    extra_encode = dict(encode_kwargs or {})
    # These two are non-negotiable for our cache path; warn rather than silently drop.
    for reserved in ("convert_to_numpy", "show_progress_bar"):
        if reserved in extra_encode:
            raise ValueError(
                f"encode_kwargs may not set {reserved!r} — managed by embed_texts."
            )

    if deduplicate:
        unique_texts, inverse = np.unique(
            np.asarray(texts, dtype=object), return_inverse=True
        )
        encode_targets: Sequence[str] = list(unique_texts)
        print(
            f"Dedup: encoding {len(encode_targets)} unique texts "
            f"(of {len(texts)} total)"
        )
    else:
        encode_targets = texts
        inverse = None

    embeddings: list[np.ndarray] = []
    for start in tqdm(
        range(0, len(encode_targets), batch_size),
        desc=f"Embedding {cache_key}",
        unit="batch",
    ):
        batch = list(encode_targets[start : start + batch_size])
        emb = model.encode(
            batch,
            convert_to_numpy=True,
            show_progress_bar=False,
            **extra_encode,
        )
        embeddings.append(emb)

    encoded = np.vstack(embeddings)
    stacked = encoded[inverse] if inverse is not None else encoded
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
    deduplicate: bool = True,
):
    """Fit UMAP on `embeddings` and return `(projection, fitted_reducer)`.

    `cosine` metric is the right default for sentence-transformer embeddings
    (they're trained with cosine similarity). `n_neighbors=30` favours global
    structure over very tight local clusters; lower it to ~15 for the opposite.

    `deduplicate=True` collapses byte-identical input rows before fitting and
    broadcasts the projection back. Without this, UMAP can place duplicate
    inputs at different macro positions because each duplicate is an independent
    node in the k-NN graph and SGD may pull them into different basins early in
    optimization — `random_state` does not prevent this. Set to False if you
    want density-aware layouts where duplicates count toward local mass.
    """
    import umap  # lazy: heavy import, optional dep

    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )

    if deduplicate:
        unique_emb, inverse = np.unique(embeddings, axis=0, return_inverse=True)
        if len(unique_emb) < len(embeddings):
            print(
                f"UMAP dedup: fitting on {len(unique_emb)} unique rows "
                f"(of {len(embeddings)} total)"
            )
        unique_proj = reducer.fit_transform(unique_emb)
        projection = unique_proj[inverse]
    else:
        projection = reducer.fit_transform(embeddings)

    return projection, reducer
