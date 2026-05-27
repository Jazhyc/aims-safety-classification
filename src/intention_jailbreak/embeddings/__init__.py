"""Embedding generation and caching for intent analysis."""

from .intent_loading import DEFAULT_TEST_DATASETS, load_intent_predictions
from .sentence_embeddings import DEFAULT_MODEL, embed_texts, pca_project, umap_project

# Lazily expose BERTopic helpers: importing them eagerly pulls in bertopic →
# umap, hdbscan, plotly, transformers — ~19s of cold-start. Resolve on access
# so consumers that only need embed_texts/pca/umap don't pay that cost.
_LAZY_NAMES = {
    "BERTopicModelWrapper": ".bertopic_wrapper",
    "load_intents_from_parquet": ".bertopic_wrapper",
    "interactive_umap_plot_toggleable": ".plotting",
}


def __getattr__(name: str):
    if name in _LAZY_NAMES:
        from importlib import import_module

        module = import_module(_LAZY_NAMES[name], __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BERTopicModelWrapper",
    "DEFAULT_MODEL",
    "DEFAULT_TEST_DATASETS",
    "embed_texts",
    "interactive_umap_plot_toggleable",
    "load_intent_predictions",
    "load_intents_from_parquet",
    "pca_project",
    "umap_project",
]
