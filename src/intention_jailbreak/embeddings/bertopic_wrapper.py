"""BERTopic wrapper with embedding caching for topic modeling."""

import hashlib
import pickle
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer


class BERTopicModelWrapper:
    """
    Wrapper around BERTopic for topic modeling with embedding caching.
    
    This class uses BERTopic's full pipeline (embedding → UMAP → HDBSCAN → c-TF-IDF)
    while caching the expensive embedding step to avoid recomputation.
    """
    
    def __init__(
        self,
        cache_dir: str | Path,
        embedding_model: Optional[str | SentenceTransformer] = None,
        **bertopic_kwargs
    ):
        """Initialize BERTopic wrapper.
        
        Args:
            cache_dir: Directory to store cached embeddings
            embedding_model: Model name or SentenceTransformer instance (default: "all-MiniLM-L6-v2")
            **bertopic_kwargs: Additional arguments for BERTopic initialization
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up embedding model
        if embedding_model is None:
            self.embedding_model = "all-MiniLM-L6-v2"
        else:
            self.embedding_model = embedding_model
            
        # Store BERTopic kwargs for model initialization
        self.bertopic_kwargs = bertopic_kwargs
        self.topic_model = None
        self._embeddings = None
    
    def _get_cache_path(self, texts: List[str], prefix: str = "embeddings") -> Path:
        """Generate cache filename based on text content hash."""
        text_content = "||".join(texts)
        content_hash = hashlib.md5(text_content.encode()).hexdigest()
        return self.cache_dir / f"{prefix}_{content_hash}.pkl"
    
    def _save_embeddings_cache(self, cache_path: Path, embeddings: np.ndarray):
        """Save embeddings to cache."""
        with open(cache_path, "wb") as f:
            pickle.dump(embeddings, f)
        print(f"💾 Cached embeddings to {cache_path.name}")
    
    def _load_embeddings_cache(self, cache_path: Path) -> np.ndarray:
        """Load embeddings from cache."""
        with open(cache_path, "rb") as f:
            embeddings = pickle.load(f)
        print(f"📂 Loaded embeddings from cache: {cache_path.name}")
        return embeddings
    
    def _get_or_create_embeddings(
        self,
        texts: List[str],
        use_cache: bool = True,
        cache_prefix: str = "embeddings"
    ) -> np.ndarray:
        """Get embeddings from cache or create new ones."""
        cache_path = self._get_cache_path(texts, cache_prefix)
        
        # Try to load from cache
        if use_cache and cache_path.exists():
            return self._load_embeddings_cache(cache_path)
        
        # Generate new embeddings
        print("🔄 Generating embeddings with sentence-transformers...")
        if isinstance(self.embedding_model, str):
            model = SentenceTransformer(self.embedding_model)
        else:
            model = self.embedding_model
            
        embeddings = model.encode(texts, show_progress_bar=True)
        
        # Save to cache
        if use_cache:
            self._save_embeddings_cache(cache_path, embeddings)
        
        return embeddings
    
    def fit_transform(
        self,
        texts: List[str],
        use_cache: bool = True,
        cache_prefix: str = "embeddings"
    ):
        """
        Fit BERTopic model and generate topics with cached embeddings.
        
        Args:
            texts: List of text documents to model
            use_cache: Whether to use cached embeddings if available
            cache_prefix: Prefix for cache filename
            
        Returns:
            topics: Topic assignments for each document
            probs: Topic probabilities for each document
        """
        # Get or create embeddings (with caching)
        self._embeddings = self._get_or_create_embeddings(texts, use_cache, cache_prefix)
        
        # Initialize BERTopic model
        print("🔄 Fitting BERTopic model...")
        self.topic_model = BERTopic(
            embedding_model=self.embedding_model,
            **self.bertopic_kwargs
        )
        
        # Fit model using pre-computed embeddings
        topics, probs = self.topic_model.fit_transform(texts, self._embeddings)
        
        print(f"✓ Topic modeling complete!")
        print(f"  Found {len(set(topics)) - (1 if -1 in topics else 0)} topics")
        
        return topics, probs
    
    def get_topic_info(self) -> pd.DataFrame:
        """Get information about discovered topics."""
        if self.topic_model is None:
            raise ValueError("Model not fitted yet. Call fit_transform first.")
        return self.topic_model.get_topic_info()
    
    def get_embeddings(self) -> np.ndarray:
        """Get the cached/generated embeddings."""
        if self._embeddings is None:
            raise ValueError("Embeddings not generated yet. Call fit_transform first.")
        return self._embeddings
    
    def get_topic(self, topic_id: int):
        """Get words and weights for a specific topic."""
        if self.topic_model is None:
            raise ValueError("Model not fitted yet. Call fit_transform first.")
        return self.topic_model.get_topic(topic_id)
    
    def visualize_topics(self):
        """Visualize topics in 2D space."""
        if self.topic_model is None:
            raise ValueError("Model not fitted yet. Call fit_transform first.")
        return self.topic_model.visualize_topics()
    
    def visualize_documents(self, docs: List[str], embeddings: Optional[np.ndarray] = None):
        """Visualize documents in 2D space colored by topic."""
        if self.topic_model is None:
            raise ValueError("Model not fitted yet. Call fit_transform first.")
        if embeddings is None:
            embeddings = self._embeddings
        return self.topic_model.visualize_documents(docs, embeddings=embeddings)


def load_intents_from_parquet(parquet_path: str | Path) -> pd.DataFrame:
    """Load intent data from parquet file.
    
    Args:
        parquet_path: Path to parquet file
        
    Returns:
        DataFrame with intent data
    """
    df = pd.read_parquet(parquet_path)
    
    # Validate required columns
    if "Intent" not in df.columns:
        raise ValueError("Parquet file must contain 'Intent' column")
    
    # Filter out empty intents
    df = df[df["Intent"].notna() & (df["Intent"] != "")]
    
    return df
