from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
from pathlib import Path
import json
import logging
import plotly.express as px
import plotly.graph_objects as go
from umap import UMAP
from intention_jailbreak.embeddings import BERTopicModelWrapper
from hdbscan import HDBSCAN

logger = logging.getLogger(__name__)

class ClusterComparator:
    """Compare clustering patterns between human and synthetic intents with visualization."""
    
    def __init__(
        self,
        cache_dir: str = "../data/embeddings",
        embedding_model: str = "Qwen/Qwen3-Embedding-0.6B",
        umap_n_neighbors: int = 15,
        umap_n_components: int = 5,
        umap_min_dist: float = 0.0,
        umap_metric: str = 'cosine',
        hdbscan_min_cluster_size: int = 15,
        hdbscan_min_samples: int = 5,
        hdbscan_metric: str = 'euclidean',
        hdbscan_cluster_selection_method: str = 'eom',
        hdbscan_prediction_data: bool = True,
        nr_topics: str = "auto",
        verbose: bool = True
    ):
        """Initialize comparator with BERTopic setup."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        umap_model = UMAP(
            n_neighbors=umap_n_neighbors,
            n_components=umap_n_components,
            min_dist=umap_min_dist,
            metric=umap_metric,
            random_state=42
        )

        hdbscan_model = HDBSCAN(
            min_cluster_size=hdbscan_min_cluster_size,
            min_samples=hdbscan_min_samples,
            metric=hdbscan_metric,
            cluster_selection_method=hdbscan_cluster_selection_method,
            prediction_data=hdbscan_prediction_data
        )

        self.topic_wrapper = BERTopicModelWrapper(
            cache_dir=str(self.cache_dir),
            embedding_model=embedding_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            nr_topics=nr_topics,
            verbose=verbose
        )

        # Store embeddings and topic info for visualization
        self._current_embeddings = None
        self._current_topic_info = None

    def cluster_intents(
        self, 
        intents: List[str], 
        cache_prefix: str
    ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Cluster intents using BERTopic.
        
        Returns:
            (topics, probabilities, topic_info)
        """
        logger.info(f"Clustering {len(intents)} intents with prefix: {cache_prefix}")
        
        topics, probs = self.topic_wrapper.fit_transform(
            intents,
            use_cache=True,
            cache_prefix=cache_prefix
        )
        
        topic_info = self.topic_wrapper.get_topic_info()
        
        # Store for visualization
        self._current_embeddings = self.topic_wrapper.get_embeddings()
        self._current_topic_info = topic_info
        
        # Log cluster statistics
        topics_array = np.array(topics)
        n_outliers = (topics_array == -1).sum()
        n_clusters = len(set(topics)) - (1 if -1 in topics else 0)
        
        logger.info(f"\nTopic Distribution:")
        logger.info(f"  Outliers (Topic -1): {n_outliers}")
        logger.info(f"  Clustered topics: {n_clusters}")
        if n_clusters > 0:
            avg_intents = len(topics_array[topics_array != -1]) / n_clusters
            logger.info(f"  Average intents per topic: {avg_intents:.1f}")
        
        logger.info(f"Topic info: {topic_info}")
        
        return topics, probs, topic_info
    
    def compare_topic_distributions(
        self,
        human_topics: np.ndarray,
        synthetic_topics: np.ndarray
    ) -> Dict[str, float]:
        """
        Compare topic distributions using various metrics.
        
        Returns:
            Dictionary of comparison metrics
        """
        logger.info("Comparing topic distributions")
        
        # Get topic counts (excluding outliers)
        human_counts = pd.Series(human_topics[human_topics != -1]).value_counts()
        synthetic_counts = pd.Series(synthetic_topics[synthetic_topics != -1]).value_counts()
        
        # Number of clusters
        n_human_clusters = len(human_counts)
        n_synthetic_clusters = len(synthetic_counts)
        
        # Compute diversity metrics
        human_entropy = -np.sum(
            (human_counts / human_counts.sum()) * 
            np.log(human_counts / human_counts.sum() + 1e-10)
        )
        synthetic_entropy = -np.sum(
            (synthetic_counts / synthetic_counts.sum()) * 
            np.log(synthetic_counts / synthetic_counts.sum() + 1e-10)
        )
        
        # Outlier rates
        human_outlier_rate = (human_topics == -1).sum() / len(human_topics)
        synthetic_outlier_rate = (synthetic_topics == -1).sum() / len(synthetic_topics)
        
        metrics = {
            'human_clusters': n_human_clusters,
            'synthetic_clusters': n_synthetic_clusters,
            'cluster_ratio': n_synthetic_clusters / n_human_clusters if n_human_clusters > 0 else float('inf'),
            'human_entropy': human_entropy,
            'synthetic_entropy': synthetic_entropy,
            'entropy_ratio': synthetic_entropy / human_entropy if human_entropy > 0 else float('inf'),
            'human_outlier_rate': human_outlier_rate,
            'synthetic_outlier_rate': synthetic_outlier_rate,
            'outlier_ratio': synthetic_outlier_rate / human_outlier_rate if human_outlier_rate > 0 else float('inf')
        }
        
        logger.info(f"Comparison metrics: {json.dumps(metrics, indent=2)}")
        return metrics
    
    def visualize_comparison_with_embeddings(
        self, 
        human_intents: List[str], 
        synthetic_intents: List[str],
        human_topics: np.ndarray, 
        synthetic_topics: np.ndarray,
        human_embeddings: np.ndarray,
        synthetic_embeddings: np.ndarray,
        output_dir: str
    ) -> None:
        """
        Create comprehensive visualizations comparing human and synthetic intents
        using pre-computed embeddings and topics.

        Args:
            human_intents: List of human-generated intents
            synthetic_intents: List of synthetic intents  
            human_topics: Topic assignments for human intents
            synthetic_topics: Topic assignments for synthetic intents
            human_embeddings: Pre-computed embeddings for human intents
            synthetic_embeddings: Pre-computed embeddings for synthetic intents
            output_dir: Directory to save visualizations
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Create individual cluster visualizations
        logger.info("Creating human intents visualization")
        human_fig = self._create_cluster_visualization(
            human_intents, human_topics, human_embeddings, "Human-Generated Intents"
        )
        human_fig.write_html(output_path / "human_intents_clusters.html")

        logger.info("Creating synthetic intents visualization")
        synthetic_fig = self._create_cluster_visualization(
            synthetic_intents, synthetic_topics, synthetic_embeddings, "Synthetic Intents"
        )
        synthetic_fig.write_html(output_path / "synthetic_intents_clusters.html")

        # Create comparison visualization
        logger.info("Creating comparison visualization")
        comparison_fig = self._create_comparison_visualization(
            human_intents, synthetic_intents, 
            human_topics, synthetic_topics,
            human_embeddings, synthetic_embeddings
        )
        comparison_fig.write_html(output_path / "intents_comparison.html")

        # Create topic distribution comparison
        dist_fig = self._create_distribution_comparison(human_topics, synthetic_topics)
        dist_fig.write_html(output_path / "topic_distribution_comparison.html")

        logger.info(f"Visualizations saved to {output_dir}")


    def _create_cluster_visualization(
        self, 
        intents: List[str], 
        topics: np.ndarray,
        embeddings: np.ndarray,
        title: str
    ) -> go.Figure:
        """
        Create 2D UMAP visualization for a single set of intents.
        """
        # Reduce to 2D for visualization
        umap_2d = UMAP(n_components=2, random_state=42, metric='cosine')
        embeddings_2d = umap_2d.fit_transform(embeddings)
        
        # Get topic info from current state
        topic_info = self.topic_wrapper.get_topic_info()
        
        # Create dataframe for plotting
        plot_df = pd.DataFrame({
            'x': embeddings_2d[:, 0],
            'y': embeddings_2d[:, 1],
            'topic': topics,
            'intent': [intent[:80] + '...' if len(intent) > 80 else intent for intent in intents],
        })
        
        # Add topic names if available
        if 'Name' in topic_info.columns and 'Topic' in topic_info.columns:
            topic_name_map = dict(zip(topic_info['Topic'], topic_info['Name']))
            plot_df['topic_name'] = plot_df['topic'].map(
                lambda x: topic_name_map.get(x, f'Topic {x}')
            )
        else:
            plot_df['topic_name'] = plot_df['topic'].apply(lambda x: f'Topic {x}')
        
        # Create interactive scatter plot
        fig = px.scatter(
            plot_df,
            x='x',
            y='y',
            color='topic',
            hover_data=['intent', 'topic_name'],
            title=f'{title} - Cluster Visualization (2D UMAP Projection)',
            labels={'x': 'UMAP 1', 'y': 'UMAP 2', 'topic': 'Topic ID'},
            color_continuous_scale='viridis',
            width=1000,
            height=700
        )
        
        fig.update_traces(marker=dict(size=5, opacity=0.6))
        fig.update_layout(
            font=dict(size=12),
            hoverlabel=dict(font_size=10)
        )
        
        return fig

    def _create_comparison_visualization(
        self,
        human_intents: List[str],
        synthetic_intents: List[str],
        human_topics: np.ndarray,
        synthetic_topics: np.ndarray,
        human_embeddings: np.ndarray,
        synthetic_embeddings: np.ndarray
    ) -> go.Figure:
        """
        Create visualization showing both human and synthetic intents together.
        """
        # Combine all embeddings
        all_embeddings = np.vstack([human_embeddings, synthetic_embeddings])
        
        # Reduce to 2D
        umap_2d = UMAP(n_components=2, random_state=42, metric='cosine')
        all_embeddings_2d = umap_2d.fit_transform(all_embeddings)
        
        # Split back to human and synthetic
        human_embeddings_2d = all_embeddings_2d[:len(human_intents)]
        synthetic_embeddings_2d = all_embeddings_2d[len(human_intents):]
        
        # Create dataframes
        human_df = pd.DataFrame({
            'x': human_embeddings_2d[:, 0],
            'y': human_embeddings_2d[:, 1],
            'topic': human_topics,
            'intent': [intent[:80] + '...' if len(intent) > 80 else intent 
                      for intent in human_intents],
            'type': 'Human'
        })
        
        synthetic_df = pd.DataFrame({
            'x': synthetic_embeddings_2d[:, 0],
            'y': synthetic_embeddings_2d[:, 1],
            'topic': synthetic_topics,
            'intent': [intent[:80] + '...' if len(intent) > 80 else intent 
                      for intent in synthetic_intents],
            'type': 'Synthetic'
        })
        
        combined_df = pd.concat([human_df, synthetic_df], ignore_index=True)
        
        # Create figure
        fig = px.scatter(
            combined_df,
            x='x',
            y='y',
            color='type',
            symbol='type',
            hover_data=['intent', 'topic'],
            title='Human vs Synthetic Intents Comparison (2D UMAP Projection)',
            labels={'x': 'UMAP 1', 'y': 'UMAP 2', 'type': 'Intent Type'},
            width=1000,
            height=700
        )
        
        fig.update_traces(marker=dict(size=5, opacity=0.6))
        fig.update_layout(
            font=dict(size=12),
            hoverlabel=dict(font_size=10)
        )
        
        return fig

    def _create_distribution_comparison(
        self, 
        human_topics: np.ndarray, 
        synthetic_topics: np.ndarray
    ) -> go.Figure:
        """
        Create bar chart comparing topic distributions.
        """
        # Count topics (including outliers)
        human_counts = pd.Series(human_topics).value_counts().sort_index()
        synthetic_counts = pd.Series(synthetic_topics).value_counts().sort_index()
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Human',
            x=human_counts.index,
            y=human_counts.values,
            marker_color='blue',
            opacity=0.7
        ))
        
        fig.add_trace(go.Bar(
            name='Synthetic', 
            x=synthetic_counts.index,
            y=synthetic_counts.values,
            marker_color='orange',
            opacity=0.7
        ))
        
        fig.update_layout(
            title='Topic Distribution Comparison',
            xaxis_title='Topic ID',
            yaxis_title='Number of Intents',
            barmode='group',
            width=1000,
            height=600,
            showlegend=True
        )
        
        return fig

    def get_topic_keywords(self, topic_id: int, top_n: int = 10) -> List[Tuple[str, float]]:
        """Get top keywords for a specific topic."""
        return self.topic_wrapper.get_topic(topic_id)

    def visualize_topics(self):
        """Visualize topics using BERTopic's built-in visualization."""
        return self.topic_wrapper.visualize_topics()

    def visualize_documents(self, docs: List[str]):
        """Visualize documents using BERTopic's built-in visualization."""
        return self.topic_wrapper.visualize_documents(docs)
