"""
Script for generating, clustering, and comparing synthetic and human-annotated intents.

Workflow:
1. Load and filter a human-annotated intent dataset from Hugging Face.
2. Optionally generate synthetic intents for each prompt using a specified model.
3. Cluster both human and synthetic intents jointly with BERTopic to compare semantic structure.
4. Optionally evaluate a model’s harm predictions against human annotations.

Usage:
    python run_intent_comparison.py \
        --dataset Jazhyc/wildguard-annotated-intents \
        --model RedHatAI/Qwen3-8B-quantized.w4a16 \
        --split train \
        --output-dir ./outputs

Optional flags:
    --skip-generation   Use existing synthetic intents from disk instead of generating new ones.
    --eval-harm         Run model harm evaluation (compute confusion matrix) instead of clustering comparison.
    --max-samples N     Limit number of human samples processed.

Outputs:
    - synthetic_intents.parquet: Generated synthetic intents (if applicable)
    - combined_topic_info.csv: Topic-level cluster metadata
    - comparison_results/: Visualizations of human vs. synthetic clusters
"""

import argparse
from pathlib import Path
import logging
import pandas as pd
from datasets import load_dataset
from transformers import Optional
from intention_jailbreak.comparison.clusteringcompare import ClusterComparator
from intention_jailbreak.comparison.syntheticintentgenerator import SyntheticIntentGenerator
from intention_jailbreak.comparison.wildguardmodelharmannotator import WildGuardModelHarmAnnotator

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic intents and evaluate against human-annotated harm labels"
    )
    parser.add_argument("--dataset", type=str, default="Jazhyc/wildguard-annotated-intents", help="HuggingFace dataset name")
    parser.add_argument("--model", type=str, default="RedHatAI/Qwen3-8B-quantized.w4a16", help="Model for generation/evaluation")
    parser.add_argument("--split", type=str, default="train", help="Dataset split")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of prompts to process")

    parser.add_argument("--output-dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--variations", type=int, default=1, help="Synthetic intents per prompt")

    parser.add_argument("--skip-generation", action="store_true", help="Skip synthetic generation and use existing data")
    parser.add_argument("--eval-harm", action="store_true")
    return parser.parse_args()


def load_and_filter_dataset(dataset_name: str, split: str, max_samples: Optional[int] = None) -> pd.DataFrame:
    logger.info("Loading human-annotated dataset")
    dataset = load_dataset(dataset_name, split=split)
    df_human = dataset.to_pandas()

    ambiguous_pattern = r'^\s*ambiguous\s*$|^\s*uncertain\s*$|^\s*unclear\s*$|^\s*$'
    df_human = df_human[
        ~df_human['Intent'].str.strip().str.lower().str.match(ambiguous_pattern, na=False)
    ].copy()
    df_human = df_human[
        df_human['Intent'].notna() & (df_human['Intent'].str.strip() != '')
    ].copy()

    if max_samples:
        df_human = df_human.sample(n=max_samples, random_state=42)

    return df_human


def evaluate_harm_model(args: argparse.Namespace) -> None:
    logger.info("Evaluating model against human annotations")
    evaluator = WildGuardModelHarmAnnotator(
        model_id=args.model,
        dataset_name=args.dataset,
        split=args.split,
        max_samples=args.max_samples
    )
    evaluator.evaluate_harm()


def generate_synthetic_intents(args: argparse.Namespace, df_human: pd.DataFrame, synthetic_path: Path) -> pd.DataFrame:
    if not args.skip_generation:
        logger.info("Generating synthetic intents")
        generator = SyntheticIntentGenerator(model_id=args.model)
        return generator.generate_synthetic_dataset(
            prompts=df_human['Prompt'].tolist(),
            num_variations=args.variations,
            output_path=str(synthetic_path)
        )

    logger.info(f"Loading existing synthetic data from {synthetic_path}")
    return pd.read_parquet(synthetic_path)


def compare_clusters(df_human: pd.DataFrame, df_synthetic: pd.DataFrame, output_dir: Path) -> None:
    logger.info("=" * 80)
    logger.info("CLUSTERING COMPARISON")
    logger.info("=" * 80)

    comparator = ClusterComparator()

    human_intents = df_human['Intent'].tolist()
    synthetic_intents = df_synthetic['synthetic_intent'].tolist()
    
    # Combine all intents for unified clustering
    all_intents = human_intents + synthetic_intents
    logger.info(f"Clustering {len(human_intents)} human + {len(synthetic_intents)} synthetic intents together")
    
    all_topics, _, all_topic_info = comparator.cluster_intents(
        all_intents, cache_prefix="combined_intents"
    )
    all_embeddings = comparator._current_embeddings
    
    # Split topics and embeddings back to human and synthetic
    human_topics = all_topics[:len(human_intents)]
    synthetic_topics = all_topics[len(human_intents):]
    human_embeddings = all_embeddings[:len(human_intents)]
    synthetic_embeddings = all_embeddings[len(human_intents):]

    # Visualize using the already-computed embeddings and topics
    comparator.visualize_comparison_with_embeddings(
        human_intents,
        synthetic_intents,
        human_topics,
        synthetic_topics,
        human_embeddings,
        synthetic_embeddings,
        output_dir=str(output_dir / "comparison_results")
    )

    all_topic_info.to_csv(output_dir / "combined_topic_info.csv", index=False)

    logger.info("=" * 80)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Results saved to {output_dir}")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Evaluate harm labels if requested
    if args.eval_harm:
        evaluate_harm_model(args)
        return

    # Load and filter dataset
    df_human = load_and_filter_dataset(args.dataset, args.split, args.max_samples)

    # Synthetic intent generation
    synthetic_path = output_dir / "synthetic_intents.parquet"
    df_synthetic = generate_synthetic_intents(args, df_human, synthetic_path)

    # Cluster comparison
    compare_clusters(df_human, df_synthetic, output_dir)


if __name__ == "__main__":
    main()
