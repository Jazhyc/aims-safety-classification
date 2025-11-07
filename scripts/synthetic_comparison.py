import argparse
from pathlib import Path
import logging
import pandas as pd
from datasets import load_dataset
from intention_jailbreak.comparison.clusteringcompare import ClusterComparator
from intention_jailbreak.comparison.syntheticintentgenerator import SyntheticIntentGenerator
from intention_jailbreak.comparison.wildguardmodelharmannotator import WildGuardModelHarmAnnotator

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate synthetic intents and evaluate against human-annotated harm labels"
    )
    parser.add_argument("--dataset", type=str, default="Jazhyc/wildguard-annotated-intents", help="HuggingFace dataset name")
    parser.add_argument("--model", type=str, default="RedHatAI/Qwen3-8B-quantized.w4a16", help="Model for generation/evaluation")
    parser.add_argument("--split", type=str, default="train", help="Dataset split")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of prompts to process")
    parser.add_argument("--num-samples", type=int, default=None, help="Number of prompts to process for synthetic generation")

    parser.add_argument("--output-dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--variations", type=int, default=1, help="Synthetic intents per prompt")

    parser.add_argument("--skip-generation", action="store_true", help="Skip synthetic generation and use existing data")
    parser.add_argument("--eval-harm", action="store_true")
    return parser.parse_args()


def load_and_filter_dataset(dataset_name, split, max_samples=None):
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


def evaluate_harm_model(args):
    logger.info("Evaluating model against human annotations")
    evaluator = WildGuardModelHarmAnnotator(
        model_id=args.model,
        dataset_name=args.dataset,
        split=args.split,
        max_samples=args.max_samples
    )
    evaluator.evaluate_harm()


def generate_synthetic_intents(args, df_human, synthetic_path):
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


def compare_clusters(df_human, df_synthetic, output_dir):
    logger.info("=" * 80)
    logger.info("CLUSTERING COMPARISON")
    logger.info("=" * 80)

    comparator = ClusterComparator()

    human_intents = df_human['Intent'].tolist()
    human_topics, _, human_topic_info = comparator.cluster_intents(
        human_intents, cache_prefix="human_intents_filtered"
    )

    synthetic_intents = df_synthetic['synthetic_intent'].tolist()
    synthetic_topics, _, synthetic_topic_info = comparator.cluster_intents(
        synthetic_intents, cache_prefix="synthetic_intents"
    )

    comparator.visualize_comparison(
        human_intents,
        synthetic_intents,
        human_topics,
        synthetic_topics,
        output_dir=str(output_dir / "comparison_results")
    )

    human_topic_info.to_csv(output_dir / "human_topic_info.csv", index=False)
    synthetic_topic_info.to_csv(output_dir / "synthetic_topic_info.csv", index=False)

    logger.info("=" * 80)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Results saved to {output_dir}")


def main():
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
