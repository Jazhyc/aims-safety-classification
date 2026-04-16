#!/usr/bin/env python3
"""
Generate a LaTeX benchmark comparison table by querying W&B projects.

Fetches F1 scores from W&B for:
- Baselines (GPT-OSS, WildGuard, LlamaGuard 4)
- SFT results (Classification, Generation)
- Best Distillation (Gemma-3-12B student with GPT-OSS-120B teacher)

Generates a formatted LaTeX table with bold/underline for top 2 results.

Usage:
    python scripts/report/generate_benchmark_table.py --output report/latex/benchmark_table.tex
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


CACHE_DIR = Path(".cache/benchmark_results")
CACHE_EXPIRY = 3600  # 1 hour in seconds


@dataclass
class ModelResult:
    """Single model's results across datasets."""
    name: str
    is_baseline: bool
    scores: Dict[str, float]  # {dataset: f1_score}


def load_cache(cache_key: str) -> Optional[Dict]:
    """Load results from cache if valid (not expired)."""
    cache_file = CACHE_DIR / f"{cache_key}.json"

    if not cache_file.exists():
        return None

    try:
        with open(cache_file, 'r') as f:
            data = json.load(f)

        # Check if cache is expired
        if time.time() - data.get("timestamp", 0) > CACHE_EXPIRY:
            return None

        return data.get("results")
    except Exception:
        return None


def save_cache(cache_key: str, results: Dict):
    """Save results to cache with timestamp."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{cache_key}.json"

    try:
        with open(cache_file, 'w') as f:
            json.dump({
                "timestamp": time.time(),
                "results": results
            }, f, indent=2)
    except Exception as e:
        print(f"Warning: Could not save cache: {e}")


def fetch_wandb_results(
    entity: str,
    project: str,
    filters: Optional[Dict] = None,
    metric_name: str = "f1_binary",
    use_cache: bool = True,
    force_refresh: bool = False,
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Fetch results from W&B project.
    Returns {dataset: [(condition, f1), ...]}
    """
    if not HAS_WANDB:
        print("Warning: wandb not installed, skipping W&B results")
        return {}

    cache_key = f"{entity}_{project}"

    # Try to load from cache
    if use_cache and not force_refresh:
        cached = load_cache(cache_key)
        if cached is not None:
            print(f"✓ Loaded {project} from cache")
            return cached

    print(f"Fetching {project} from W&B...")
    api = wandb.Api()
    results = {}

    try:
        runs = api.runs(f"{entity}/{project}", filters=filters or {})

        for run in runs:
            if run.state != "finished":
                continue

            summary = run.summary
            config = run.config

            # Extract dataset from run config or name
            dataset = config.get("dataset", {}).get("name") or run.name.split("-")[-1]

            # Get F1 score
            f1 = summary.get(metric_name)
            if f1 is None:
                continue

            # Get condition/model name
            condition = config.get("condition") or summary.get("condition") or run.name

            if dataset not in results:
                results[dataset] = []

            results[dataset].append((condition, f1))

        # Save to cache
        if use_cache:
            save_cache(cache_key, results)

    except Exception as e:
        print(f"Warning: Could not fetch results from {project}: {e}")

    return results


def get_hardcoded_results() -> Dict[str, Dict[str, float]]:
    """
    Return hardcoded benchmark results.
    Users can update these values from W&B or compute them directly.
    """
    return {
        # Baselines (from baselines_results.ipynb plot)
        "GPT-OSS-Safeguard 120B": {
            "wildguardmix": 0.871,
            "xstest": 0.944,
            "aegis": 0.797,
            "toxic_chat": 0.643,
            "openai_moderation": 0.780,
        },
        "WildGuard": {
            "wildguardmix": 0.888,
            "xstest": 0.945,
            "aegis": 0.809,
            "toxic_chat": 0.652,
            "openai_moderation": 0.724,
        },
        "LlamaGuard 4": {
            "wildguardmix": 0.738,
            "xstest": 0.836,
            "aegis": 0.705,
            "toxic_chat": 0.441,
            "openai_moderation": 0.736,
        },
        "GuardReasoner 8B": {
            "wildguardmix": 0.8881,
            "xstest": 0.9187,
            "aegis": 0.8296,
            "toxic_chat": 0.6813,
            "openai_moderation": 0.7040,
        },
        # SFT Generation (from baselines_results.ipynb plot)
        "SFT Generation": {
            "wildguardmix": 0.863,
            "xstest": 0.900,
            "aegis": 0.797,
            "toxic_chat": 0.655,
            "openai_moderation": 0.737,
        },
        # Best Distillation: Gemma-3-12B (student) with GPT-OSS-120B (teacher)
        "Distillation": {
            "wildguardmix": 0.879,
            "xstest": 0.923,
            "aegis": 0.817,
            "toxic_chat": 0.681,
            "openai_moderation": 0.796,
        },
    }


def rank_models_per_dataset(models: List[ModelResult]) -> Dict[str, List[tuple]]:
    """
    For each dataset, rank models by score (1st, 2nd, etc).
    Returns {dataset: [(rank, model), ...]}.
    """
    datasets = set()
    for model in models:
        datasets.update(model.scores.keys())

    rankings = {}
    for dataset in sorted(datasets):
        # Get all models with scores for this dataset
        scored = [(model.scores[dataset], model) for model in models if dataset in model.scores]
        # Sort by score descending
        scored.sort(key=lambda x: x[0], reverse=True)
        # Assign ranks
        rankings[dataset] = [(i, model) for i, (_, model) in enumerate(scored)]

    return rankings


def format_score(score: float, rank: int) -> str:
    """Format a score with LaTeX bold/underline for top 2."""
    score_str = f"{score:.3f}"
    if rank == 0:  # Best
        return f"$\\bm{{{score_str}}}$"
    elif rank == 1:  # Second best
        return f"$\\underline{{{score_str}}}$"
    else:
        return score_str


def compute_average(scores: Dict[str, float], datasets: List[str]) -> float:
    """Compute average F1 across datasets."""
    valid_scores = [scores[d] for d in datasets if d in scores]
    if not valid_scores:
        return 0.0
    return sum(valid_scores) / len(valid_scores)


def generate_latex_table(
    models: List[ModelResult],
    datasets: List[str],
) -> str:
    """Generate the LaTeX table code."""

    # Rank models per dataset
    rankings = rank_models_per_dataset(models)

    # Compute averages for ranking
    averages = {model.name: compute_average(model.scores, datasets) for model in models}
    avg_ranking = sorted([(avg, name) for name, avg in averages.items()], reverse=True)
    avg_ranks = {name: rank for rank, (_, name) in enumerate(avg_ranking)}

    # Build LaTeX table
    lines = []
    lines.append(r"\begin{table*}[ht]")
    lines.append(r"\centering")
    lines.append(r"\small")

    # Column spec: Condition + datasets + Average (booktabs style - no vertical bars)
    col_spec = "l" + "c" * (len(datasets) + 1)

    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")

    # Header row
    header = r"\textbf{Condition}"
    for dataset in datasets:
        # Format dataset names
        ds_name = {
            "wildguardmix": "WGTest",
            "xstest": "XSTest",
            "aegis": "AEGIS 2",
            "toxic_chat": "ToxicChat",
            "openai_moderation": "OAI Mod",
        }.get(dataset, dataset.replace("_", " ").title())
        header += f" & \\textbf{{{ds_name}}}"
    header += r" & \textbf{Average}"
    header += r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    # Baselines section
    baseline_models = [m for m in models if m.is_baseline]
    for model in baseline_models:
        row = f"\\textit{{{model.name}}}"
        for dataset in datasets:
            if dataset in model.scores:
                rank = next(r for r, m in rankings[dataset] if m.name == model.name)
                score = format_score(model.scores[dataset], rank)
                row += f" & {score}"
            else:
                row += " & —"
        # Add average
        avg = averages[model.name]
        avg_rank = avg_ranks[model.name]
        avg_score = format_score(avg, avg_rank)
        row += f" & {avg_score}"
        row += r" \\"
        lines.append(row)

    # Separator
    lines.append(r"\midrule")

    # Our results section
    our_models = [m for m in models if not m.is_baseline]
    for model in our_models:
        row = f"\\textbf{{{model.name}}}"
        for dataset in datasets:
            if dataset in model.scores:
                rank = next(r for r, m in rankings[dataset] if m.name == model.name)
                score = format_score(model.scores[dataset], rank)
                row += f" & {score}"
            else:
                row += " & —"
        # Add average
        avg = averages[model.name]
        avg_rank = avg_ranks[model.name]
        avg_score = format_score(avg, avg_rank)
        row += f" & {avg_score}"
        row += r" \\"
        lines.append(row)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{F1 score comparison across benchmarks. "
                  r"Best results are bolded; second-best are underlined.}")
    lines.append(r"\label{fig:f1_benchmark}")
    lines.append(r"\end{table*}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None,
                       help="Output LaTeX file (default: print to stdout)")
    parser.add_argument("--results-file", type=Path, default=None,
                       help="Optional JSON file with results to use instead of hardcoded values")
    parser.add_argument("--datasets", type=str,
                       default="wildguardmix,xstest,aegis,toxic_chat,openai_moderation",
                       help="Comma-separated list of datasets in order")
    parser.add_argument("--use-cache", action="store_true", default=True,
                       help="Use cached W&B results if available (default: True)")
    parser.add_argument("--force-refresh", action="store_true",
                       help="Force refresh W&B results and skip cache")
    parser.add_argument("--use-wandb", action="store_true",
                       help="Fetch results from W&B projects (requires wandb and credentials)")

    args = parser.parse_args()

    # Load results
    if args.results_file and args.results_file.exists():
        print(f"Loading results from {args.results_file}")
        with open(args.results_file, 'r') as f:
            all_results = json.load(f)
    elif args.use_wandb:
        print("Fetching results from W&B...")
        all_results = {}

        # Fetch from different W&B projects
        # TODO: Implement W&B fetching for baselines, sft, and distillation projects
        # For now, fall back to hardcoded results
        print("Using hardcoded benchmark results as fallback")
        all_results = get_hardcoded_results()
    else:
        print("Using hardcoded benchmark results")
        all_results = get_hardcoded_results()

    # Parse dataset names
    datasets = [d.strip() for d in args.datasets.split(",")]
    baseline_names = ["GPT-OSS-Safeguard 120B", "WildGuard", "LlamaGuard 4", "GuardReasoner 8B"]

    # Build model list
    models = []

    # Add baselines first (in order)
    for name in baseline_names:
        if name in all_results:
            models.append(ModelResult(
                name=name,
                is_baseline=True,
                scores=all_results[name],
            ))

    # Add our results in specific order: SFT Generation, Distillation
    our_results_order = ["SFT Generation", "Distillation"]
    for name in our_results_order:
        if name in all_results:
            models.append(ModelResult(
                name=name,
                is_baseline=False,
                scores=all_results[name],
            ))

    # Generate LaTeX
    latex_code = generate_latex_table(models, datasets)

    # Output
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w') as f:
            f.write(latex_code)
        print(f"✓ LaTeX table written to {args.output}")
    else:
        print(latex_code)


if __name__ == "__main__":
    main()
