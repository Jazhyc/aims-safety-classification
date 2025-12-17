import argparse
import json
from pathlib import Path
from statistics import mean, median, pstdev
import os
import evaluate

def load_std_jsonl(path: Path):
    """
    Load standardized JSONL with fields:
        { "id": ..., "true": "...", "pred": "..." }
    """
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    ids, refs, preds = [], [], []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)

            if "id" not in ex or "true" not in ex or "pred" not in ex:
                raise KeyError(
                    f"Expected keys 'id', 'true', 'pred' in example, got: {ex}"
                )

            ids.append(ex["id"])
            refs.append(str(ex["true"]))
            preds.append(str(ex["pred"]))

    if not ids:
        raise ValueError(f"No examples found in {path}")

    return ids, refs, preds


def compute_bleu_rouge(refs, preds):
    """
    Compute BLEU and ROUGE scores for lists of references and predictions.
    
    Returns:
        dict with keys:
            - 'bleu': Google BLEU scores (list)
            - 'rouge1': ROUGE-1 F1 scores (list)
            - 'rouge2': ROUGE-2 F1 scores (list)
            - 'rougeL': ROUGE-L F1 scores (list)
    """
    if len(refs) != len(preds):
        raise ValueError(
            f"refs and preds length mismatch: {len(refs)} vs {len(preds)}"
        )

    # Load metrics once
    bleu_metric = evaluate.load("google_bleu")
    rouge_metric = evaluate.load("rouge")

    # BLEU: must compute per-example as google_bleu doesn't return per-example scores
    bleu_scores = []
    for ref, pred in zip(refs, preds):
        bleu_result = bleu_metric.compute(predictions=[pred], references=[[ref]])
        bleu_scores.append(bleu_result["google_bleu"])
    
    # ROUGE: compute all at once (much faster)
    rouge_result = rouge_metric.compute(
        predictions=preds, 
        references=refs,
        use_aggregator=False  # Return per-example scores
    )

    return {
        "bleu": bleu_scores,
        "rouge1": rouge_result["rouge1"],
        "rouge2": rouge_result["rouge2"],
        "rougeL": rouge_result["rougeL"],
    }


def compute_and_log_metrics(refs, preds, split_name="val", wandb_log=True):
    """
    Compute BLEU and ROUGE scores and optionally log to wandb.
    
    Args:
        refs: List of reference strings
        preds: List of prediction strings
        split_name: Name of the split (e.g., "val", "test")
        wandb_log: Whether to log to wandb
    
    Returns:
        dict: Mean metrics with keys like {split}_bleu, {split}_rouge1, etc.
    """
    scores = compute_bleu_rouge(refs, preds)
    
    # Calculate mean scores
    metrics = {
        f"{split_name}_bleu": sum(scores["bleu"]) / len(scores["bleu"]),
        f"{split_name}_rouge1": sum(scores["rouge1"]) / len(scores["rouge1"]),
        f"{split_name}_rouge2": sum(scores["rouge2"]) / len(scores["rouge2"]),
        f"{split_name}_rougeL": sum(scores["rougeL"]) / len(scores["rougeL"]),
    }
    
    # Print metrics
    print(f"\n=== {split_name.capitalize()} Metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    
    # Log to wandb if available and requested
    if wandb_log:
        try:
            import wandb
            if wandb.run is not None:
                wandb.log(metrics)
        except ImportError:
            pass
    
    return metrics



def save_per_example(
    path: Path, ids, refs, preds, all_scores, model_name: str | None = None
):
    """
    Save per-example BLEU and ROUGE scores to JSONL:
        { "id": ..., "true": "...", "pred": "...", "bleu": 0.5, "rouge1": 0.6, "rouge2": 0.4, "rougeL": 0.55, "model": "..." }
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for i, ex_id in enumerate(ids):
            obj = {
                "id": ex_id,
                "true": refs[i],
                "pred": preds[i],
                "bleu": float(all_scores["bleu"][i]),
                "rouge1": float(all_scores["rouge1"][i]),
                "rouge2": float(all_scores["rouge2"][i]),
                "rougeL": float(all_scores["rougeL"][i]),
            }
            if model_name is not None:
                obj["model"] = model_name
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    print(f"Per-example metrics written to: {path}")


def append_summary_csv(
    path: Path,
    model_name: str,
    split: str,
    all_scores,
):
    """
    Append a summary row to a CSV file with columns:
        model,split,n,bleu_mean,bleu_median,bleu_std,rouge1_mean,rouge1_median,rouge1_std,rouge2_mean,rouge2_median,rouge2_std,rougeL_mean,rougeL_median,rougeL_std
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    n = len(all_scores["bleu"])
    
    metrics = {}
    for metric_name in ["bleu", "rouge1", "rouge2", "rougeL"]:
        scores = all_scores[metric_name]
        metrics[metric_name] = {
            "mean": mean(scores),
            "median": median(scores),
            "std": pstdev(scores) if n > 1 else 0.0,
        }

    new_line = (
        f"{model_name},{split},{n},"
        f"{metrics['bleu']['mean']:.6f},{metrics['bleu']['median']:.6f},{metrics['bleu']['std']:.6f},"
        f"{metrics['rouge1']['mean']:.6f},{metrics['rouge1']['median']:.6f},{metrics['rouge1']['std']:.6f},"
        f"{metrics['rouge2']['mean']:.6f},{metrics['rouge2']['median']:.6f},{metrics['rouge2']['std']:.6f},"
        f"{metrics['rougeL']['mean']:.6f},{metrics['rougeL']['median']:.6f},{metrics['rougeL']['std']:.6f}\n"
    )

    header = "model,split,n,bleu_mean,bleu_median,bleu_std,rouge1_mean,rouge1_median,rouge1_std,rouge2_mean,rouge2_median,rouge2_std,rougeL_mean,rougeL_median,rougeL_std\n"
    if not path.exists():
        with path.open("w", encoding="utf-8") as f:
            f.write(header)
            f.write(new_line)
    else:
        with path.open("a", encoding="utf-8") as f:
            f.write(new_line)

    print(f"Summary appended to: {path}")
    print("--- Summary row ---")
    print(new_line.strip())


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compute BLEU and ROUGE scores for a standardized predictions JSONL "
            "({id, true, pred})."
        )
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        type=str,
        help="Path to standardized JSONL input (e.g. eval_data/t5-base_val.std.jsonl).",
    )
    parser.add_argument(
        "--model-name",
        "-m",
        type=str,
        default=None,
        help="Optional model name for logging/summary (e.g. 't5-base').",
    )
    parser.add_argument(
        "--split",
        "-s",
        type=str,
        default="val",
        help="Data split name for summary reporting (e.g. 'val', 'test').",
    )
    parser.add_argument(
        "--per-example-out",
        type=str,
        default=None,
        help=(
            "Optional path to write per-example JSONL with metric scores. "
            "If omitted, only aggregate stats are printed."
        ),
    )
    parser.add_argument(
        "--summary-csv",
        type=str,
        default=None,
        help=(
            "Optional path to a CSV file to append a summary row "
            "with BLEU and ROUGE statistics."
        ),
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    ids, refs, preds = load_std_jsonl(input_path)

    print(f"Loaded {len(ids)} examples from {input_path}")
    print(f"Computing BLEU and ROUGE scores...")

    all_scores = compute_bleu_rouge(refs, preds)

    # Aggregate stats
    n = len(all_scores["bleu"])
    
    print("=== Metric statistics ===")
    print(f"n: {n}")
    for metric_name in ["bleu", "rouge1", "rouge2", "rougeL"]:
        scores = all_scores[metric_name]
        m = mean(scores)
        med = median(scores)
        std = pstdev(scores) if n > 1 else 0.0
        print(f"{metric_name:8s} - mean: {m:.6f}, median: {med:.6f}, std: {std:.6f}")

    # Optional per-example output
    if args.per_example_out is not None:
        per_example_path = Path(args.per_example_out)
        save_per_example(
            per_example_path,
            ids,
            refs,
            preds,
            all_scores,
            model_name=args.model_name,
        )

    # Optional summary CSV
    if args.summary_csv is not None and args.model_name is not None:
        summary_path = Path(args.summary_csv)
        append_summary_csv(
            summary_path, args.model_name, args.split, all_scores
        )
    elif args.summary_csv is not None and args.model_name is None:
        print(
            "Warning: --summary-csv was provided but --model-name is None; "
            "skipping summary CSV."
        )


if __name__ == "__main__":
    main()
