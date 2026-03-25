import argparse
import json
import re
from pathlib import Path
from statistics import mean, median, pstdev
from sentence_transformers import SentenceTransformer
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics.pairwise import cosine_similarity


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


def compute_semantic_similarity(refs, preds, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """
    Compute cosine similarity between references and predictions using sentence embeddings.

    Returns:
        list: Cosine similarity scores (one per example)
    """
    if len(refs) != len(preds):
        raise ValueError(
            f"refs and preds length mismatch: {len(refs)} vs {len(preds)}"
        )

    model = SentenceTransformer(model_name)
    ref_embeddings = model.encode(refs, convert_to_numpy=True, show_progress_bar=False)
    pred_embeddings = model.encode(preds, convert_to_numpy=True, show_progress_bar=False)

    sim_matrix = cosine_similarity(ref_embeddings, pred_embeddings)
    return [float(sim_matrix[i, i]) for i in range(len(ref_embeddings))]


def extract_harm_labels(completions):
    """Extract binary harm label from completions.

    Handles two output formats:
      - Joint generation: "Intent: <text>; Harm: safe/harmful"
      - Classification-only: bare "safe" or "harmful"

    Returns None for entries where the label cannot be parsed.
    """
    labels = []
    for c in completions:
        # Try structured format first: "Harm: safe/harmful"
        m = re.search(r'Harm:\s*(\w+)', c, re.IGNORECASE)
        if m:
            labels.append(m.group(1).lower())
        else:
            # Fallback: bare label (classification_only mode)
            stripped = c.strip().lower()
            labels.append(stripped if stripped in ("harmful", "safe") else None)
    return labels


def compute_harm_f1(refs, preds):
    """Compute harm classification F1/precision/recall from completion strings.
    Skips examples where either ref or pred label cannot be parsed.
    Returns None if no valid examples exist."""
    ref_labels = extract_harm_labels(refs)
    pred_labels = extract_harm_labels(preds)

    pairs = [(r, p) for r, p in zip(ref_labels, pred_labels) if r is not None and p is not None]
    if not pairs:
        return None

    y_true, y_pred = zip(*pairs)
    return {
        "f1": f1_score(y_true, y_pred, pos_label="harmful", average="binary", zero_division=0),
        "precision": precision_score(y_true, y_pred, pos_label="harmful", average="binary", zero_division=0),
        "recall": recall_score(y_true, y_pred, pos_label="harmful", average="binary", zero_division=0),
        "n": len(pairs),
    }


def compute_and_log_metrics(refs, preds, split_name="val", wandb_log=True):
    """
    Compute semantic similarity and harm F1, and optionally log to wandb.

    Returns:
        dict: Metrics with keys like {split}_semantic_sim, {split}_harm_f1, etc.
    """
    semantic_scores = compute_semantic_similarity(refs, preds)

    metrics = {
        f"{split_name}_semantic_sim": mean(semantic_scores),
    }

    harm = compute_harm_f1(refs, preds)
    if harm is not None:
        metrics[f"{split_name}_harm_f1"] = harm["f1"]
        metrics[f"{split_name}_harm_precision"] = harm["precision"]
        metrics[f"{split_name}_harm_recall"] = harm["recall"]

    print(f"\n=== {split_name.capitalize()} Metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    if wandb_log:
        try:
            import wandb
            if wandb.run is not None:
                wandb.log(metrics)
        except ImportError:
            pass

    return metrics


def save_per_example(
    path: Path, ids, refs, preds, semantic_scores, model_name: str | None = None
):
    """
    Save per-example semantic similarity scores to JSONL:
        { "id": ..., "true": "...", "pred": "...", "semantic_sim": 0.85, "model": "..." }
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for i, ex_id in enumerate(ids):
            obj = {
                "id": ex_id,
                "true": refs[i],
                "pred": preds[i],
                "semantic_sim": float(semantic_scores[i]),
            }
            if model_name is not None:
                obj["model"] = model_name
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    print(f"Per-example metrics written to: {path}")


def append_summary_csv(
    path: Path,
    model_name: str,
    split: str,
    semantic_scores: list,
):
    """
    Append a summary row to a CSV file with columns:
        model,split,n,semantic_sim_mean,semantic_sim_median,semantic_sim_std
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    n = len(semantic_scores)
    sem_mean   = mean(semantic_scores)
    sem_median = median(semantic_scores)
    sem_std    = pstdev(semantic_scores) if n > 1 else 0.0

    new_line = f"{model_name},{split},{n},{sem_mean:.6f},{sem_median:.6f},{sem_std:.6f}\n"
    header   = "model,split,n,semantic_sim_mean,semantic_sim_median,semantic_sim_std\n"

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
        description="Compute semantic similarity and harm F1 for a standardized predictions JSONL ({id, true, pred})."
    )
    parser.add_argument("--input", "-i", required=True, type=str,
                        help="Path to standardized JSONL input.")
    parser.add_argument("--model-name", "-m", type=str, default=None,
                        help="Optional model name for logging/summary.")
    parser.add_argument("--split", "-s", type=str, default="val",
                        help="Data split name for summary reporting (e.g. 'val', 'test').")
    parser.add_argument("--per-example-out", type=str, default=None,
                        help="Optional path to write per-example JSONL with metric scores.")
    parser.add_argument("--summary-csv", type=str, default=None,
                        help="Optional path to a CSV file to append a summary row.")

    args = parser.parse_args()

    input_path = Path(args.input)
    ids, refs, preds = load_std_jsonl(input_path)

    print(f"Loaded {len(ids)} examples from {input_path}")
    print("Computing semantic similarity and harm F1...")

    semantic_scores = compute_semantic_similarity(refs, preds)
    n = len(semantic_scores)

    print("=== Metric statistics ===")
    print(f"n: {n}")
    sem_mean   = mean(semantic_scores)
    sem_median = median(semantic_scores)
    sem_std    = pstdev(semantic_scores) if n > 1 else 0.0
    print(f"{'semantic_sim':12s} - mean: {sem_mean:.6f}, median: {sem_median:.6f}, std: {sem_std:.6f}")

    harm = compute_harm_f1(refs, preds)
    if harm is not None:
        print(f"harm_f1      - f1: {harm['f1']:.6f}, precision: {harm['precision']:.6f}, recall: {harm['recall']:.6f}  (n={harm['n']})")

    if args.per_example_out is not None:
        save_per_example(Path(args.per_example_out), ids, refs, preds, semantic_scores, model_name=args.model_name)

    if args.summary_csv is not None and args.model_name is not None:
        append_summary_csv(Path(args.summary_csv), args.model_name, args.split, semantic_scores)
    elif args.summary_csv is not None and args.model_name is None:
        print("Warning: --summary-csv provided but --model-name is None; skipping summary CSV.")


if __name__ == "__main__":
    main()
