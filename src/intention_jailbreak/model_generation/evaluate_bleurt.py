import argparse
import json
from pathlib import Path
from statistics import mean, median, pstdev
import os
import evaluate
os.environ["CUDA_VISIBLE_DEVICES"] = ""

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


def compute_bleurt(refs, preds, checkpoint: str = "bleurt-20", batch_size: int = 32):
    """
    Compute BLEURT scores for lists of references and predictions.

    Some versions of the BLEURT metric (via `evaluate`) do not accept
    `batch_size` in `.compute()`, so we manually batch using `add_batch`.
    """
    if len(refs) != len(preds):
        raise ValueError(
            f"refs and preds length mismatch: {len(refs)} vs {len(preds)}"
        )

    # Load BLEURT metric with the chosen checkpoint
    metric = evaluate.load("bleurt", checkpoint)

    # Manually add in mini-batches
    if batch_size is None or batch_size <= 0:
        batch_size = len(refs)

    for i in range(0, len(refs), batch_size):
        batch_refs = refs[i : i + batch_size]
        batch_preds = preds[i : i + batch_size]
        metric.add_batch(predictions=batch_preds, references=batch_refs)

    results = metric.compute()
    scores = results["scores"]

    if len(scores) != len(refs):
        raise RuntimeError(
            f"BLEURT returned {len(scores)} scores for {len(refs)} examples."
        )

    return scores



def save_per_example(
    path: Path, ids, refs, preds, scores, model_name: str | None = None
):
    """
    Save per-example BLEURT scores to JSONL:
        { "id": ..., "true": "...", "pred": "...", "bleurt": 0.123, "model": "..." }
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for i, (ex_id, ref, pred, sc) in enumerate(
            zip(ids, refs, preds, scores), start=1
        ):
            obj = {
                "id": ex_id,
                "true": ref,
                "pred": pred,
                "bleurt": float(sc),
            }
            if model_name is not None:
                obj["model"] = model_name
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    print(f"Per-example BLEURT written to: {path}")


def append_summary_csv(
    path: Path,
    model_name: str,
    split: str,
    scores,
):
    """
    Append a summary row to a CSV file with columns:
        model,split,n,bleurt_mean,bleurt_median,bleurt_std
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    n = len(scores)
    m = mean(scores)
    med = median(scores)
    std = pstdev(scores) if n > 1 else 0.0

    new_line = f"{model_name},{split},{n},{m:.6f},{med:.6f},{std:.6f}\n"

    header = "model,split,n,bleurt_mean,bleurt_median,bleurt_std\n"
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
            "Compute BLEURT scores for a standardized predictions JSONL "
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
        "--checkpoint",
        "-c",
        type=str,
        default="bleurt-20",
        help=(
            "BLEURT checkpoint name (e.g. 'bleurt-20', 'bleurt-base-128', "
            "'bleurt-tiny-128'). Default: 'bleurt-20'."
        ),
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=32,
        help="Batch size for BLEURT computation. Default: 32.",
    )
    parser.add_argument(
        "--per-example-out",
        type=str,
        default=None,
        help=(
            "Optional path to write per-example JSONL with BLEURT scores. "
            "If omitted, only aggregate stats are printed."
        ),
    )
    parser.add_argument(
        "--summary-csv",
        type=str,
        default=None,
        help=(
            "Optional path to a CSV file to append a summary row "
            "(model,split,n,bleurt_mean,bleurt_median,bleurt_std)."
        ),
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    ids, refs, preds = load_std_jsonl(input_path)

    print(f"Loaded {len(ids)} examples from {input_path}")
    print(f"Computing BLEURT with checkpoint '{args.checkpoint}'...")

    scores = compute_bleurt(
        refs, preds, checkpoint=args.checkpoint, batch_size=args.batch_size
    )

    # Aggregate stats
    n = len(scores)
    m = mean(scores)
    med = median(scores)
    std = pstdev(scores) if n > 1 else 0.0

    print("=== BLEURT statistics ===")
    print(f"n              : {n}")
    print(f"mean           : {m:.6f}")
    print(f"median         : {med:.6f}")
    print(f"std (population): {std:.6f}")

    # Optional per-example output
    if args.per_example_out is not None:
        per_example_path = Path(args.per_example_out)
        save_per_example(
            per_example_path,
            ids,
            refs,
            preds,
            scores,
            model_name=args.model_name,
        )

    # Optional summary CSV
    if args.summary_csv is not None and args.model_name is not None:
        summary_path = Path(args.summary_csv)
        append_summary_csv(
            summary_path, args.model_name, args.split, scores
        )
    elif args.summary_csv is not None and args.model_name is None:
        print(
            "Warning: --summary-csv was provided but --model-name is None; "
            "skipping summary CSV."
        )


if __name__ == "__main__":
    main()
