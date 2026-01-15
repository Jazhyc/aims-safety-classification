import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix


def load_harm_jsonl(path: Path):
    ids, y_true, y_pred = [], [], []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            ids.append(ex.get("id"))
            y_true.append(ex.get("true_harm"))
            y_pred.append(ex.get("pred_harm"))
    return ids, y_true, y_pred


def main():
    parser = argparse.ArgumentParser(description="Evaluate harm classification predictions JSONL.")
    parser.add_argument("--input", "-i", required=True, type=str, help="Path to harm predictions JSONL.")
    parser.add_argument("--labels", "-l", default=None, type=str, help="Optional comma-separated label order.")
    parser.add_argument("--print-matrix", action="store_true", help="Print confusion matrix.")
    args = parser.parse_args()

    path = Path(args.input)
    ids, y_true, y_pred = load_harm_jsonl(path)

    # Filter missing truths (if any)
    pairs = [(t, p) for t, p in zip(y_true, y_pred) if t is not None and str(t).strip() != ""]
    if not pairs:
        raise ValueError("No non-empty true_harm labels found in file.")

    y_true_f = [t for t, _ in pairs]
    y_pred_f = [p for _, p in pairs]

    labels = None
    if args.labels:
        labels = [x.strip() for x in args.labels.split(",") if x.strip()]

    f1_macro = f1_score(y_true_f, y_pred_f, average="macro", labels=labels)
    acc = accuracy_score(y_true_f, y_pred_f)

    print(f"File: {path}")
    print(f"n: {len(y_true_f)}")
    print(f"accuracy: {acc:.4f}")
    print(f"f1_macro: {f1_macro:.4f}\n")

    print("Classification report:")
    print(classification_report(y_true_f, y_pred_f, labels=labels, zero_division=0))

    if args.print_matrix:
        used_labels = labels if labels is not None else sorted(set(y_true_f) | set(y_pred_f))
        cm = confusion_matrix(y_true_f, y_pred_f, labels=used_labels)
        print("\nConfusion matrix (rows=true, cols=pred):")
        print("labels:", used_labels)
        print(cm)


if __name__ == "__main__":
    main()
