#!/usr/bin/env python3
"""Quick summary table of all DPO condition results.

Usage:
    python scripts/dpo/check_dpo_results.py
    python scripts/dpo/check_dpo_results.py --base /custom/path
"""

import argparse
import json
from pathlib import Path

try:
    from sklearn.metrics import f1_score, accuracy_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


CONDITIONS = {
    "hard_dpo":               "dpo-hard",
    "hard_dpo-seed42":        "dpo-hard-seed42",
    "hard_dpo-ep1":           "dpo-hard-ep1",
    "hard_dpo-beta0.1":       "dpo-hard-beta0.1",
    "hard_dpo-beta0.3":       "dpo-hard-beta0.3",
    "judge_standalone":       "dpo-judge-standalone",
    "judge_dpo":              "dpo-judge",
    "judge_gptoss":           "dpo-judge-gptoss",
    "judge_gptoss_standalone": "dpo-judge-gptoss-standalone",
    "judge_decent":           "dpo-judge-decent",
    "union_decent":           "dpo-union-decent",
    "judge_union":            "dpo-judge-union",
}

DATASET_ORDER = [
    "annotated_intents",
    "annotated_intents_val",
    "wildguardmix",
    "xstest",
    "toxic_chat",
    "aegis",
    "openai_moderation",
]

SHORT_NAMES = {
    "annotated_intents":     "ann-test",
    "annotated_intents_val": "ann-val",
    "wildguardmix":          "wildguard",
    "xstest":                "xstest",
    "toxic_chat":            "toxic",
    "aegis":                 "aegis",
    "openai_moderation":     "openai",
}

MODEL_SLUG = "meta-llama_Llama-3.1-8B-Instruct_finetuned_generation.jsonl"


def find_pred_file(pred_dir: Path, ds_key: str) -> Path | None:
    """Find prediction file, tolerating hyphen/underscore variants in dir name."""
    if not pred_dir.exists():
        return None
    for variant in [ds_key, ds_key.replace("_", "-")]:
        p = pred_dir / variant / MODEL_SLUG
        if p.exists():
            return p
    # Fallback: glob for any subdir containing the file
    for p in pred_dir.glob(f"*/{MODEL_SLUG}"):
        stem = p.parent.name.replace("-", "_")
        if stem == ds_key:
            return p
    return None


def load_metrics(path: Path) -> dict | None:
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    if not rows:
        return None

    y_true, y_pred = [], []
    unparsed = 0
    for r in rows:
        th = r.get("true_harm_binary")
        ph = r.get("predicted_harm")
        if th is None or ph is None:
            unparsed += 1
        else:
            y_true.append(th.lower())
            y_pred.append(ph.lower())

    if not y_true:
        return None

    if HAS_SKLEARN:
        acc = accuracy_score(y_true, y_pred)
        f1  = f1_score(y_true, y_pred, pos_label="harmful", zero_division=0)
    else:
        correct = sum(t == p for t, p in zip(y_true, y_pred))
        acc = correct / len(y_true)
        tp = sum(t == "harmful" and p == "harmful" for t, p in zip(y_true, y_pred))
        fp = sum(t != "harmful" and p == "harmful" for t, p in zip(y_true, y_pred))
        fn = sum(t == "harmful" and p != "harmful" for t, p in zip(y_true, y_pred))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    return {"acc": round(acc, 4), "f1": round(f1, 4),
            "n": len(y_true), "unparsed": unparsed}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="/scratch/s4351495/intention-jailbreak/trained_models/causal",
                   help="Base directory containing dpo-* model dirs")
    args = p.parse_args()
    base = Path(args.base)

    ds_keys = DATASET_ORDER
    short  = [SHORT_NAMES[d] for d in ds_keys]

    col_w = 12
    cond_w = 26

    header = f"{'Condition':<{cond_w}}" + "".join(f"{s:>{col_w}}" for s in short)
    print("F1 (harmful class)")
    print(header)
    print("-" * len(header))

    coverage_rows = []

    for cond_name, dir_name in CONDITIONS.items():
        pred_dir = base / dir_name / "predictions"
        row_f1   = f"{cond_name:<{cond_w}}"
        row_cov  = f"{cond_name:<{cond_w}}"

        for ds_key in ds_keys:
            pred_file = find_pred_file(pred_dir, ds_key)
            if pred_file is None:
                row_f1   += f"{'—':>{col_w}}"
                row_cov  += f"{'—':>{col_w}}"
            else:
                m = load_metrics(pred_file)
                if m is None:
                    row_f1   += f"{'(empty)':>{col_w}}"
                    row_cov  += f"{'—':>{col_w}}"
                else:
                    row_f1   += f"{m['f1']:.4f}".rjust(col_w)
                    row_cov += f"{m['n']}/{m['unparsed']}".rjust(col_w)

        print(row_f1)
        coverage_rows.append(row_cov)

    print()
    print("Coverage (n/miss)")
    print(header)
    print("-" * len(header))
    for row_cov in coverage_rows:
        print(row_cov)

    print()
    print("Legend: F1 = harmful-class F1, coverage = parsed predictions/missing.")
    print("— = predictions not yet saved.")


if __name__ == "__main__":
    main()
