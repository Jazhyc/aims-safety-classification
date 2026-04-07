"""
Compare SFT, DPO, and Contrastive models on harm classification and intent quality.

Loads prediction files from each model and prints per-dataset metrics + summary tables.

Datasets evaluated (if prediction files exist):
  - Annotated Intents (test_predictions.jsonl)
  - WildGuardTest     (wildguardtest_predictions.jsonl)
  - XSTest            (xstest_predictions.jsonl)

Usage (from project root):
    python scripts/compare_models.py \\
        --sft         data/predictions/sft_baseline \\
        --dpo         trained_models/causal/llama-dpo/predictions \\
        --contrastive trained_models/causal/llama-contrastive/predictions \\
        --output-dir  data/comparison
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score, classification_report,
    f1_score, precision_score, recall_score,
)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_predictions(path: str) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def harm_metrics(records: list[dict]) -> dict:
    """Compute harm classification metrics from a prediction file."""
    valid = [
        (r["true_harm"], r["predicted_harm"])
        for r in records
        if r.get("true_harm") and r.get("predicted_harm")
    ]
    if not valid:
        return {}

    y_true = [t for t, _ in valid]
    y_pred = [p for _, p in valid]
    labels = ["harmful", "safe"]

    return {
        "n":             len(valid),
        "n_unparsed":    len(records) - len(valid),
        "accuracy":      accuracy_score(y_true, y_pred),
        "f1_macro":      f1_score(y_true, y_pred, average="macro",    labels=labels, zero_division=0),
        "f1_harmful":    f1_score(y_true, y_pred, pos_label="harmful", average="binary", zero_division=0),
        "f1_binary":     f1_score(y_true, y_pred, pos_label="harmful", average="binary", zero_division=0),
        "f1_safe":       f1_score(y_true, y_pred, pos_label="safe",    average="binary", zero_division=0),
        "prec_harmful":  precision_score(y_true, y_pred, pos_label="harmful", average="binary", zero_division=0),
        "rec_harmful":   recall_score(y_true, y_pred,    pos_label="harmful", average="binary", zero_division=0),
        "prec_safe":     precision_score(y_true, y_pred, pos_label="safe",    average="binary", zero_division=0),
        "rec_safe":      recall_score(y_true, y_pred,    pos_label="safe",    average="binary", zero_division=0),
        "report":        classification_report(y_true, y_pred, labels=labels, zero_division=0),
        "y_true":        y_true,
        "y_pred":        y_pred,
    }


def intent_metrics(records: list[dict]) -> dict:
    """
    Compute intent quality metrics against the gold intent annotation.
    Only possible when records include a 'gold_intent' field.
    """
    pairs = [
        (r["gold_intent"], r["generated_intent"])
        for r in records
        if r.get("gold_intent") and r.get("generated_intent")
    ]
    if not pairs:
        return {}

    refs  = [p[0] for p in pairs]
    preds = [p[1] for p in pairs]

    # ROUGE-L
    try:
        import evaluate as hf_evaluate
        rouge = hf_evaluate.load("rouge")
        rouge_scores = rouge.compute(
            predictions=preds, references=refs, use_aggregator=False
        )
        rougeL_mean = float(np.mean(rouge_scores["rougeL"]))
    except Exception:
        rougeL_mean = None

    # Semantic similarity
    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity as cos_sim

        embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        ref_emb  = embedder.encode(refs,  convert_to_numpy=True, show_progress_bar=False)
        pred_emb = embedder.encode(preds, convert_to_numpy=True, show_progress_bar=False)
        sim_mat  = cos_sim(ref_emb, pred_emb)
        sims     = [float(sim_mat[i, i]) for i in range(len(refs))]
        sem_sim  = float(np.mean(sims))
    except Exception:
        sem_sim = None

    return {
        "n":          len(pairs),
        "rougeL":     rougeL_mean,
        "sem_sim":    sem_sim,
    }


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------

def _bar(value: float, width: int = 20) -> str:
    filled = round(value * width)
    return "█" * filled + "░" * (width - filled)


def print_harm_report(name: str, m: dict):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    if not m:
        print("  No valid predictions.")
        return
    print(f"  n={m['n']}  unparsed={m['n_unparsed']}")
    print(f"\n  Accuracy : {m['accuracy']:.4f}  {_bar(m['accuracy'])}")
    print(f"  F1 macro : {m['f1_macro']:.4f}  {_bar(m['f1_macro'])}")
    print(f"\n  Per-class F1:")
    print(f"    harmful  P={m['prec_harmful']:.3f}  R={m['rec_harmful']:.3f}  F1={m['f1_harmful']:.3f}")
    print(f"    safe     P={m['prec_safe']:.3f}  R={m['rec_safe']:.3f}  F1={m['f1_safe']:.3f}")
    print(f"\n  Classification report:\n{m['report']}")


def print_summary_table(results: dict[str, dict], intent_results: dict[str, dict]):
    print("\n" + "=" * 72)
    print("  COMPARISON SUMMARY")
    print("=" * 72)

    # Header
    col_w = 14
    models = list(results.keys())
    header = f"{'Metric':<20}" + "".join(f"{m:>{col_w}}" for m in models)
    print(header)
    print("-" * (20 + col_w * len(models)))

    def row(label, getter, fmt=".4f"):
        vals = []
        for m in models:
            v = getter(results[m])
            vals.append(f"{v:{fmt}}" if v is not None else "  N/A  ")
        best = None
        numeric = [float(v) for v in vals if v.strip() != "N/A"]
        if numeric:
            best_v = max(numeric)
            best = [i for i, v in enumerate(vals) if v.strip() != "N/A" and abs(float(v) - best_v) < 1e-9]
        line = f"{label:<20}"
        for i, v in enumerate(vals):
            marker = " *" if best and i in best else "  "
            line += f"{v:>{col_w-2}}{marker}"
        print(line)

    row("Accuracy",      lambda m: m.get("accuracy"))
    row("F1 macro",      lambda m: m.get("f1_macro"))
    row("F1 harmful",    lambda m: m.get("f1_harmful"))
    row("F1 safe",       lambda m: m.get("f1_safe"))
    row("Prec harmful",  lambda m: m.get("prec_harmful"))
    row("Rec harmful",   lambda m: m.get("rec_harmful"))
    row("Unparsed",      lambda m: m.get("n_unparsed"), fmt="d")

    if any(intent_results.values()):
        print("-" * (20 + col_w * len(models)))
        for m_name in models:
            ir = intent_results.get(m_name, {})
            rL = ir.get("rougeL")
            ss = ir.get("sem_sim")
            rl_str = f"{rL:.4f}" if rL is not None else "  N/A  "
            ss_str = f"{ss:.4f}" if ss is not None else "  N/A  "
            results[m_name]["_rougeL"]  = rL
            results[m_name]["_sem_sim"] = ss

        row("ROUGE-L (intent)", lambda m: m.get("_rougeL"))
        row("Sem-sim (intent)", lambda m: m.get("_sem_sim"))

    print("=" * 72)
    print("  * = best value in row")
    print("=" * 72)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

DATASETS = [
    ("annotated_intents", "test_predictions.jsonl"),
    ("wildguardtest",     "wildguardtest_predictions.jsonl"),
    ("xstest",            "xstest_predictions.jsonl"),
]

DATASET_LABELS = {
    "annotated_intents": "Annotated Intents",
    "wildguardtest":     "WildGuardTest",
    "xstest":            "XSTest",
}


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--sft",         type=str, default="data/predictions/sft_baseline")
    p.add_argument("--dpo-hard",    type=str, default=None,
                   help="Predictions dir for hard-only DPO condition.")
    p.add_argument("--dpo-judge",   type=str, default=None,
                   help="Predictions dir for judge-augmented DPO condition.")
    p.add_argument("--dpo-aug",     type=str, default=None,
                   help="Predictions dir for WildGuard-augmented DPO condition.")
    p.add_argument("--output-dir",  type=str, default="data/comparison")
    p.add_argument("--no-intent",   action="store_true",
                   help="Skip intent quality metrics (faster if sentence-transformers is slow).")
    args = p.parse_args()

    candidates = [
        ("SFT baseline", args.sft),
        ("DPO-hard",     args.dpo_hard),
        ("DPO-judge",    args.dpo_judge),
        ("DPO-aug",      args.dpo_aug),
    ]
    model_dirs = {name: path for name, path in candidates if path is not None}

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # results[dataset_key][model_name] = harm_metrics dict
    all_harm    = {ds: {} for ds, _ in DATASETS}
    all_intent  = {ds: {} for ds, _ in DATASETS}

    for ds_key, filename in DATASETS:
        ds_label = DATASET_LABELS[ds_key]
        print(f"\n{'#'*60}")
        print(f"  DATASET: {ds_label}")
        print(f"{'#'*60}")

        for model_name, pred_dir in model_dirs.items():
            path = Path(pred_dir) / filename
            if not path.exists():
                print(f"[WARN] {model_name} / {ds_label}: file not found — {path}")
                all_harm[ds_key][model_name]   = {}
                all_intent[ds_key][model_name] = {}
                continue

            print(f"\nLoading {model_name} from {path}")
            records = load_predictions(path)
            print(f"  {len(records)} records")

            all_harm[ds_key][model_name] = harm_metrics(records)
            print_harm_report(model_name, all_harm[ds_key][model_name])

            if not args.no_intent:
                all_intent[ds_key][model_name] = intent_metrics(records)
                ir = all_intent[ds_key][model_name]
                if ir:
                    rL = f"{ir['rougeL']:.4f}" if ir.get("rougeL") is not None else "N/A"
                    ss = f"{ir['sem_sim']:.4f}" if ir.get("sem_sim") is not None else "N/A"
                    print(f"  ROUGE-L: {rL}   Sem-sim: {ss}   (n={ir['n']})")
            else:
                all_intent[ds_key][model_name] = {}

        if any(all_harm[ds_key].values()):
            print(f"\n--- Summary table: {ds_label} ---")
            print_summary_table(all_harm[ds_key], all_intent[ds_key])

    # ── Save summary JSON ─────────────────────────────────────────────────
    summary = {}
    for ds_key, _ in DATASETS:
        ds_label = DATASET_LABELS[ds_key]
        summary[ds_label] = {}
        for model_name in model_dirs:
            m  = all_harm[ds_key].get(model_name, {})
            ir = all_intent[ds_key].get(model_name, {})
            summary[ds_label][model_name] = {
                k: v for k, v in {
                    "accuracy":     m.get("accuracy"),
                    "f1_macro":     m.get("f1_macro"),
                    "f1_binary":    m.get("f1_binary"),
                    "f1_harmful":   m.get("f1_harmful"),
                    "f1_safe":      m.get("f1_safe"),
                    "prec_harmful": m.get("prec_harmful"),
                    "rec_harmful":  m.get("rec_harmful"),
                    "n":            m.get("n"),
                    "n_unparsed":   m.get("n_unparsed"),
                    "rougeL":       ir.get("rougeL"),
                    "sem_sim":      ir.get("sem_sim"),
                }.items() if v is not None
            }

    out_path = output_dir / "comparison_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved → {out_path}")


if __name__ == "__main__":
    main()
