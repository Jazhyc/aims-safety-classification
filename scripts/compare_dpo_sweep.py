"""
Compare all DPO sweep runs in one table.

Reads test_predictions.jsonl from data/predictions/dpo_sweep/run_XX/ and
the SFT baseline from data/predictions/sft_baseline/, computes metrics,
and prints a ranked table + saves data/results/dpo_sweep_comparison.json.

Usage:
    python scripts/compare_dpo_sweep.py
    python scripts/compare_dpo_sweep.py --pred-dir data/predictions/dpo_sweep
"""

import argparse
import json
from pathlib import Path

from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support

# Run metadata — matches dpo_sweep.sh numbering
RUN_META = {
    # Phase 1
    1:  dict(phase=1, loss="sigmoid", beta=0.1, lr="5e-5", epochs=3, ls=0.0, pairs="unbal"),
    2:  dict(phase=1, loss="sigmoid", beta=0.2, lr="5e-5", epochs=3, ls=0.0, pairs="unbal"),
    3:  dict(phase=1, loss="sigmoid", beta=0.3, lr="5e-5", epochs=3, ls=0.0, pairs="unbal"),
    4:  dict(phase=1, loss="sigmoid", beta=0.5, lr="5e-5", epochs=3, ls=0.0, pairs="unbal"),
    5:  dict(phase=1, loss="ipo",     beta=0.1, lr="5e-5", epochs=3, ls=0.0, pairs="unbal"),
    6:  dict(phase=1, loss="ipo",     beta=0.2, lr="5e-5", epochs=3, ls=0.0, pairs="unbal"),
    7:  dict(phase=1, loss="ipo",     beta=0.3, lr="5e-5", epochs=3, ls=0.0, pairs="unbal"),
    8:  dict(phase=1, loss="ipo",     beta=0.5, lr="5e-5", epochs=3, ls=0.0, pairs="unbal"),
    # Phase 2
    9:  dict(phase=2, loss="sigmoid", beta=0.5, lr="1e-5", epochs=3, ls=0.0, pairs="unbal"),
    10: dict(phase=2, loss="sigmoid", beta=0.5, lr="5e-5", epochs=3, ls=0.0, pairs="unbal"),
    11: dict(phase=2, loss="sigmoid", beta=0.5, lr="5e-5", epochs=1, ls=0.0, pairs="unbal"),
    12: dict(phase=2, loss="sigmoid", beta=0.5, lr="5e-5", epochs=3, ls=0.1, pairs="unbal"),
    13: dict(phase=2, loss="sigmoid", beta=0.5, lr="5e-5", epochs=3, ls=0.2, pairs="unbal"),
    14: dict(phase=2, loss="sigmoid", beta=0.7, lr="5e-5", epochs=3, ls=0.0, pairs="unbal"),
    15: dict(phase=2, loss="sigmoid", beta=1.0, lr="5e-5", epochs=3, ls=0.0, pairs="unbal"),
    # Phase 3
    16: dict(phase=3, loss="sigmoid", beta=0.5, lr="5e-5", epochs=1, ls=0.0, pairs="bal"),
    17: dict(phase=3, loss="sigmoid", beta=0.5, lr="5e-5", epochs=1, ls=0.1, pairs="bal"),
    18: dict(phase=3, loss="ipo",     beta=0.5, lr="5e-5", epochs=1, ls=0.0, pairs="bal"),
    19: dict(phase=3, loss="ipo",     beta=0.3, lr="5e-5", epochs=1, ls=0.0, pairs="bal"),
}


def load_predictions(path: Path):
    preds, golds = [], []
    n_unparsed = 0
    for line in path.read_text().strip().splitlines():
        r = json.loads(line)
        pred = r.get("predicted_harm") or r.get("pred_harm") or r.get("prediction")
        gold = r.get("gold_harm") or r.get("true_harm") or r.get("label")
        if pred in ("harmful", "safe") and gold in ("harmful", "safe"):
            preds.append(pred)
            golds.append(gold)
        else:
            n_unparsed += 1
    return preds, golds, n_unparsed


def compute_metrics(preds, golds):
    labels = ["harmful", "safe"]
    acc = accuracy_score(golds, preds)
    f1_macro = f1_score(golds, preds, average="macro", labels=labels)
    prec, rec, f1, _ = precision_recall_fscore_support(golds, preds, labels=labels)
    return {
        "accuracy":    round(acc, 4),
        "f1_macro":    round(f1_macro, 4),
        "f1_harmful":  round(f1[0], 4),
        "rec_harmful": round(rec[0], 4),
        "prec_harmful":round(prec[0], 4),
        "f1_safe":     round(f1[1], 4),
        "n":           len(preds),
    }


def main(args):
    pred_dir = Path(args.pred_dir)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results = {}

    # SFT baseline
    sft_file = Path("data/predictions/sft_baseline/test_predictions.jsonl")
    if sft_file.exists():
        preds, golds, n_unp = load_predictions(sft_file)
        results["SFT"] = {**compute_metrics(preds, golds), "n_unparsed": n_unp,
                          "phase": 0, "loss": "-", "beta": "-", "lr": "-",
                          "epochs": "-", "ls": "-", "pairs": "-"}

    # DPO sweep runs
    for run_id, meta in sorted(RUN_META.items()):
        run_dir = pred_dir / f"run_{run_id:02d}"
        pred_file = run_dir / "test_predictions.jsonl"
        if not pred_file.exists():
            continue
        preds, golds, n_unp = load_predictions(pred_file)
        results[f"run_{run_id:02d}"] = {
            **compute_metrics(preds, golds),
            "n_unparsed": n_unp,
            **meta,
        }

    if not results:
        print("No prediction files found. Run eval_dpo_sweep.sh first.")
        return

    # Print ranked table
    sorted_runs = sorted(results.items(), key=lambda x: -x[1]["f1_macro"])

    header = f"{'Run':<12} {'Ph':>2} {'Loss':<8} {'β':>4} {'lr':<6} {'ep':>2} {'ls':>4} {'pairs':<6} │ {'F1mac':>6} {'RecH':>6} {'F1H':>6} {'F1S':>6} {'Acc':>6}"
    print("\n" + "=" * len(header))
    print(header)
    print("─" * len(header))

    for name, m in sorted_runs:
        ph   = str(m.get("phase", "-"))
        loss = str(m.get("loss",  "-"))
        beta = str(m.get("beta",  "-"))
        lr   = str(m.get("lr",    "-"))
        ep   = str(m.get("epochs","-"))
        ls   = str(m.get("ls",    "-"))
        pairs= str(m.get("pairs", "-"))
        marker = " ←best" if name == sorted_runs[0][0] else ""
        print(f"{name:<12} {ph:>2} {loss:<8} {beta:>4} {lr:<6} {ep:>2} {ls:>4} {pairs:<6} │ "
              f"{m['f1_macro']:>6.3f} {m['rec_harmful']:>6.3f} {m['f1_harmful']:>6.3f} "
              f"{m['f1_safe']:>6.3f} {m['accuracy']:>6.3f}{marker}")

    print("=" * len(header))
    print(f"\nRuns evaluated: {len(results)}  |  Missing: {len(RUN_META) + 1 - len(results)}")

    out_path.write_text(json.dumps(results, indent=2))
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pred-dir", default="data/predictions/dpo_sweep")
    p.add_argument("--output",   default="data/results/dpo_sweep_comparison.json")
    main(p.parse_args())
