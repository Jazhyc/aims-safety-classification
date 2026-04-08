"""
Compare DPO-judge hyperparameter sweep runs.

Reads predictions from trained_models/causal/dpo-judge-sweep/beta*_e*/predictions/
and prints a ranked table by annotated intents F1 macro.

Usage:
    python scripts/compare_judge_sweep.py
    python scripts/compare_judge_sweep.py --sweep-dir trained_models/causal/dpo-judge-sweep
"""

import argparse
import json
from pathlib import Path

from sklearn.metrics import f1_score, accuracy_score


DATASETS = [
    ("annotated",     "test_predictions.jsonl"),
    ("wildguardtest", "wildguardtest_predictions.jsonl"),
    ("xstest",        "xstest_predictions.jsonl"),
]


def load_preds(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def metrics(preds: list[dict]) -> dict:
    valid = [(r["true_harm"], r["predicted_harm"]) for r in preds
             if r.get("true_harm") and r.get("predicted_harm")]
    if not valid:
        return {"f1": 0.0, "acc": 0.0, "n": 0, "unparsed": len(preds)}
    y_true, y_pred = zip(*valid)
    return {
        "f1":       round(f1_score(list(y_true), list(y_pred), average="macro", labels=["harmful","safe"]), 4),
        "acc":      round(accuracy_score(list(y_true), list(y_pred)), 4),
        "n":        len(valid),
        "unparsed": len(preds) - len(valid),
    }


def parse_run_name(name: str) -> tuple[float, int]:
    """Extract (beta, epochs) from 'beta0.3_e1'."""
    try:
        beta  = float(name.split("_")[0].replace("beta", ""))
        epoch = int(name.split("_")[1].replace("e", ""))
        return beta, epoch
    except Exception:
        return 0.0, 0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sweep-dir", default="trained_models/causal/dpo-judge-sweep")
    args = p.parse_args()

    sweep_dir = Path(args.sweep_dir)
    runs = sorted(sweep_dir.iterdir()) if sweep_dir.exists() else []

    rows = []
    for run_dir in runs:
        pred_dir = run_dir / "predictions"
        if not pred_dir.exists():
            continue
        beta, epochs = parse_run_name(run_dir.name)
        row = {"name": run_dir.name, "beta": beta, "epochs": epochs}
        for ds_name, fname in DATASETS:
            fpath = pred_dir / fname
            if fpath.exists():
                row[ds_name] = metrics(load_preds(fpath))
            else:
                row[ds_name] = None
        rows.append(row)

    if not rows:
        print(f"No completed runs found in {sweep_dir}")
        return

    # Sort by annotated intents F1 descending
    rows.sort(key=lambda r: r["annotated"]["f1"] if r["annotated"] else 0, reverse=True)

    # Print table
    header = f"{'Run':<22} {'Beta':>5} {'Ep':>3}  {'Annot F1':>9} {'WG F1':>7} {'XS F1':>7} {'Unp':>4}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for r in rows:
        ann = r["annotated"]["f1"]  if r["annotated"] else "-"
        wg  = r["wildguardtest"]["f1"] if r["wildguardtest"] else "-"
        xs  = r["xstest"]["f1"]    if r["xstest"] else "-"
        unp = r["annotated"]["unparsed"] if r["annotated"] else "-"
        print(f"{r['name']:<22} {r['beta']:>5} {r['epochs']:>3}  {ann:>9} {wg:>7} {xs:>7} {unp:>4}")
    print("=" * len(header))

    # Save
    out = sweep_dir / "sweep_results.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
