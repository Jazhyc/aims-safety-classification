"""
Compare DPO sweep runs — generic, config-driven.

Run directories must contain prediction files directly or under a predictions/
subdirectory. Directory names are expected to follow the 'beta{X}_e{Y}'
convention used by submit_judge_sweep.py, but anything works — unrecognised
names are shown as-is.

Usage:
    # From a YAML config (sweep_dir + metadata come from the config):
    python scripts/compare_sweep.py --config configs/experiments/judge_sweep.yaml
    python scripts/compare_sweep.py --config configs/experiments/hard_sweep.yaml

    # Or point directly at any sweep directory:
    python scripts/compare_sweep.py --sweep-dir trained_models/causal/dpo-judge-sweep
    python scripts/compare_sweep.py --sweep-dir trained_models/causal/dpo-hard-sweep

    # Rank by a different dataset:
    python scripts/compare_sweep.py --config configs/experiments/judge_sweep.yaml \\
        --rank-by wildguardtest_predictions.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support


PRED_FILES = [
    "test_predictions.jsonl",
    "wildguardtest_predictions.jsonl",
    "xstest_predictions.jsonl",
]

DS_LABELS = {
    "test_predictions.jsonl":          "Annot",
    "wildguardtest_predictions.jsonl": "WG",
    "xstest_predictions.jsonl":        "XS",
}


def load_predictions(path: Path) -> tuple[list, list, int]:
    preds, golds = [], []
    n_unparsed = 0
    for line in path.read_text().strip().splitlines():
        r = json.loads(line)
        pred = r.get("predicted_harm") or r.get("pred_harm") or r.get("prediction")
        gold = r.get("true_harm") or r.get("gold_harm") or r.get("label")
        if pred in ("harmful", "safe") and gold in ("harmful", "safe"):
            preds.append(pred)
            golds.append(gold)
        else:
            n_unparsed += 1
    return preds, golds, n_unparsed


def compute_metrics(preds: list, golds: list) -> dict:
    labels = ["harmful", "safe"]
    f1_macro = f1_score(golds, preds, average="macro", labels=labels)
    acc = accuracy_score(golds, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(golds, preds, labels=labels)
    return {
        "f1_macro":    round(f1_macro, 4),
        "accuracy":    round(acc, 4),
        "f1_harmful":  round(f1[0], 4),
        "rec_harmful": round(rec[0], 4),
        "f1_safe":     round(f1[1], 4),
        "n":           len(preds),
    }


def parse_dir_name(name: str) -> dict:
    """Parse 'beta0.3_e2' → {beta: 0.3, epochs: 2}. Returns {} on failure."""
    try:
        parts = name.split("_")
        beta   = float(parts[0].replace("beta", ""))
        epochs = int(parts[1].replace("e", ""))
        return {"beta": beta, "epochs": epochs}
    except Exception:
        return {}


def find_pred_file(run_dir: Path, fname: str) -> Path | None:
    """Look for fname directly in run_dir, then in run_dir/predictions/."""
    for candidate in [run_dir / fname, run_dir / "predictions" / fname]:
        if candidate.exists():
            return candidate
    return None


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--config",    metavar="YAML",
                        help="Path to a sweep YAML in configs/experiments/.")
    source.add_argument("--sweep-dir", metavar="DIR",
                        help="Sweep directory to scan directly.")
    p.add_argument("--rank-by", default=None,
                   help=f"Prediction file to rank by (default: {PRED_FILES[0]}).")
    p.add_argument("--output", default=None,
                   help="Where to save results JSON (default: <sweep_dir>/sweep_results.json).")
    args = p.parse_args()

    # Resolve sweep directory and optional config metadata.
    cfg: dict = {}
    if args.config:
        cfg = yaml.safe_load(Path(args.config).read_text())
        sweep_dir = Path(cfg["sweep_dir"])
    else:
        sweep_dir = Path(args.sweep_dir)

    if not sweep_dir.exists():
        print(f"Sweep directory not found: {sweep_dir}")
        return

    rank_file = args.rank_by or PRED_FILES[0]
    out_path  = Path(args.output) if args.output else sweep_dir / "sweep_results.json"

    # Print sweep-level info when loaded from config.
    if cfg:
        print(f"\nSweep: {cfg.get('condition', '?')}  |  "
              f"betas={cfg.get('betas')}  epochs={cfg.get('epochs')}  "
              f"unbalanced={cfg.get('unbalanced', False)}  "
              f"harmful_weight={cfg.get('harmful_weight', 1.0)}")

    # Collect results for each run directory.
    rows: list[dict] = []
    for run_dir in sorted(sweep_dir.iterdir()):
        if not run_dir.is_dir():
            continue

        ds_metrics: dict[str, dict | None] = {}
        any_found = False
        for fname in PRED_FILES:
            fpath = find_pred_file(run_dir, fname)
            if fpath:
                preds, golds, n_unp = load_predictions(fpath)
                if preds:
                    m = compute_metrics(preds, golds)
                    m["n_unparsed"] = n_unp
                    ds_metrics[fname] = m
                    any_found = True
                else:
                    ds_metrics[fname] = None
            else:
                ds_metrics[fname] = None

        if not any_found:
            continue

        parsed = parse_dir_name(run_dir.name)
        rows.append({"name": run_dir.name, **parsed, "datasets": ds_metrics})

    if not rows:
        print(f"No completed runs found in {sweep_dir}")
        return

    # Sort by primary ranking dataset F1.
    rows.sort(key=lambda r: (r["datasets"].get(rank_file) or {}).get("f1_macro", 0.0),
              reverse=True)

    # Only show datasets that have at least one result.
    active_files = [f for f in PRED_FILES if any(r["datasets"].get(f) for r in rows)]

    # Table.
    has_beta   = any("beta"   in r for r in rows)
    has_epochs = any("epochs" in r for r in rows)

    header_parts = [f"{'Run':<24}"]
    if has_beta:
        header_parts.append(f"{'beta':>6}")
    if has_epochs:
        header_parts.append(f"{'ep':>3}")
    for fname in active_files:
        lbl = DS_LABELS.get(fname, fname.replace("_predictions.jsonl", ""))
        header_parts.append(f"{lbl+' F1':>9}")
    header_parts.append(f"{'Unp':>4}")
    header = "  ".join(header_parts)

    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))

    best_name = rows[0]["name"]
    for r in rows:
        parts = [f"{r['name']:<24}"]
        if has_beta:
            parts.append(f"{r.get('beta', '-'):>6}")
        if has_epochs:
            parts.append(f"{r.get('epochs', '-'):>3}")
        for fname in active_files:
            m = r["datasets"].get(fname)
            parts.append(f"{m['f1_macro']:>9.4f}" if m else f"{'—':>9}")
        primary = r["datasets"].get(rank_file)
        unp = primary["n_unparsed"] if primary else "-"
        marker = " ←best" if r["name"] == best_name else ""
        parts.append(f"{unp:>4}{marker}")
        print("  ".join(parts))

    print("=" * len(header))
    print(f"\nRanked by: {DS_LABELS.get(rank_file, rank_file)}  |  {len(rows)} runs")

    # Save — flatten datasets into top-level keys.
    save_rows = []
    for r in rows:
        row = {k: v for k, v in r.items() if k != "datasets"}
        for fname in active_files:
            m = r["datasets"].get(fname)
            if m:
                row[DS_LABELS.get(fname, fname)] = m
        save_rows.append(row)
    out_path.write_text(json.dumps(save_rows, indent=2))
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
