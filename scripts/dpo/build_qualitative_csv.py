"""Build per-dataset CSVs of rows where SFT was wrong but a DPO model was right.

For each of three datasets (annotated-intents test, wildguardmix test,
toxic-chat test), joins the prediction JSONLs from SFT, hard-DPO, and
judge-standalone on `id`, keeps rows where SFT is wrong AND at least one DPO
model is right, and writes a CSV with intent + harm columns for visual review
in Google Sheets.

Correctness semantics mirror scripts/eval_safety_classifier.compute_metrics:
strict `harmful`/`safe` equality between `predicted_harm` and `true_harm_binary`
after lowercase. Rows with missing/null values on either side are skipped.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

DATASETS = ["annotated-intents", "wildguardmix", "toxic-chat"]

DEFAULTS = {
    "sft":   "trained_models/causal/llama31-8b-sft-canonical/predictions",
    "hard":  "trained_models/causal/llama31-8b-k10-e-hard-b0.3-e3-s22/predictions_test_selected",
    "judge": "trained_models/causal/llama31-8b-k10-c-judge-gemma-standalone-b0.3-e3-s22/predictions_test_selected",
    "out":   "data/qualitative",
}

CSV_FIELDS = [
    "id", "prompt", "true_harm", "true_intent",
    "sft_pred", "sft_intent",
    "hard_pred", "hard_intent",
    "judge_pred", "judge_intent",
    "winning_condition",
]


def load_jsonl_by_id(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        rid = r.get("id")
        if rid is None:
            continue
        rows[str(rid)] = r
    return rows


def find_jsonl(root: Path, dataset: str) -> Path | None:
    ds_dir = root / dataset
    if not ds_dir.is_dir():
        return None
    candidates = sorted(ds_dir.glob("*.jsonl"))
    return candidates[0] if candidates else None


def is_correct(row: dict) -> bool | None:
    """Return True/False per eval_safety_classifier semantics, or None if undecidable."""
    t = row.get("true_harm_binary")
    p = row.get("predicted_harm")
    if t is None or p is None:
        return None
    t, p = str(t).lower(), str(p).lower()
    if t not in ("harmful", "safe") or p not in ("harmful", "safe"):
        return None
    return t == p


def winning_condition(hard_correct: bool, judge_correct: bool) -> str:
    if hard_correct and judge_correct:
        return "both"
    if hard_correct:
        return "hard"
    return "judge"  # only judge_correct (caller has already filtered)


def build_dataset_csv(dataset: str, sft_root: Path, hard_root: Path,
                      judge_root: Path, out_dir: Path) -> dict:
    sft_path = find_jsonl(sft_root, dataset)
    hard_path = find_jsonl(hard_root, dataset)
    judge_path = find_jsonl(judge_root, dataset)
    missing = [(name, p) for name, p in
               [("sft", sft_path), ("hard", hard_path), ("judge", judge_path)] if p is None]
    if missing:
        return {"dataset": dataset, "skipped": True,
                "missing": [name for name, _ in missing]}

    sft = load_jsonl_by_id(sft_path)
    hard = load_jsonl_by_id(hard_path)
    judge = load_jsonl_by_id(judge_path)

    common_ids = set(sft) & set(hard) & set(judge)

    csv_rows: list[dict] = []
    n_undec = n_sft_wrong = 0
    n_hard_only = n_judge_only = n_both = 0

    for rid in sorted(common_ids):
        s_row, h_row, j_row = sft[rid], hard[rid], judge[rid]
        sc, hc, jc = is_correct(s_row), is_correct(h_row), is_correct(j_row)
        if sc is None or hc is None or jc is None:
            n_undec += 1
            continue
        if sc:
            continue  # SFT was right; skip
        n_sft_wrong += 1
        if not (hc or jc):
            continue  # neither DPO model fixed it
        win = winning_condition(hc, jc)
        if win == "both":
            n_both += 1
        elif win == "hard":
            n_hard_only += 1
        else:
            n_judge_only += 1
        csv_rows.append({
            "id":           rid,
            "prompt":       s_row.get("prompt", ""),
            "true_harm":    s_row.get("true_harm_binary", ""),
            "true_intent":  s_row.get("true_intent", "") or "",
            "sft_pred":     s_row.get("predicted_harm", ""),
            "sft_intent":   s_row.get("generated_intent", "") or "",
            "hard_pred":    h_row.get("predicted_harm", ""),
            "hard_intent":  h_row.get("generated_intent", "") or "",
            "judge_pred":   j_row.get("predicted_harm", ""),
            "judge_intent": j_row.get("generated_intent", "") or "",
            "winning_condition": win,
        })

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"sft_vs_dpo_{dataset.replace('-', '_')}.csv"
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, quoting=csv.QUOTE_ALL)
        w.writeheader()
        w.writerows(csv_rows)

    return {
        "dataset": dataset,
        "skipped": False,
        "out_path": str(out_path),
        "n_common": len(common_ids),
        "n_undecidable": n_undec,
        "n_sft_wrong": n_sft_wrong,
        "n_kept": len(csv_rows),
        "n_hard_only": n_hard_only,
        "n_judge_only": n_judge_only,
        "n_both": n_both,
    }


def main(argv: Iterable[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sft-root", default=DEFAULTS["sft"],
                   help="SFT predictions root (parent of per-dataset dirs).")
    p.add_argument("--hard-root", default=DEFAULTS["hard"])
    p.add_argument("--judge-root", default=DEFAULTS["judge"])
    p.add_argument("--out-dir", default=DEFAULTS["out"])
    p.add_argument("--datasets", nargs="+", default=DATASETS)
    args = p.parse_args(list(argv) if argv is not None else None)

    sft_root = Path(args.sft_root)
    hard_root = Path(args.hard_root)
    judge_root = Path(args.judge_root)
    out_dir = Path(args.out_dir)

    for name, root in [("sft", sft_root), ("hard", hard_root), ("judge", judge_root)]:
        if not root.is_dir():
            print(f"[WARN] {name} root not found: {root}")

    results = []
    for ds in args.datasets:
        res = build_dataset_csv(ds, sft_root, hard_root, judge_root, out_dir)
        results.append(res)

    print()
    print(f"{'dataset':<22}{'common':<10}{'sft_wrong':<12}{'kept':<8}"
          f"{'hard':<7}{'judge':<8}{'both':<7}{'out':<60}")
    print("-" * 130)
    for r in results:
        if r["skipped"]:
            print(f"{r['dataset']:<22}SKIPPED — missing: {','.join(r['missing'])}")
            continue
        print(f"{r['dataset']:<22}{r['n_common']:<10}{r['n_sft_wrong']:<12}"
              f"{r['n_kept']:<8}{r['n_hard_only']:<7}{r['n_judge_only']:<8}"
              f"{r['n_both']:<7}{r['out_path']:<60}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
