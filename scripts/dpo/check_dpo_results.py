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
    "hard_dpo-beta0.3-ep1":   "dpo-hard-beta0.3-ep1",
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

# Datasets included in the external average (excludes annotated-intents — seen during SFT training)
EXT_AVG_DATASETS = {"wildguardmix", "xstest", "toxic_chat", "aegis", "openai_moderation"}

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
    rows = []
    for l in path.read_text().splitlines():
        if not l.strip():
            continue
        try:
            rows.append(json.loads(l))
        except json.JSONDecodeError:
            pass
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


OOD_DATASETS = ["toxic_chat", "aegis"]


def load_ood_f1(ood_base: Path, dir_name: str) -> tuple[float, list] | tuple[None, None]:
    """Return (avg_ood_f1, [f1s]) or (None, None) if not available.

    Checks two layouts:
      - nested:  ood_base / dir_name / dir_name_adapter / dataset   (normal loop)
      - flat:    ood_base / dir_name_adapter / dataset              (direct submit)
    """
    adapter_slug = f"{dir_name}_adapter"
    candidate_dirs = [
        ood_base / dir_name / adapter_slug,  # nested (standard)
        ood_base / adapter_slug,             # flat (direct submit without subdir)
    ]
    f1s = []
    for ds in OOD_DATASETS:
        found = None
        for candidate in candidate_dirs:
            pred_file = find_pred_file(candidate, ds)
            if pred_file is not None:
                found = pred_file
                break
        if found is None:
            return None, None
        m = load_metrics(found)
        if m is None:
            return None, None
        f1s.append(m["f1"])
    avg = round(sum(f1s) / len(f1s), 4)
    return avg, f1s


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="/scratch/s4351495/intention-jailbreak/trained_models/causal",
                   help="Base directory containing dpo-* model dirs (test predictions)")
    p.add_argument("--ood-base", default="data/safety_experiment/ood_validation/dpo",
                   help="Base directory for OOD validation predictions")
    args = p.parse_args()
    base     = Path(args.base)
    ood_base = Path(args.ood_base)

    ds_keys = DATASET_ORDER
    short   = [SHORT_NAMES[d] for d in ds_keys]

    col_w  = 12
    cond_w = 26

    # ---- OOD validation table ----
    ood_header = f"{'Condition':<{cond_w}}{'toxic-F1':>12}{'aegis-F1':>12}{'avg-OOD':>12}  {'*'}"
    print("OOD Validation F1  (model selection — do NOT use for reporting)")
    print(ood_header)
    print("-" * (cond_w + 38))

    ood_avgs = {}
    for cond_name, dir_name in CONDITIONS.items():
        avg, f1s = load_ood_f1(ood_base, dir_name)
        if avg is None:
            ood_avgs[cond_name] = None
            print(f"{cond_name:<{cond_w}}{'—':>12}{'—':>12}{'—':>12}")
        else:
            ood_avgs[cond_name] = avg
            print(f"{cond_name:<{cond_w}}{f1s[0]:>12.4f}{f1s[1]:>12.4f}{avg:>12.4f}")

    valid = {k: v for k, v in ood_avgs.items() if v is not None}
    if valid:
        best = max(valid, key=lambda k: valid[k])
        print(f"\n  Best by OOD avg: {best}  ({valid[best]:.4f})")

    # ---- Test F1 table ----
    print()
    header = f"{'Condition':<{cond_w}}" + "".join(f"{s:>{col_w}}" for s in short) + f"{'ext-avg':>{col_w}}"
    print("Test F1 (report only for OOD-selected model)")
    print(header)
    print("-" * len(header))

    coverage_rows = []
    for cond_name, dir_name in CONDITIONS.items():
        pred_dir = base / dir_name / "predictions"
        row_f1  = f"{cond_name:<{cond_w}}"
        row_cov = f"{cond_name:<{cond_w}}"
        ext_f1s = []

        for ds_key in ds_keys:
            pred_file = find_pred_file(pred_dir, ds_key)
            if pred_file is None:
                row_f1  += f"{'—':>{col_w}}"
                row_cov += f"{'—':>{col_w}}"
            else:
                m = load_metrics(pred_file)
                if m is None:
                    row_f1  += f"{'(empty)':>{col_w}}"
                    row_cov += f"{'—':>{col_w}}"
                else:
                    row_f1  += f"{m['f1']:.4f}".rjust(col_w)
                    row_cov += f"{m['n']}/{m['unparsed']}".rjust(col_w)
                    if ds_key in EXT_AVG_DATASETS:
                        ext_f1s.append(m["f1"])

        if len(ext_f1s) == len(EXT_AVG_DATASETS):
            row_f1 += f"{sum(ext_f1s)/len(ext_f1s):>{col_w}.4f}"
        else:
            row_f1 += f"{'—':>{col_w}}"

        print(row_f1)
        coverage_rows.append(row_cov)

    print()
    print("Coverage (n/unparsed)")
    print(header)
    print("-" * len(header))
    for row_cov in coverage_rows:
        print(row_cov)

    print()
    print("Legend: — = predictions not yet saved.")


if __name__ == "__main__":
    main()
