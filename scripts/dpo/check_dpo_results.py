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
    # Hard mislabel conditions (Llama-3.1-8B + SFT adapter)
    "hard (llama8b, b0.5, e2, s22)":          "dpo-hard",
    "hard (llama8b, b0.5, e1, s22)":          "dpo-hard-ep1",
    "hard (llama8b, b0.3, e2, s22)":          "dpo-hard-beta0.3",
    "hard (llama8b, b0.3, e1, s22)":          "dpo-hard-beta0.3-ep1",

    # Judge = Gemma-3-27B
    "judge=gemma bad-only (llama8b, b0.5, e2)":      "dpo-judge-standalone",
    "judge=gemma bad-only (llama8b, b0.3, e2)":      "dpo-judge-standalone-beta0.3",
    "judge=gemma bad-only (llama8b, b0.5, e1)":      "dpo-judge-standalone-ep1",
    "hard+judge=gemma bad (llama8b, b0.5, e2)":      "dpo-judge",
    "hard+judge=gemma bad (llama8b, b0.1, e2)":      "dpo-judge-beta0.1",
    "hard+judge=gemma bad (llama8b, b0.3, e2)":      "dpo-judge-beta0.3",
    "hard+judge=gemma bad (llama8b, b0.5, e1)":      "dpo-judge-ep1",
    "judge=gemma bad+decent (llama8b, b0.5, e2)":    "dpo-judge-decent",
    "judge=gemma bad+decent (llama8b, b0.1, e2)":    "dpo-judge-decent-beta0.1",
    "judge=gemma bad+decent (llama8b, b0.3, e2)":    "dpo-judge-decent-beta0.3",
    "judge=gemma bad+decent (llama8b, b0.5, e1)":    "dpo-judge-decent-ep1",
    "hard+judge=gemma bad+decent (llama8b, b0.5, e2)": "dpo-union-decent",

    # Judge = GPT-OSS

    # Ratio sweep (hard + X% judge bad), beta=0.3 family
    "ratio=0.00 (llama8b, b0.3, e2)":           "llama31-8b-ratio-r0.00-b0.3-e2-s22",
    "ratio=0.25 (llama8b, b0.3, e2)":           "llama31-8b-ratio-r0.25-b0.3-e2-s22",
    "ratio=0.50 (llama8b, b0.3, e2)":           "llama31-8b-ratio-r0.50-b0.3-e2-s22",
    "ratio=0.75 (llama8b, b0.3, e2)":           "llama31-8b-ratio-r0.75-b0.3-e2-s22",
    "ratio=1.00 (llama8b, b0.3, e2)":           "llama31-8b-ratio-r1.00-b0.3-e2-s22",

    # Ratio sweep legacy beta=0.5 pilots

    # Curriculum
    "curriculum p1=hard p2=judge (llama8b, b0.3, e2+e1)": "llama31-8b-curriculum-b0.3-p1e2-p2e1-s22",

    # Gemma 12B baselines / DPO
    "gemma12b-sft (Jazhyc SFT baseline)":             "gemma-sft-baseline",
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
GEMMA_MODEL_SLUG = "google_gemma-3-12b-it_finetuned_generation.jsonl"


def find_pred_file(pred_dir: Path, ds_key: str,
                   extra_slugs: list[str] | None = None) -> Path | None:
    """Find prediction file, tolerating hyphen/underscore variants in dir name.

    Checks MODEL_SLUG (Llama) first, then GEMMA_MODEL_SLUG, then any extra_slugs.
    """
    if not pred_dir.exists():
        return None
    slugs = [MODEL_SLUG, GEMMA_MODEL_SLUG] + (extra_slugs or [])
    for variant in [ds_key, ds_key.replace("_", "-")]:
        for slug in slugs:
            p = pred_dir / variant / slug
            if p.exists():
                return p
    # Fallback: glob for any subdir containing the file (Llama slug only for backward compat)
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
    print("OOD Validation F1")
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
