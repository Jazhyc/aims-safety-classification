#!/usr/bin/env python3
"""Quick summary table of all DPO condition results.

Usage:
    python scripts/dpo/check_dpo_results.py
    python scripts/dpo/check_dpo_results.py --base /custom/path
"""

import argparse
import json
import math
import statistics
from pathlib import Path

try:
    from sklearn.metrics import f1_score, accuracy_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


CONDITIONS = {
    # Hard mislabel conditions (Llama-3.1-8B + SFT adapter)
    "hard (llama8b, b0.3, e2, s22)":          "llama31-8b-hard-b0.3-e2-s22",

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
    "gemma12b-student (GPT-OSS distilled baseline)":  "gemma-student-baseline",
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
GEMMA_MODEL_SLUG_CLF = "google_gemma-3-12b-it_finetuned_classification.jsonl"


def find_pred_file(pred_dir: Path, ds_key: str,
                   extra_slugs: list[str] | None = None) -> Path | None:
    """Find prediction file, tolerating hyphen/underscore variants in dir name.

    Checks MODEL_SLUG (Llama) first, then GEMMA_MODEL_SLUG, then any extra_slugs.
    """
    if not pred_dir.exists():
        return None
    slugs = [MODEL_SLUG, GEMMA_MODEL_SLUG, GEMMA_MODEL_SLUG_CLF] + (extra_slugs or [])
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
    p.add_argument("--k10", action="store_true",
                   help="Print k10 (a..j) condition summary/ranking from training_summary.json best checkpoint OOD avg.")
    p.add_argument("--k10-test", action="store_true",
                   help="Print k10 test-set tables across runs: mean/min/max per condition.")
    args = p.parse_args()
    base     = Path(args.base)
    ood_base = Path(args.ood_base)

    if args.k10:
        summarize_k10(base)
        return
    if args.k10_test:
        summarize_k10_test(base)
        return

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


def summarize_k10(base: Path) -> None:
    tags = "abcdefghij"
    conds = {
        "hard-dpo": "llama31-8b-k10-{t}-hard-b0.3-e3-s22",
        "judge-gemma-bad": "llama31-8b-k10-{t}-judge-gemma-bad-b0.3-e3-s22",
        "judge-gemma-standalone": "llama31-8b-k10-{t}-judge-gemma-standalone-b0.3-e3-s22",
        "judge-gemma-decent": "llama31-8b-k10-{t}-judge-gemma-decent-b0.3-e3-s22",
        "hard-plus-gemma-decent": "llama31-8b-k10-{t}-hard-plus-gemma-decent-b0.3-e3-s22",
        "curriculum": "llama31-8b-k10-{t}-curriculum-b0.3-p1e2-p2e1-s22",
    }
    for r in ("0.00", "0.25", "0.50", "0.75", "1.00"):
        conds[f"ratio-{r}"] = f"llama31-8b-k10-{{t}}-ratio-r{r}-b0.3-e3-s22"

    rows = {c: [] for c in conds}
    print("Per-run best checkpoint (OOD)")
    print("=" * 120)
    print(f"{'condition':28s} {'tag':>3s} {'checkpoint':14s} {'ood_avg':>10s}")
    print("-" * 120)

    for cond, pat in conds.items():
        for t in tags:
            p = base / pat.format(t=t) / "training_summary.json"
            if not p.exists():
                rows[cond].append((t, None, float("nan")))
                print(f"{cond:28s} {t:>3s} {'MISSING':14s} {'nan':>10s}")
                continue
            try:
                d = json.loads(p.read_text())
            except Exception:
                rows[cond].append((t, None, float("nan")))
                print(f"{cond:28s} {t:>3s} {'BROKEN':14s} {'nan':>10s}")
                continue

            best = (d.get("best_checkpoint_selection") or {}).get("best") or {}
            ck = best.get("checkpoint")
            avg = best.get("ood_avg")
            avg = float(avg) if avg is not None else float("nan")
            rows[cond].append((t, ck, avg))
            avg_s = f"{avg:.4f}" if not math.isnan(avg) else "nan"
            print(f"{cond:28s} {t:>3s} {str(ck):14s} {avg_s:>10s}")

    print("\nCondition summary (best-checkpoint OOD avg across runs)")
    print("=" * 120)
    print(f"{'condition':28s} {'n':>3s} {'mean':>10s} {'std':>10s} {'min':>10s} {'max':>10s} {'missing':>8s}")

    ranking = []
    for cond in conds:
        vals = [v for _, _, v in rows[cond] if not math.isnan(v)]
        missing = len(tags) - len(vals)
        if not vals:
            print(f"{cond:28s} {0:3d} {'nan':>10s} {'nan':>10s} {'nan':>10s} {'nan':>10s} {missing:8d}")
            continue
        mean = statistics.mean(vals)
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        vmin = min(vals)
        vmax = max(vals)
        print(f"{cond:28s} {len(vals):3d} {mean:10.4f} {std:10.4f} {vmin:10.4f} {vmax:10.4f} {missing:8d}")
        ranking.append((mean, cond))

    ranking.sort(reverse=True)
    print("\nRanking by mean OOD avg")
    print("=" * 120)
    for i, (m, cond) in enumerate(ranking, 1):
        print(f"{i:2d}. {cond:28s} mean={m:.4f}")


def summarize_k10_test(base: Path) -> None:
    tags = "abcdefghij"
    conds = {
        "hard-dpo": "llama31-8b-k10-{t}-hard-b0.3-e3-s22",
        "judge-gemma-bad": "llama31-8b-k10-{t}-judge-gemma-bad-b0.3-e3-s22",
        "judge-gemma-standalone": "llama31-8b-k10-{t}-judge-gemma-standalone-b0.3-e3-s22",
        "judge-gemma-decent": "llama31-8b-k10-{t}-judge-gemma-decent-b0.3-e3-s22",
        "hard-plus-gemma-decent": "llama31-8b-k10-{t}-hard-plus-gemma-decent-b0.3-e3-s22",
        "curriculum": "llama31-8b-k10-{t}-curriculum-b0.3-p1e2-p2e1-s22",
    }
    for r in ("0.00", "0.25", "0.50", "0.75", "1.00"):
        conds[f"ratio-{r}"] = f"llama31-8b-k10-{{t}}-ratio-r{r}-b0.3-e3-s22"

    ds = [
        ("annotated_intents", "ann-test"),
        ("annotated_intents_val", "ann-val"),
        ("wildguardmix", "wildguard"),
        ("xstest", "xstest"),
        ("toxic_chat", "toxic"),
        ("aegis", "aegis"),
        ("openai_moderation", "openai"),
    ]
    ext_keys = {"wildguardmix", "xstest", "toxic_chat", "aegis", "openai_moderation"}

    metric_keys = [k for k, _ in ds]
    # condition -> list[(tag, {metric_key: f1, ext_avg: f1})]
    rows: dict[str, list[tuple[str, dict[str, float]]]] = {c: [] for c in conds}
    # condition -> tag -> ood_avg (from training_summary best checkpoint selection)
    ood_by_tag: dict[str, dict[str, float]] = {c: {} for c in conds}

    for cond, pat in conds.items():
        for t in tags:
            run = pat.format(t=t)
            summary_path = base / run / "training_summary.json"
            if summary_path.exists():
                try:
                    d = json.loads(summary_path.read_text())
                    best = (d.get("best_checkpoint_selection") or {}).get("best") or {}
                    ood = best.get("ood_avg")
                    if ood is not None:
                        ood_by_tag[cond][t] = float(ood)
                except Exception:
                    pass

            pred_dir = base / run / "predictions_test_selected"
            if not pred_dir.exists():
                pred_dir = base / run / "predictions"
            if not pred_dir.exists():
                continue

            mrow: dict[str, float] = {}
            ext_f1s = []
            for ds_key, _ in ds:
                pred_file = find_pred_file(pred_dir, ds_key)
                if pred_file is None:
                    continue
                m = load_metrics(pred_file)
                if m is None:
                    continue
                mrow[ds_key] = m["f1"]
                if ds_key in ext_keys:
                    ext_f1s.append(m["f1"])

            if len(ext_f1s) == len(ext_keys):
                mrow["ext_avg"] = round(sum(ext_f1s) / len(ext_f1s), 4)
            if mrow:
                rows[cond].append((t, mrow))

    def print_table(title: str, reducer: str) -> None:
        print(f"\n{title}")
        col_w = 16 if reducer in {"min", "max"} else 12
        header = f"{'Condition':<28}" + "".join(f"{name:>{col_w}}" for _, name in ds) + f"{'ext-avg':>{col_w}}{'n':>6}"
        print(header)
        print("-" * len(header))
        for cond in conds:
            row = f"{cond:<28}"
            cruns = rows[cond]
            n_runs = len(cruns)

            if reducer == "mean":
                for ds_key, _ in ds:
                    xs = [m.get(ds_key) for _, m in cruns if ds_key in m]
                    if not xs:
                        row += f"{'—':>{col_w}}"
                    else:
                        row += f"{statistics.mean(xs):>{col_w}.4f}"
                ex = [m.get("ext_avg") for _, m in cruns if "ext_avg" in m]
                row += f"{statistics.mean(ex):>{col_w}.4f}" if ex else f"{'—':>{col_w}}"
            else:
                ex_rows = [(t, m) for t, m in cruns if "ext_avg" in m]
                if not ex_rows:
                    row += "".join(f"{'—':>{col_w}}" for _ in ds)
                    row += f"{'—':>{col_w}}"
                else:
                    # Select min/max run by OOD validation average when available.
                    # Fallback to test ext_avg if OOD is missing.
                    scored = []
                    for t, m in ex_rows:
                        ood = ood_by_tag[cond].get(t)
                        score = ood if ood is not None else m["ext_avg"]
                        scored.append((t, m, score))
                    if reducer == "min":
                        sel_t, sel_m, _ = min(scored, key=lambda x: x[2])
                    else:
                        sel_t, sel_m, _ = max(scored, key=lambda x: x[2])
                    for ds_key, _ in ds:
                        v = sel_m.get(ds_key)
                        if v is None:
                            row += f"{'—':>{col_w}}"
                        else:
                            row += f"{f'{v:.4f}[{sel_t}]':>{col_w}}"
                    row += f"{f'{sel_m['ext_avg']:.4f}[{sel_t}]':>{col_w}}"
            row += f"{n_runs:>6d}"
            print(row)

    print_table("K10 Test Results — Average Table", "mean")
    print_table("K10 Test Results — Min Table (run chosen by OOD validation)", "min")
    print_table("K10 Test Results — Max Table (run chosen by OOD validation)", "max")


if __name__ == "__main__":
    main()
