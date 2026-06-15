#!/usr/bin/env python3
"""Quick summary table of all DPO condition results.

Usage:
    python scripts/dpo/check_dpo_results.py
    python scripts/dpo/check_dpo_results.py --base /custom/path
"""

import argparse
import json
import math
import re
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

K10_TAGS = "abcdefghij"

K10_MAIN_CONDITIONS = {
    "LE-DPO": "llama31-8b-k10-{t}-hard-b0.3-e3-s22",
    "IF-DPO": "llama31-8b-k10-{t}-judge-gemma-standalone-b0.3-e3-s22",
    "hard+IF-bad": "llama31-8b-k10-{t}-judge-gemma-bad-b0.3-e3-s22",
    "IF-DPO-bad+decent": "llama31-8b-k10-{t}-judge-gemma-decent-b0.3-e3-s22",
    "LE+IF-bad+decent": "llama31-8b-k10-{t}-hard-plus-gemma-decent-b0.3-e3-s22",
}

K10_EXT_DATASETS = [
    ("wildguardmix", "wildguard"),
    ("xstest", "xstest"),
    ("toxic_chat", "toxic"),
    ("aegis", "aegis"),
    ("openai_moderation", "openai"),
]


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


def read_best_ood(base: Path, run: str, ood_base: Path) -> dict:
    """Read OOD checkpoint-selection metrics, with saved OOD predictions fallback."""
    summary_path = base / run / "training_summary.json"
    if summary_path.exists():
        try:
            d = json.loads(summary_path.read_text())
            best = (d.get("best_checkpoint_selection") or {}).get("best") or {}
            avg = best.get("ood_avg")
            if avg is not None:
                return {
                    "checkpoint": best.get("checkpoint"),
                    "toxic_f1": _maybe_round(best.get("toxic_f1")),
                    "aegis_f1": _maybe_round(best.get("aegis_f1")),
                    "ood_avg": round(float(avg), 4),
                    "source": "summary",
                }
        except Exception:
            pass

    avg_fb, f1s = load_ood_f1(ood_base, run)
    if avg_fb is None:
        return {
            "checkpoint": None,
            "toxic_f1": None,
            "aegis_f1": None,
            "ood_avg": None,
            "source": "-",
        }
    return {
        "checkpoint": "(final)",
        "toxic_f1": f1s[0],
        "aegis_f1": f1s[1],
        "ood_avg": avg_fb,
        "source": "fallback",
    }


def _maybe_round(value: object) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value), 4)
    except (TypeError, ValueError):
        return None


def read_test_metrics(base: Path, run: str) -> dict[str, float]:
    """Read five-benchmark test F1 metrics for a run if predictions exist."""
    pred_dir = base / run / "predictions_test_selected"
    if not pred_dir.exists():
        pred_dir = base / run / "predictions"
    if not pred_dir.exists():
        return {}

    metrics: dict[str, float] = {}
    ext_f1s = []
    for ds_key, _ in K10_EXT_DATASETS:
        pred_file = find_pred_file(pred_dir, ds_key)
        if pred_file is None:
            continue
        m = load_metrics(pred_file)
        if m is None:
            continue
        metrics[ds_key] = m["f1"]
        ext_f1s.append(m["f1"])

    if len(ext_f1s) == len(K10_EXT_DATASETS):
        metrics["ext_avg"] = round(sum(ext_f1s) / len(ext_f1s), 4)
    return metrics


def summarize_values(values: list[float], total: int) -> dict[str, float | int | None]:
    if not values:
        return {
            "n": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "missing": total,
        }
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "missing": total - len(values),
    }


def format_table(headers: list[str], rows: list[list[object]], widths: list[int] | None = None) -> None:
    if widths is None:
        widths = []
        for i, header in enumerate(headers):
            width = len(str(header))
            for row in rows:
                width = max(width, len(str(row[i])))
            widths.append(width + 2)

    header = "".join(str(h).ljust(widths[i]) for i, h in enumerate(headers)).rstrip()
    print(header)
    print("-" * len(header))
    for row in rows:
        print("".join(str(v).ljust(widths[i]) for i, v in enumerate(row)).rstrip())


def _fmt_num(value: float | int | None, digits: int = 4) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "-"


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
    p.add_argument("--k10-thesis", action="store_true",
                   help="Print thesis-oriented k10 summary: main variants only, test averages, and HP diagnostics.")
    args = p.parse_args()
    base     = Path(args.base)
    ood_base = Path(args.ood_base)

    if args.k10:
        summarize_k10(base, ood_base)
        return
    if args.k10_test:
        summarize_k10_test(base)
        return
    if args.k10_thesis:
        summarize_k10_thesis(base, ood_base)
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


def summarize_k10_thesis(base: Path, ood_base: Path) -> None:
    """Print ten-pool thesis tables for the main DPO variants."""
    runs: dict[str, list[dict]] = {cond: [] for cond in K10_MAIN_CONDITIONS}

    for cond, pattern in K10_MAIN_CONDITIONS.items():
        for tag in K10_TAGS:
            run = pattern.format(t=tag)
            runs[cond].append({
                "condition": cond,
                "tag": tag,
                "run": run,
                "ood": read_best_ood(base, run, ood_base),
                "test": read_test_metrics(base, run),
            })

    print("K10 Thesis DPO Summary")
    print("=" * 120)
    print("Main variants exclude ratio sweeps. OOD validation = mean F1 over toxic_chat and aegis.")
    print("LE-DPO maps to hard-label DPO; IF-DPO maps to judge-standalone intent-filter DPO.\n")

    _print_k10_ood_per_run(runs)
    _print_k10_ood_summary(runs)
    _print_k10_test_summary(runs)
    _print_k10_best_rows(runs)
    _print_k10_hp_table(base, ood_base)


def _print_k10_ood_per_run(runs: dict[str, list[dict]]) -> None:
    print("Per-run OOD validation F1")
    print("=" * 120)
    rows = []
    for cond, cruns in runs.items():
        for r in cruns:
            ood = r["ood"]
            rows.append([
                cond,
                r["tag"],
                ood.get("checkpoint") or "-",
                _fmt_num(ood.get("toxic_f1")),
                _fmt_num(ood.get("aegis_f1")),
                _fmt_num(ood.get("ood_avg")),
                ood.get("source") or "-",
            ])
    format_table(
        ["condition", "tag", "checkpoint", "toxic", "aegis", "ood_avg", "source"],
        rows,
        widths=[22, 5, 18, 10, 10, 10, 10],
    )
    print()


def _print_k10_ood_summary(runs: dict[str, list[dict]]) -> None:
    print("Per-variant OOD validation summary across pools")
    print("=" * 120)
    rows = []
    ranking = []
    for cond, cruns in runs.items():
        vals = [float(r["ood"]["ood_avg"]) for r in cruns if r["ood"].get("ood_avg") is not None]
        summary = summarize_values(vals, len(K10_TAGS))
        valid_runs = [r for r in cruns if r["ood"].get("ood_avg") is not None]
        best_run = max(valid_runs, key=lambda r: r["ood"]["ood_avg"]) if valid_runs else None
        rows.append([
            cond,
            summary["n"],
            _fmt_num(summary["mean"]),
            _fmt_num(summary["std"]),
            _fmt_num(summary["min"]),
            _fmt_num(summary["max"]),
            best_run["tag"] if best_run else "-",
            summary["missing"],
        ])
        if summary["mean"] is not None:
            ranking.append((float(summary["mean"]), cond))
    format_table(
        ["condition", "n", "mean", "std", "min", "max", "best_tag", "missing"],
        rows,
        widths=[22, 5, 10, 10, 10, 10, 10, 8],
    )

    ranking.sort(reverse=True)
    if ranking:
        print("\nRanking by mean OOD validation avg")
        for i, (mean, cond) in enumerate(ranking, 1):
            print(f"{i:2d}. {cond:<22} mean={mean:.4f}")
    print()


def _print_k10_test_summary(runs: dict[str, list[dict]]) -> None:
    print("Per-variant five-benchmark test summary across evaluable pools")
    print("=" * 120)
    rows = []
    for cond, cruns in runs.items():
        row = [cond]
        for ds_key, _ in K10_EXT_DATASETS:
            xs = [float(r["test"][ds_key]) for r in cruns if ds_key in r["test"]]
            row.append(_fmt_num(statistics.mean(xs)) if xs else "-")
        ext = [float(r["test"]["ext_avg"]) for r in cruns if "ext_avg" in r["test"]]
        n_test = len(ext)
        row.append(_fmt_num(statistics.mean(ext)) if ext else "-")
        row.append(n_test)
        if n_test == 0:
            note = "OOD-only available"
        elif n_test < len(K10_TAGS):
            note = f"partial test ({n_test}/{len(K10_TAGS)})"
        else:
            note = "all test"
        row.append(note)
        rows.append(row)
    format_table(
        ["condition", *[name for _, name in K10_EXT_DATASETS], "ext-avg", "n_test", "note"],
        rows,
        widths=[22, 11, 10, 10, 10, 10, 10, 8, 22],
    )

    ext_rows = []
    for cond, cruns in runs.items():
        ext = [float(r["test"]["ext_avg"]) for r in cruns if "ext_avg" in r["test"]]
        if not ext:
            continue
        summary = summarize_values(ext, len(K10_TAGS))
        best = max((r for r in cruns if "ext_avg" in r["test"]), key=lambda r: r["test"]["ext_avg"])
        ext_rows.append([
            cond,
            summary["n"],
            _fmt_num(summary["mean"]),
            _fmt_num(summary["std"]),
            _fmt_num(summary["min"]),
            _fmt_num(summary["max"]),
            best["tag"],
            summary["missing"],
        ])
    if ext_rows:
        print("\nTest ext-avg variance summary")
        format_table(
            ["condition", "n", "mean", "std", "min", "max", "best_tag", "missing"],
            ext_rows,
            widths=[22, 5, 10, 10, 10, 10, 10, 8],
        )
    print()


def _print_k10_best_rows(runs: dict[str, list[dict]]) -> None:
    print("Best rows by OOD validation selection")
    print("=" * 120)
    rows = []
    for cond, cruns in runs.items():
        valid = [r for r in cruns if r["ood"].get("ood_avg") is not None]
        if not valid:
            rows.append([cond, "-", "-", "-", "-", "-", "-", "-", "-", "-"])
            continue
        best = max(valid, key=lambda r: r["ood"]["ood_avg"])
        test = best["test"]
        rows.append([
            cond,
            best["tag"],
            best["ood"].get("checkpoint") or "-",
            _fmt_num(best["ood"].get("ood_avg")),
            _fmt_num(test.get("ext_avg")),
            *[_fmt_num(test.get(ds_key)) for ds_key, _ in K10_EXT_DATASETS],
        ])
    format_table(
        ["condition", "tag", "checkpoint", "ood_avg", "test_avg", *[name for _, name in K10_EXT_DATASETS]],
        rows,
        widths=[22, 5, 18, 10, 10, 11, 10, 10, 10, 10],
    )
    print()


def _print_k10_hp_table(base: Path, ood_base: Path) -> None:
    hp_runs = _discover_hp_runs(base)
    print("Hyperparameter probe runs")
    print("=" * 120)
    if not hp_runs:
        print("No k10 hyperparameter probe directories found under --base.")
        print()
        return

    rows = []
    for item in hp_runs:
        ood = read_best_ood(base, item["run"], ood_base)
        test = read_test_metrics(base, item["run"])
        rows.append([
            item["family"],
            item["tag"],
            item["beta"] or "-",
            item["lr"] or "-",
            item["run"],
            ood.get("checkpoint") or "-",
            _fmt_num(ood.get("ood_avg")),
            _fmt_num(test.get("ext_avg")),
        ])
    format_table(
        ["family", "tag", "beta", "lr", "run", "checkpoint", "ood_avg", "test_avg"],
        rows,
        widths=[18, 5, 8, 10, 62, 18, 10, 10],
    )
    print()


def _discover_hp_runs(base: Path) -> list[dict[str, str | None]]:
    items = []
    pat = re.compile(
        r"^llama31-8b-k10-(?P<tag>[a-j])-(?P<family>.+?)-hp-"
        r"(?P<params>.+?)-s\d+$"
    )
    for p in sorted(base.glob("llama31-8b-k10-*-*-hp-*")):
        if not p.is_dir() or p.name.endswith("_adapter"):
            continue
        m = pat.match(p.name)
        if not m:
            continue
        params = m.group("params")
        items.append({
            "tag": m.group("tag"),
            "family": m.group("family"),
            "beta": _extract_param(params, r"b([0-9.]+)"),
            "lr": _extract_param(params, r"lr([^-]+)"),
            "run": p.name,
        })
    return items


def _extract_param(text: str, pattern: str) -> str | None:
    m = re.search(pattern, text)
    return m.group(1) if m else None


def summarize_k10(base: Path, ood_base: Path) -> None:
    tags = "abcdefghij"
    conds = {
        "hard-dpo": "llama31-8b-k10-{t}-hard-b0.3-e3-s22",
        "judge-gemma-bad": "llama31-8b-k10-{t}-judge-gemma-bad-b0.3-e3-s22",
        "judge-gemma-standalone": "llama31-8b-k10-{t}-judge-gemma-standalone-b0.3-e3-s22",
        "judge-gemma-decent": "llama31-8b-k10-{t}-judge-gemma-decent-b0.3-e3-s22",
        "hard-plus-gemma-decent": "llama31-8b-k10-{t}-hard-plus-gemma-decent-b0.3-e3-s22",
    }
    for r in ("0.00", "0.25", "0.50", "0.75", "1.00"):
        conds[f"ratio-{r}"] = f"llama31-8b-k10-{{t}}-ratio-r{r}-b0.3-e3-s22"

    rows = {c: [] for c in conds}
    print("Per-run best checkpoint (OOD)")
    print("=" * 120)
    print(f"{'condition':28s} {'tag':>3s} {'checkpoint':14s} {'ood_avg':>10s} {'source':>12s}")
    print("-" * 120)

    for cond, pat in conds.items():
        for t in tags:
            run = pat.format(t=t)
            p = base / run / "training_summary.json"
            if not p.exists():
                # fallback: infer OOD avg from saved OOD eval outputs if present
                avg_fb, _ = load_ood_f1(ood_base, run)
                if avg_fb is None:
                    rows[cond].append((t, None, float("nan")))
                    print(f"{cond:28s} {t:>3s} {'MISSING':14s} {'nan':>10s} {'-':>12s}")
                else:
                    rows[cond].append((t, None, avg_fb))
                    print(f"{cond:28s} {t:>3s} {'(final)':14s} {avg_fb:10.4f} {'fallback':>12s}")
                continue
            try:
                d = json.loads(p.read_text())
            except Exception:
                avg_fb, _ = load_ood_f1(ood_base, run)
                if avg_fb is None:
                    rows[cond].append((t, None, float("nan")))
                    print(f"{cond:28s} {t:>3s} {'BROKEN':14s} {'nan':>10s} {'-':>12s}")
                else:
                    rows[cond].append((t, None, avg_fb))
                    print(f"{cond:28s} {t:>3s} {'(final)':14s} {avg_fb:10.4f} {'fallback':>12s}")
                continue

            best = (d.get("best_checkpoint_selection") or {}).get("best") or {}
            ck = best.get("checkpoint")
            avg = best.get("ood_avg")
            if avg is not None:
                avg = float(avg)
                rows[cond].append((t, ck, avg))
                print(f"{cond:28s} {t:>3s} {str(ck):14s} {avg:>10.4f} {'summary':>12s}")
            else:
                # ratio runs often have no best-checkpoint metadata; use OOD outputs if available
                avg_fb, _ = load_ood_f1(ood_base, run)
                if avg_fb is None:
                    rows[cond].append((t, ck, float("nan")))
                    print(f"{cond:28s} {t:>3s} {str(ck):14s} {'nan':>10s} {'none':>12s}")
                else:
                    rows[cond].append((t, ck, avg_fb))
                    print(f"{cond:28s} {t:>3s} {'(final)':14s} {avg_fb:>10.4f} {'fallback':>12s}")

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
                    ext_cell = f"{sel_m['ext_avg']:.4f}[{sel_t}]"
                    row += f"{ext_cell:>{col_w}}"
            row += f"{n_runs:>6d}"
            print(row)

    print_table("K10 Test Results — Average Table", "mean")
    print_table("K10 Test Results — Min Table (run chosen by OOD validation)", "min")
    print_table("K10 Test Results — Max Table (run chosen by OOD validation)", "max")


if __name__ == "__main__":
    main()
