"""Qwen3-32B reasoning mode impact: two separate plots — F1 and output token counts."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score

plt.rcParams["figure.dpi"] = 120

DATA_DIR = Path("data/safety_experiment/pipeline")
FIGS_DIR = Path("data/report")

DATASETS = {
    "annotated-intents": "Annotated Intents",
    "wildguardmix":      "WildguardTest",
    "xstest":            "XSTest",
}
CONDITIONS = {
    "finetuned_reasoning_classification": "Classification",
    "finetuned_reasoning_generation":     "Generation",
}
DATASET_ORDER   = ["Annotated Intents", "WildguardTest", "XSTest", "Average"]
CONDITION_ORDER = ["Classification", "Generation"]

RAW_CONDITION_MAP = {
    "without_intent": "Classification",
    "with_intent":    "Generation",
}

VERSIONS = ("No Reasoning", "With Reasoning")
VERSION_COLORS = {
    "No Reasoning":   "#4393c3",
    "With Reasoning": "#d6604d",
}

TRACE_PATHS = {
    "No Reasoning":   Path("data/reasoning_traces/RedHatAI-Qwen3-32B-quantized.w4a16/raw_outputs.json"),
    "With Reasoning": Path("data/reasoning_traces_v3/RedHatAI-Qwen3-32B-quantized.w4a16/raw_outputs.json"),
}

SLUGS = {
    "RedHatAI-Qwen3-32B-quantized.w4a16-v2": "No Reasoning",
    "RedHatAI-Qwen3-32B-quantized.w4a16-v3": "With Reasoning",
}


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(path: Path) -> dict:
    records = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    before  = len(records)
    records = [r for r in records if r.get("predicted_harm") is not None]
    if (dropped := before - len(records)):
        print(f"  [{path.parent.parent.name}/{path.parent.name}] dropped {dropped}/{before} unparsed")
    y_true = [r["true_harm_binary"] for r in records]
    y_pred = [r["predicted_harm"]   for r in records]
    return {
        "f1":        f1_score(y_true, y_pred, pos_label="harmful"),
        "precision": precision_score(y_true, y_pred, pos_label="harmful", zero_division=0),
        "recall":    recall_score(y_true, y_pred, pos_label="harmful", zero_division=0),
    }


def load_df(versions: dict) -> pd.DataFrame:
    rows = []
    for slug, version in versions.items():
        for ds_key, ds_name in DATASETS.items():
            for cond_key, cond_name in CONDITIONS.items():
                fname = f"meta-llama_Llama-3.1-8B-Instruct_{cond_key}.jsonl"
                fpath = DATA_DIR / slug / ds_key / fname
                if not fpath.exists():
                    print(f"MISSING: {fpath}")
                    continue
                rows.append({"version": version, "condition": cond_name,
                             "dataset": ds_name, **compute_metrics(fpath)})
    df = pd.DataFrame(rows)
    avg = (df.groupby(["version", "condition"])[["f1", "precision", "recall"]]
             .mean().reset_index().assign(dataset="Average"))
    return pd.concat([df, avg], ignore_index=True)


def load_token_counts() -> pd.DataFrame:
    """Approximate token count as whitespace-split word count of raw_output."""
    rows = []
    for version, path in TRACE_PATHS.items():
        with open(path) as f:
            data = json.load(f)
        for entry in data:
            raw  = entry.get("raw_output", "")
            cond = RAW_CONDITION_MAP.get(entry.get("condition", ""), entry.get("condition", ""))
            rows.append({"version": version, "condition": cond, "tokens": len(raw.split())})
    return pd.DataFrame(rows)


# ── Plot 1: F1 comparison ─────────────────────────────────────────────────────

def plot_f1(df: pd.DataFrame, filename: str) -> None:
    bar_w = 0.35
    x     = np.arange(len(DATASET_ORDER))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    bars_for_legend, labels_for_legend = [], []

    for ax, condition in zip(axes, CONDITION_ORDER):
        df_c = df[df["condition"] == condition]
        for i, version in enumerate(VERSIONS):
            df_cv   = df_c[df_c["version"] == version].set_index("dataset")
            f1_vals = [df_cv.loc[ds, "f1"] if ds in df_cv.index else 0.0 for ds in DATASET_ORDER]
            offset  = (i - 0.5) * bar_w
            bars = ax.bar(x + offset, f1_vals, bar_w * 0.88, label=version,
                          color=VERSION_COLORS[version], edgecolor="white", linewidth=0.5)
            if condition == CONDITION_ORDER[0]:
                bars_for_legend.append(bars[0])
                labels_for_legend.append(version)
            for bar, v in zip(bars, f1_vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.007,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=8.5)
        ax.set_xticks(x)
        ax.set_xticklabels(DATASET_ORDER, fontsize=11)
        ax.set_title(condition, fontsize=13, pad=10)
        ax.set_xlabel("Evaluation Dataset", fontsize=11)
        ax.set_ylim(0, 1.08)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    axes[0].set_ylabel("F1 Score ↑", fontsize=12)
    fig.suptitle("Qwen3-32B — Reasoning Mode Impact\nStudent: Llama-3.1-8B", fontsize=14, y=1.02)
    fig.legend(bars_for_legend, labels_for_legend,
               title="Reasoning Mode", fontsize=11, title_fontsize=11,
               loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.06), frameon=True)

    plt.tight_layout()
    fig.savefig(FIGS_DIR / f"{filename}.pdf", bbox_inches="tight")
    fig.savefig(FIGS_DIR / f"{filename}.png", bbox_inches="tight", dpi=150)
    print(f"Saved: {FIGS_DIR / filename}.pdf  and  .png")
    plt.close(fig)


# ── Plot 2: Output token counts ───────────────────────────────────────────────

def plot_tokens(df: pd.DataFrame, filename: str) -> None:
    # Cap at 99th percentile of With Reasoning to keep bulk of distribution visible
    cap = int(np.percentile(df["tokens"], 99)) + 200

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    bars_for_legend, labels_for_legend = [], []

    for ax, condition in zip(axes, CONDITION_ORDER):
        df_c = df[df["condition"] == condition]
        data_by_version = [df_c[df_c["version"] == v]["tokens"].clip(upper=cap).values for v in VERSIONS]

        parts = ax.violinplot(data_by_version, positions=[0, 1],
                              showmedians=True, showextrema=False, widths=0.6)
        for pc, version in zip(parts["bodies"], VERSIONS):
            pc.set_facecolor(VERSION_COLORS[version])
            pc.set_alpha(0.7)
            pc.set_edgecolor("white")
        parts["cmedians"].set_color("black")
        parts["cmedians"].set_linewidth(1.5)

        for j, (version, vals) in enumerate(zip(VERSIONS, data_by_version)):
            mean, std = vals.mean(), vals.std()
            ax.errorbar(j, mean, yerr=std, fmt="o", color="black",
                        markersize=4, capsize=4, linewidth=1.2, zorder=5)
            ax.text(j, mean + std + cap * 0.03, f"μ={mean:.0f}",
                    ha="center", va="bottom", fontsize=9)
            if condition == CONDITION_ORDER[0]:
                bars_for_legend.append(parts["bodies"][j])
                labels_for_legend.append(version)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(VERSIONS, fontsize=11)
        ax.set_title(condition, fontsize=13, pad=10)
        ax.set_xlabel("Reasoning Mode", fontsize=11)
        ax.set_ylim(0, cap)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    axes[0].set_ylabel("Output Tokens (approx.) ↓", fontsize=12)
    fig.suptitle("Qwen3-32B — Teacher Output Token Count\nStudent: Llama-3.1-8B", fontsize=14, y=1.02)
    fig.legend(bars_for_legend, labels_for_legend,
               title="Reasoning Mode", fontsize=11, title_fontsize=11,
               loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.10), frameon=True)

    plt.tight_layout()
    fig.savefig(FIGS_DIR / f"{filename}.pdf", bbox_inches="tight")
    fig.savefig(FIGS_DIR / f"{filename}.png", bbox_inches="tight", dpi=150)
    print(f"Saved: {FIGS_DIR / filename}.pdf  and  .png")
    plt.close(fig)


if __name__ == "__main__":
    df_f1     = load_df(SLUGS)
    df_tokens = load_token_counts()

    plot_f1(df_f1, "qwen32b_reasoning_mode_f1")
    plot_tokens(df_tokens, "qwen32b_reasoning_mode_tokens")
