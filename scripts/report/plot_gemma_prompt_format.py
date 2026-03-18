"""Generate Gemma 3 27B prompt format impact plot with clearer labels."""
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

VERSION_COLORS = {
    "Without Instructions": "#4393c3",
    "With Instructions":    "#d6604d",
}


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


def plot_comparison_single_legend(df: pd.DataFrame, title: str, filename: str,
                                  versions: tuple = ("Without Instructions", "With Instructions")) -> None:
    """Side-by-side grouped bar chart with a single shared legend at the bottom."""
    bar_w = 0.35
    x     = np.arange(len(DATASET_ORDER))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    bars_for_legend = []
    labels_for_legend = []

    for ax, condition in zip(axes, CONDITION_ORDER):
        df_c = df[df["condition"] == condition]
        for i, version in enumerate(versions):
            df_cv   = df_c[df_c["version"] == version].set_index("dataset")
            f1_vals = [df_cv.loc[ds, "f1"] if ds in df_cv.index else 0.0 for ds in DATASET_ORDER]
            offset  = (i - 0.5) * bar_w
            bars = ax.bar(x + offset, f1_vals, bar_w * 0.88, label=version,
                          color=VERSION_COLORS.get(version, "#aaaaaa"),
                          edgecolor="white", linewidth=0.5)
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
    fig.suptitle(f"{title}\nStudent: Llama-3.1-8B", fontsize=14, y=1.02)

    # Single shared legend at the bottom
    fig.legend(bars_for_legend, labels_for_legend,
               title="Prompt Format", fontsize=11, title_fontsize=11,
               loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.06),
               frameon=True)

    plt.tight_layout()
    fig.savefig(FIGS_DIR / f"{filename}.pdf", bbox_inches="tight")
    fig.savefig(FIGS_DIR / f"{filename}.png", bbox_inches="tight", dpi=150)
    print(f"Saved: {FIGS_DIR / filename}.pdf  and  .png")
    plt.close(fig)


if __name__ == "__main__":
    df_gemma = load_df({
        "RedHatAI-gemma-3-27b-it-quantized.w4a16":    "Without Instructions",
        "RedHatAI-gemma-3-27b-it-quantized.w4a16-v3": "With Instructions",
    })

    plot_comparison_single_legend(
        df_gemma,
        "Gemma 3 27B — Prompt Format Impact",
        "gemma_prompt_format_f1",
    )
