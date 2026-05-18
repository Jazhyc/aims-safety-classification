#!/usr/bin/env python3
"""
Generate the LaTeX benchmark comparison table (``report/latex/benchmark_table.tex``).

Models are organised into ordered groups (zero-shot LLMs, guardrails, SFT, distillation).
Each group renders as a ``\\multicolumn`` sub-header followed by its rows; best/2nd-best
per column are bolded/underlined across the whole table.

Usage:
    python scripts/report/generate_benchmark_table.py
    python scripts/report/generate_benchmark_table.py --output report/latex/benchmark_table.tex
"""

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CACHE_DIR = _PROJECT_ROOT / ".cache/benchmark_results"
CACHE_EXPIRY = 3600  # 1 hour in seconds


# ─── Data shape ───────────────────────────────────────────────────────────────
@dataclass
class Row:
    """One table row: display name + optional method annotation + scores per dataset."""
    name: str
    scores: Dict[str, float]
    note: Optional[str] = None  # e.g. "CoT Classification" — rendered in small parens after name


@dataclass
class Group:
    """A labelled group of rows rendered with a \\multicolumn sub-header."""
    header: str               # raw LaTeX shown in the group sub-header
    rows: List[Row] = field(default_factory=list)


# ─── W&B cache plumbing (currently unused — see TODO in main) ─────────────────
def load_cache(cache_key: str) -> Optional[Dict]:
    cache_file = CACHE_DIR / f"{cache_key}.json"
    if not cache_file.exists():
        return None
    try:
        with open(cache_file, "r") as f:
            data = json.load(f)
        if time.time() - data.get("timestamp", 0) > CACHE_EXPIRY:
            return None
        return data.get("results")
    except Exception:
        return None


def save_cache(cache_key: str, results: Dict):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with open(CACHE_DIR / f"{cache_key}.json", "w") as f:
            json.dump({"timestamp": time.time(), "results": results}, f, indent=2)
    except Exception as e:
        print(f"Warning: Could not save cache: {e}")


# ─── Source of truth: hardcoded results ───────────────────────────────────────
def get_hardcoded_groups() -> List[Group]:
    """All benchmark numbers + grouping. Edit here to change the table."""
    return [
        Group(
            # Prompting-condition provenance (kept as comments; not rendered in the table):
            #   Llama-3.1-8B   → CoT Generation (better of CoT Cls/Gen by benchmark avg).
            #                    CoT Classification baseline: WG 0.803, XS 0.876, AEGIS 0.801,
            #                    TC 0.492, OAI 0.740 (avg 0.742) — dropped from the table.
            #   Gemma-3-12B    → Vanilla Generation was best in the per-condition sweep.
            #   GPT-OSS-120B   → CoT Classification was best.
            #   Claude / GPT-5 → Vanilla Generation only (no per-condition sweep due to API cost).
            header=r"\textit{Zero-shot LLMs}",
            rows=[
                Row("Llama-3.1-8B", scores={
                    "wildguardmix": 0.762, "xstest": 0.904, "aegis": 0.800,
                    "toxic_chat": 0.516, "openai_moderation": 0.761,
                }),
                Row("Gemma-3-12B", scores={
                    "wildguardmix": 0.853, "xstest": 0.902, "aegis": 0.820,
                    "toxic_chat": 0.644, "openai_moderation": 0.793,
                }),
                Row("GPT-OSS-120B", scores={
                    "wildguardmix": 0.884, "xstest": 0.911, "aegis": 0.806,
                    "toxic_chat": 0.641, "openai_moderation": 0.775,
                }),
                Row("Claude Sonnet 4.6", scores={
                    "wildguardmix": 0.838, "xstest": 0.860, "aegis": 0.762,
                    "toxic_chat": 0.667, "openai_moderation": 0.785,
                }),
                Row("GPT-5.4", scores={
                    "wildguardmix": 0.880, "xstest": 0.920, "aegis": 0.809,
                    "toxic_chat": 0.676, "openai_moderation": 0.791,
                }),
            ],
        ),
        Group(
            header=r"\textit{Dedicated Safety Guardrails}",
            rows=[
                Row("LlamaGuard 4", scores={
                    "wildguardmix": 0.738, "xstest": 0.836, "aegis": 0.705,
                    "toxic_chat": 0.441, "openai_moderation": 0.736,
                }),
                Row("ShieldGemma 27B", scores={
                    "wildguardmix": 0.5123, "xstest": 0.8226, "aegis": 0.6943,
                    "toxic_chat": 0.7032, "openai_moderation": 0.8136,
                }),
                Row("WildGuard", scores={
                    "wildguardmix": 0.888, "xstest": 0.945, "aegis": 0.809,
                    "toxic_chat": 0.652, "openai_moderation": 0.724,
                }),
                Row("GuardReasoner 8B", scores={
                    "wildguardmix": 0.8881, "xstest": 0.9187, "aegis": 0.8296,
                    "toxic_chat": 0.6813, "openai_moderation": 0.7040,
                }),
                Row("GPT-OSS-Safeguard 120B", scores={
                    "wildguardmix": 0.871, "xstest": 0.944, "aegis": 0.797,
                    "toxic_chat": 0.643, "openai_moderation": 0.780,
                }),
                Row("Nemotron Safety 4B", scores={
                    "wildguardmix": 0.8515, "xstest": 0.8510, "aegis": 0.8597,
                    "toxic_chat": 0.7333, "openai_moderation": 0.7472,
                }),
            ],
        ),
        Group(
            header=r"\textit{Ours --- SFT on Annotated Intents}",
            rows=[
                Row("Llama-3.1-8B", scores={
                    "wildguardmix": 0.856, "xstest": 0.908, "aegis": 0.803,
                    "toxic_chat": 0.664, "openai_moderation": 0.728,
                }),
                # Gemma 3 12B SFT generation — selected by OOD val F1 via submit_sft_eval.py
                # --project sft-hyperparam-sweep-gemma --mode test.
                Row("Gemma-3-12B", scores={
                    "wildguardmix": 0.857, "xstest": 0.884, "aegis": 0.811,
                    "toxic_chat": 0.727, "openai_moderation": 0.761,
                }),
            ],
        ),
        Group(
            header=r"\textit{Ours --- Reasoning Distillation "
                   r"(GPT-OSS-120B $\rightarrow$ Gemma-3-12B)}",
            rows=[
                # Best Distillation: GPT-OSS-120B (teacher) → Gemma-3-12B (student) / Human Intent.
                # Trained on the annotated-intents dataset (~745 train ex, uncertainty-filtered).
                # Selected by submit_distillation_eval.py --mode test using marginal-mean selection.
                Row("Annotated intents only", scores={
                    "wildguardmix": 0.871, "xstest": 0.934, "aegis": 0.804,
                    "toxic_chat": 0.687, "openai_moderation": 0.791,
                }),
                # Broader-intent-mix ablation: 50/50 blend of annotated human_intent traces
                # (n=1378) + harm-stratified synthetic_intent traces from the WildGuardMix
                # scaling pool (n=1378), lr=2e-5, 5-epoch ceiling with early stopping.
                # See submit_broader_intent_mix.py.
                Row("+ Broader-pool synthetic intents", scores={
                    "wildguardmix": 0.8887, "xstest": 0.9519, "aegis": 0.8168,
                    "toxic_chat": 0.6941, "openai_moderation": 0.7764,
                }),
                # Scaling run (full WildGuardMix ~86.7k, 1 epoch, synthetic_intent) — disabled
                # until finalised. See submit_scaling.py.
                # Row("Full WildGuardMix (synthetic intents)", scores={
                #     "wildguardmix": 0.8947, "xstest": 0.9510, "aegis": 0.8321,
                #     "toxic_chat": 0.7108, "openai_moderation": 0.7412,
                # }),
                # Row("Mixed synthetic (annotated half = synthetic intents)", scores={
                #     "wildguardmix": 0.8821, "xstest": 0.9340, "aegis": 0.8142,
                #     "toxic_chat": 0.6893, "openai_moderation": 0.7705,
                # }),
            ],
        ),
        Group(
            # Best-by-OOD-validation single runs lifted from
            # report/latex/dpo_results.tex (the "Best run by OOD validation" block).
            # Canonical-sample-set tag is in brackets: hard-dpo [e], judge-gemma-standalone [c].
            header=r"\textit{Ours --- DPO}",
            rows=[
                Row("hard-dpo", scores={
                    "wildguardmix": 0.8560, "xstest": 0.8841, "aegis": 0.8240,
                    "toxic_chat": 0.7327, "openai_moderation": 0.7651,
                }),
                Row("judge-gemma-standalone", scores={
                    "wildguardmix": 0.8505, "xstest": 0.9086, "aegis": 0.8135,
                    "toxic_chat": 0.7079, "openai_moderation": 0.7663,
                }),
            ],
        ),
        Group(
            # Red-highlighted "best by OOD validation" GRPO runs from GRPO_results.tex,
            # both initialised from the SFT checkpoint:
            #   Label reward            → grpo_fn__with_reasoning__format_label__n8_b16_chat_x21_initSFT_max512_temp12
            #   Label and intent reward → grpo_fn__with_reasoning__format_label_judge_n8_b16_chat_x22_initSFT_max512_temp12
            header=r"\textit{Ours --- GRPO (continued from SFT)}",
            rows=[
                Row("Label reward", scores={
                    "wildguardmix": 0.886, "xstest": 0.892, "aegis": 0.835,
                    "toxic_chat": 0.704, "openai_moderation": 0.791,
                }),
                Row("Label and intent reward", scores={
                    "wildguardmix": 0.872, "xstest": 0.916, "aegis": 0.828,
                    "toxic_chat": 0.705, "openai_moderation": 0.810,
                }),
            ],
        ),
    ]


# ─── Rendering ────────────────────────────────────────────────────────────────
DATASET_LABELS = {
    "wildguardmix": "WGTest",
    "xstest": "XSTest",
    "aegis": "AEGIS 2",
    "toxic_chat": "ToxicChat",
    "openai_moderation": "OAI Mod",
}


def compute_average(scores: Dict[str, float], datasets: List[str]) -> float:
    valid = [scores[d] for d in datasets if d in scores]
    return sum(valid) / len(valid) if valid else 0.0


def column_ranks(values: List[float]) -> List[int]:
    """Return rank (0 = best) for each value using dense ranking.

    Ties share a rank and do NOT consume the next rank slot — so if two values
    are tied for best, both get rank 0 (bold) and the next distinct value gets
    rank 1 (underline). Comparison is at 3-decimal display precision so values
    that render identically (e.g. 0.7327 and 0.7333 → "0.733") count as tied.
    """
    rounded = [round(v, 3) for v in values]
    distinct_sorted = sorted(set(rounded), reverse=True)
    rank_of = {v: i for i, v in enumerate(distinct_sorted)}
    return [rank_of[v] for v in rounded]


def format_score(score: float, rank: int) -> str:
    s = f"{score:.3f}"
    if rank == 0:
        return rf"$\bm{{{s}}}$"
    if rank == 1:
        return rf"$\underline{{{s}}}$"
    return s


def render_row_label(row: Row) -> str:
    return row.name


def generate_latex_table(groups: List[Group], datasets: List[str]) -> str:
    # Flat list of (group_idx, row_idx, row) for global ranking
    flat = [(gi, ri, r) for gi, g in enumerate(groups) for ri, r in enumerate(g.rows)]
    n = len(flat)

    # Per-column ranks across all rows
    col_ranks: Dict[str, List[int]] = {}
    for d in datasets:
        col_ranks[d] = column_ranks([
            r.scores.get(d, float("-inf")) for _, _, r in flat
        ])
    averages = [compute_average(r.scores, datasets) for _, _, r in flat]
    avg_ranks = column_ranks(averages)

    n_cols = len(datasets) + 2  # label + datasets + average
    col_spec = "l" + "c" * (len(datasets) + 1)

    lines = [
        r"\begin{table*}[ht]",
        r"\centering",
        r"\small",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
    ]

    # Header row
    header_cells = [r"\textbf{Model}"]
    for d in datasets:
        header_cells.append(rf"\textbf{{{DATASET_LABELS.get(d, d)}}}")
    header_cells.append(r"\textbf{Average}")
    lines.append(" & ".join(header_cells) + r" \\")

    # Group blocks
    for gi, group in enumerate(groups):
        lines.append(r"\midrule")
        lines.append(rf"\multicolumn{{{n_cols}}}{{l}}{{{group.header}}} \\")
        lines.append(r"\midrule")
        for ri, row in enumerate(group.rows):
            flat_idx = next(i for i, (g, r, _) in enumerate(flat) if g == gi and r == ri)
            cells = [render_row_label(row)]
            for d in datasets:
                if d in row.scores:
                    cells.append(format_score(row.scores[d], col_ranks[d][flat_idx]))
                else:
                    cells.append("—")
            cells.append(format_score(averages[flat_idx], avg_ranks[flat_idx]))
            lines.append(" & ".join(cells) + r" \\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{F1 score comparison across five external safety benchmarks. "
        r"Open zero-shot LLMs are evaluated under their strongest prompting condition "
        r"from Section~\ref{sec:sft_model}; closed-source models (GPT-5.4, Claude Sonnet 4.6) "
        r"use vanilla generation only, as a per-condition sweep was prohibitive. "
        r"Best per column in \textbf{bold}; second-best \underline{underlined}.}",
        r"\label{fig:f1_benchmark}",
        r"\end{table*}",
    ])
    return "\n".join(lines) + "\n"


# ─── Entry point ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=_PROJECT_ROOT / "report/latex/benchmark_table.tex",
        help="Output LaTeX file (default: report/latex/benchmark_table.tex)",
    )
    parser.add_argument(
        "--datasets", type=str,
        default="wildguardmix,xstest,aegis,toxic_chat,openai_moderation",
        help="Comma-separated dataset keys in display order",
    )
    parser.add_argument(
        "--print", action="store_true",
        help="Also print the LaTeX to stdout",
    )
    args = parser.parse_args()

    # TODO: W&B fetching for baselines/sft/distillation projects — currently all values
    # are hardcoded in get_hardcoded_groups(). See cache helpers above when implementing.

    groups = get_hardcoded_groups()
    datasets = [d.strip() for d in args.datasets.split(",")]
    latex = generate_latex_table(groups, datasets)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(latex)
    print(f"✓ LaTeX table written to {args.output}")
    if args.print:
        print(latex)


if __name__ == "__main__":
    main()
