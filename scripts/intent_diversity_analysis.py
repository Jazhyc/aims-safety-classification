"""
Intent Diversity Analysis for DPO / Contrastive Learning

What it does:
  1. Loads the fine-tuned Llama 3.1 8B generation LoRA adapter with VLLM.
  2. For each prompt in the test split, samples k completions at higher temperature.
  3. Parses intent text and harm label from each sample.
  4. Computes intra-prompt semantic diversity using sentence-transformers.
  5. Builds DPO-format chosen/rejected pairs from label disagreements.
  6. Saves diversity_stats.jsonl, dpo_pairs.jsonl, summary.json, histogram.

Usage:
    python scripts/intent_diversity_analysis.py \
        --adapter-path trained_models/causal/hyperparam_sweep/lr_0.0005_e_5_adapter \
        --base-model meta-llama/Llama-3.1-8B-Instruct \
        --output-dir data/diversity_analysis/t0.8 \
        --num-samples 5 \
        --temperature 0.8
"""

import argparse
import json
import re
import os
from pathlib import Path
from collections import Counter
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from intention_jailbreak.model_generation.preprocessing import preprocess_data
from intention_jailbreak.model_generation.data_utils import (
    train_val_test_split,
    apply_binary_harm_mapping,
)


# ---------------------------------------------------------------------------
# Parser — handles all output formats the model may produce
# ---------------------------------------------------------------------------

def extract_intent_and_harm(raw_text: str):
    """
    Parse intent and harm from model output.

    Handles these formats (all observed in prediction files):
      "Intent: <text>; Harm: harmful"       <- standard format
      "Intent: <text>\nHarm: harmful"       <- newline separator  
      "Intent: <text> Harm: harmful"        <- space separator
      "<text>; Harm: harmful"               <- missing Intent: prefix
      "<text>"                              <- intent only, no harm

    Returns (intent_str, harm_str) — either may be None.
    """
    text = raw_text.strip()

    # Strip thinking tags (reasoning models)
    if "<think>" in text and "</think>" in text:
        text = text.split("</think>")[-1].strip()

    def norm_harm(s: str) -> Optional[str]:
        s = s.strip().lower().rstrip(".").rstrip(";")
        if s in {"safe", "harmless", "benign", "s", "b"}:
            return "safe"
        if s in {"harmful", "unsafe", "dangerous", "h", "u"}:
            return "harmful"
        return None

    # Pattern 1: "Intent: <text>; Harm: <label>"  (standard)
    m = re.search(r"Intent:\s*(.+?);\s*Harm:\s*(\S+)", text, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip(), norm_harm(m.group(2))

    # Pattern 2: "Intent: <text>\nHarm: <label>"
    m = re.search(r"Intent:\s*(.+?)\n\s*Harm:\s*(\S+)", text, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip().rstrip("."), norm_harm(m.group(2))

    # Pattern 3: "Intent: <text> Harm: <label>"  (no separator)
    m = re.search(r"Intent:\s*(.+?)\s+Harm:\s*(\S+)", text, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip().rstrip("."), norm_harm(m.group(2))

    # Pattern 4: "<text>; Harm: <label>"  (missing Intent: prefix)
    m = re.search(r"^(.+?);\s*Harm:\s*(\S+)", text, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip(), norm_harm(m.group(2))

    # Pattern 5: Harm label appears anywhere after a semicolon or newline
    m = re.search(r"[;\n]\s*Harm:\s*(\S+)", text, re.IGNORECASE)
    if m:
        harm = norm_harm(m.group(1))
        # Intent is everything before the match
        intent = text[:m.start()].strip()
        intent = re.sub(r"^Intent:\s*", "", intent, flags=re.IGNORECASE).strip()
        return intent if intent else text, harm

    # Pattern 6: Intent only — no harm label present
    m = re.search(r"^Intent:\s*(.+?)(?:;|$)", text, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip(), None

    # Pattern 7: Raw text with no structure — treat whole thing as intent
    return text if text else None, None


# ---------------------------------------------------------------------------
# Diversity computation
# ---------------------------------------------------------------------------

def compute_pairwise_diversity(embeddings: np.ndarray) -> float:
    """Average pairwise cosine distance (1 - sim). Range [0, 1]."""
    k = len(embeddings)
    if k < 2:
        return 0.0
    sim_matrix = cosine_similarity(embeddings)
    upper = [(i, j) for i in range(k) for j in range(i + 1, k)]
    distances = [1.0 - float(sim_matrix[i, j]) for i, j in upper]
    return float(np.mean(distances))


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_analysis(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Dataset — reproduce exact test split from training
    # ------------------------------------------------------------------
    print("\n=== Loading dataset ===")
    raw_dataset = preprocess_data()
    raw_dataset = apply_binary_harm_mapping(raw_dataset, binary_harm_mapping=True)

    split_config = {"data": {"test_size": args.test_size, "val_size": args.val_size, "seed": args.seed}}
    _, _, test_dataset = train_val_test_split(raw_dataset, split_config)

    if args.max_prompts is not None and args.max_prompts < len(test_dataset):
        test_dataset = test_dataset.select(range(args.max_prompts))

    print(f"Test set size: {len(test_dataset)}")

    # ------------------------------------------------------------------
    # 2. Load VLLM — pass tokenizer path explicitly to silence warning
    #    and ensure the fine-tuned tokenizer is used, not base Llama.
    # ------------------------------------------------------------------
    print(f"\n=== Loading model ===")
    print(f"  Base model  : {args.base_model}")
    print(f"  LoRA adapter: {args.adapter_path}")

    llm = LLM(
        model=args.base_model,
        tokenizer=args.adapter_path,   # use fine-tuned tokenizer
        enable_lora=True,
        max_lora_rank=64,
        max_loras=1,
        limit_mm_per_prompt={"image": 0},
        gpu_memory_utilization=0.90,
        max_model_len=2048,
        dtype="bfloat16",
        enforce_eager=True,
    )
    lora_request = LoRARequest("generation_lora", 1, args.adapter_path)

    sampling_params = SamplingParams(
        n=args.num_samples,
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        skip_special_tokens=True,
    )

    # ------------------------------------------------------------------
    # 3. Generate k samples per prompt
    # ------------------------------------------------------------------
    print(f"\n=== Generating {args.num_samples} samples per prompt (temperature={args.temperature}) ===")

    examples = list(test_dataset)
    # Match exact prompt format used during training (causal.py format_prompt)
    prompts = [f"{ex['prompt']}\n" for ex in examples]

    outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)

    # ------------------------------------------------------------------
    # 4. Sentence-transformer for diversity
    # ------------------------------------------------------------------
    print("=== Loading sentence-transformer ===")
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # ------------------------------------------------------------------
    # 5. Parse, compute diversity, build DPO pairs
    # ------------------------------------------------------------------
    print("\n=== Analysing diversity and building DPO pairs ===")

    diversity_stats = []
    dpo_pairs = []
    parse_failures = 0
    label_missing = 0
    total_completions = 0

    for ex, output in zip(examples, outputs):
        prompt_text = ex["prompt"]
        gold_harm = ex.get("Annotator Harm")
        true_intent = ex.get("intent", "")
        ex_id = ex.get("id", "")

        parsed = []
        for completion in output.outputs:
            total_completions += 1
            raw_text = completion.text.strip()
            intent, harm = extract_intent_and_harm(raw_text)

            if not intent:
                parse_failures += 1
            if harm is None:
                label_missing += 1

            parsed.append({"raw": raw_text, "intent": intent, "harm": harm})

        # Only keep samples where both intent and harm were parsed
        valid = [p for p in parsed if p["intent"] and p["harm"]]

        # Diversity
        diversity_score = 0.0
        if len(valid) >= 2:
            embeddings = embedder.encode(
                [p["intent"] for p in valid],
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            diversity_score = compute_pairwise_diversity(embeddings)

        # Label consistency
        harm_labels = [p["harm"] for p in valid]
        label_counts = Counter(harm_labels)
        n_valid = len(valid)
        majority_label = label_counts.most_common(1)[0][0] if label_counts else None
        label_agreement_rate = (label_counts[majority_label] / n_valid) if n_valid > 0 else 0.0

        # DPO pair construction
        n_pairs_built = 0
        if gold_harm and n_valid >= 2:
            chosen_cands   = [p for p in valid if p["harm"] == gold_harm]
            rejected_cands = [p for p in valid if p["harm"] != gold_harm]
            if chosen_cands and rejected_cands:
                chosen   = max(chosen_cands,   key=lambda p: len(p["intent"]))
                rejected = max(rejected_cands, key=lambda p: len(p["intent"]))
                dpo_pairs.append({
                    "id": ex_id,
                    "prompt": prompt_text,
                    "true_intent": true_intent,
                    "gold_harm": gold_harm,
                    "chosen": chosen["raw"],
                    "chosen_intent": chosen["intent"],
                    "chosen_harm": chosen["harm"],
                    "rejected": rejected["raw"],
                    "rejected_intent": rejected["intent"],
                    "rejected_harm": rejected["harm"],
                    "diversity_score": diversity_score,
                    "n_valid_samples": n_valid,
                })
                n_pairs_built = 1

        diversity_stats.append({
            "id": ex_id,
            "prompt": prompt_text[:120],
            "gold_harm": gold_harm,
            "n_valid_samples": n_valid,
            "n_total_samples": len(parsed),
            "diversity_score": diversity_score,
            "label_agreement_rate": label_agreement_rate,
            "majority_predicted_harm": majority_label,
            "majority_correct": majority_label == gold_harm if gold_harm else None,
            "n_dpo_pairs": n_pairs_built,
            # Save first 2 raw outputs for manual inspection
            "sample_outputs": [p["raw"][:200] for p in parsed[:2]],
        })

    # ------------------------------------------------------------------
    # 6. Save outputs
    # ------------------------------------------------------------------
    stats_path = output_dir / "diversity_stats.jsonl"
    with stats_path.open("w", encoding="utf-8") as f:
        for row in diversity_stats:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Per-prompt stats -> {stats_path}")

    dpo_path = output_dir / "dpo_pairs.jsonl"
    with dpo_path.open("w", encoding="utf-8") as f:
        for pair in dpo_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    print(f"DPO pairs        -> {dpo_path}")

    # ------------------------------------------------------------------
    # 7. Summary
    # ------------------------------------------------------------------
    all_diversity = [r["diversity_score"] for r in diversity_stats if r["n_valid_samples"] >= 2]
    all_agreement = [r["label_agreement_rate"] for r in diversity_stats if r["n_valid_samples"] > 0]
    majority_correct = [r["majority_correct"] for r in diversity_stats if r["majority_correct"] is not None]

    summary = {
        "adapter_path": str(args.adapter_path),
        "base_model": args.base_model,
        "temperature": args.temperature,
        "n_prompts_total": len(examples),
        "n_prompts_with_valid_samples": sum(1 for r in diversity_stats if r["n_valid_samples"] > 0),
        "n_dpo_pairs": len(dpo_pairs),
        "total_completions": total_completions,
        "parse_failures_total": parse_failures,
        "label_missing_total": label_missing,
        "diversity": {
            "mean":   float(np.mean(all_diversity))   if all_diversity else 0.0,
            "median": float(np.median(all_diversity)) if all_diversity else 0.0,
            "std":    float(np.std(all_diversity))    if all_diversity else 0.0,
            "min":    float(np.min(all_diversity))    if all_diversity else 0.0,
            "max":    float(np.max(all_diversity))    if all_diversity else 0.0,
            "n_prompts": len(all_diversity),
        },
        "label_agreement": {
            "mean": float(np.mean(all_agreement)) if all_agreement else 0.0,
            "fully_consistent_rate": float(
                sum(1 for a in all_agreement if a == 1.0) / len(all_agreement)
            ) if all_agreement else 0.0,
        },
        "majority_vote_accuracy": float(np.mean(majority_correct)) if majority_correct else 0.0,
    }

    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary          -> {summary_path}")

    # ------------------------------------------------------------------
    # 8. readable summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("DIVERSITY ANALYSIS SUMMARY")
    print("=" * 65)
    print(f"Total prompts          : {summary['n_prompts_total']}")
    print(f"Total completions      : {summary['total_completions']}")
    print(f"Parse failures         : {summary['parse_failures_total']} "
          f"({100*summary['parse_failures_total']/max(summary['total_completions'],1):.1f}%)")
    print(f"Label missing          : {summary['label_missing_total']} "
          f"({100*summary['label_missing_total']/max(summary['total_completions'],1):.1f}%)")
    print(f"Prompts with valid k   : {summary['n_prompts_with_valid_samples']}")
    print(f"DPO pairs constructed  : {summary['n_dpo_pairs']}")
    print()
    d = summary["diversity"]
    print(f"Diversity  mean={d['mean']:.4f}  median={d['median']:.4f}  std={d['std']:.4f}  range=[{d['min']:.4f},{d['max']:.4f}]")
    la = summary["label_agreement"]
    print(f"Label agreement  mean={la['mean']:.4f}  fully_consistent={la['fully_consistent_rate']:.2%}")
    print(f"Majority-vote accuracy : {summary['majority_vote_accuracy']:.4f}")
    print("=" * 65)

    _plot_histogram(all_diversity, output_dir / "diversity_histogram.png", args.temperature)
    print(f"\nDone. All outputs in {output_dir}/")


def _plot_histogram(scores, save_path: Path, temperature: float):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if not scores:
            return
        plt.figure(figsize=(8, 5))
        plt.hist(scores, bins=30, edgecolor="black", color="steelblue", alpha=0.85)
        mean_val = float(np.mean(scores))
        plt.axvline(mean_val, color="red", linestyle="--", linewidth=1.5, label=f"Mean={mean_val:.3f}")
        plt.xlabel("Avg pairwise cosine distance", fontsize=12)
        plt.ylabel("Number of prompts", fontsize=12)
        plt.title(f"Intent Diversity — temperature={temperature}", fontsize=13)
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Histogram        -> {save_path}")
    except ImportError:
        print("matplotlib not available — skipping histogram.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--adapter-path",   type=str, default="trained_models/causal/hyperparam_sweep/lr_0.0005_e_5_adapter")
    p.add_argument("--base-model",     type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--output-dir",     type=str, default="data/diversity_analysis")
    p.add_argument("--num-samples","-k", type=int, default=5)
    p.add_argument("--temperature",    type=float, default=0.8)
    p.add_argument("--top-p",          type=float, default=0.95)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--test-size",      type=float, default=0.1)
    p.add_argument("--val-size",       type=float, default=0.1)
    p.add_argument("--seed",           type=int, default=22)
    p.add_argument("--max-prompts",    type=int, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_analysis(args)