"""
Intent Diversity Analysis for DPO / Contrastive Learning

This script assesses whether sampled intents from the fine-tuned SFT generation
model are diverse enough and structurally suited for DPO or contrastive learning.

What it does:
  1. Loads the fine-tuned Llama 3.1 8B generation LoRA adapter with VLLM.
  2. For each prompt in the test split, samples k completions at higher temperature.
     Each completion has format: "Intent: <text>; Harm: <label>"
  3. Parses intent text and harm label from each sample.
  4. Computes intra-prompt semantic diversity using sentence-transformers cosine similarity.
  5. Checks label consistency: how often k samples agree on harmful/safe.
  6. Builds DPO-format pairs where samples disagree with each other and one matches gold:
       chosen  = sample whose harm label matches gold label
       rejected = sample whose harm label does NOT match gold label
  7. Saves:
       - data/diversity_analysis/diversity_stats.jsonl     (per-prompt stats)
       - data/diversity_analysis/dpo_pairs.jsonl           (chosen/rejected pairs)
       - data/diversity_analysis/summary.json              (aggregate statistics)
       - data/diversity_analysis/diversity_histogram.png   (plot)

Usage (on cluster, from repo root):
    python scripts/intent_diversity_analysis.py \\
        --adapter-path trained_models/causal/hyperparam_sweep/lr_0.0005_e_5_adapter \\
        --base-model meta-llama/Llama-3.1-8B-Instruct \\
        --output-dir data/diversity_analysis \\
        --num-samples 5 \\
        --temperature 0.8 \\
        --max-prompts 500
"""

import argparse
import json
import re
import os
from pathlib import Path
from collections import Counter
from typing import Optional

import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

# ---------------------------------------------------------------------------
# Reuse preprocessing + split logic from the existing codebase so the test
# split here is identical to the one used during training evaluation.
# ---------------------------------------------------------------------------
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from intention_jailbreak.model_generation.preprocessing import preprocess_data
from intention_jailbreak.model_generation.data_utils import (
    train_val_test_split,
    apply_binary_harm_mapping,
)


# ---------------------------------------------------------------------------
# Parsing helpers — mirrors extract_intent_and_harm from safety_experiment.py
# ---------------------------------------------------------------------------

def extract_intent_and_harm(generated_text: str):
    """
    Parse 'Intent: <text>; Harm: <label>' from model output.

    Returns (intent_str, harm_str) where either may be None if not found.
    Mirrors the logic in safety_experiment.py so behaviour is consistent.
    """
    text = generated_text.strip()

    # Strip thinking tags (Qwen / reasoning models)
    if "<think>" in text and "</think>" in text:
        text = text.split("</think>")[-1].strip()

    def normalise_harm(s: str) -> Optional[str]:
        s = s.strip().lower().rstrip(".")
        if s in {"safe", "harmless", "benign", "s", "b"}:
            return "safe"
        if s in {"harmful", "unsafe", "dangerous", "h", "u"}:
            return "harmful"
        return None

    # Pattern 1: "Intent: <text>; Harm: <label>"
    m = re.search(r"Intent:\s*(.+?);\s*Harm:\s*(\S+)", text, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip(), normalise_harm(m.group(2))

    # Pattern 2: "Intent: <text>\nHarm: <label>"
    m = re.search(r"Intent:\s*(.+?)\n\s*Harm:\s*(\S+)", text, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip().rstrip("."), normalise_harm(m.group(2))

    # Pattern 3: "Intent: <text> Harm: <label>" (no separator)
    m = re.search(r"Intent:\s*(.+?)\s+Harm:\s*(\S+)", text, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip().rstrip("."), normalise_harm(m.group(2))

    # Pattern 4: intent only
    m = re.search(r"Intent:\s*(.+?)(?:\n|$)", text, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip(), None

    return text, None


# ---------------------------------------------------------------------------
# Diversity computation
# ---------------------------------------------------------------------------

def compute_pairwise_diversity(embeddings: np.ndarray) -> float:
    """
    Average pairwise cosine *distance* (1 - similarity) between k embeddings.
    Returns a value in [0, 1] where 0 = all identical, 1 = maximally diverse.
    """
    k = len(embeddings)
    if k < 2:
        return 0.0
    sim_matrix = cosine_similarity(embeddings)
    # Upper triangle indices (excluding diagonal)
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
    # 1. Load dataset and reproduce the same test split as training
    # ------------------------------------------------------------------
    print("\n=== Loading dataset ===")
    raw_dataset = preprocess_data()

    # Apply binary harm mapping (4 categories -> harmful/safe), same as training
    raw_dataset = apply_binary_harm_mapping(raw_dataset, binary_harm_mapping=True)

    # Reproduce train/val/test split with the same seed used in training (default 22)
    # We only need the test split for evaluation
    split_config = {
        "data": {
            "test_size": args.test_size,
            "val_size": args.val_size,
            "seed": args.seed,
        }
    }
    _, _, test_dataset = train_val_test_split(raw_dataset, split_config)

    # Optional: limit number of prompts for faster iteration / debugging
    if args.max_prompts is not None and args.max_prompts < len(test_dataset):
        test_dataset = test_dataset.select(range(args.max_prompts))

    print(f"Test set size (after optional limit): {len(test_dataset)}")

    # ------------------------------------------------------------------
    # 2. Load VLLM with LoRA adapter
    # ------------------------------------------------------------------
    print(f"\n=== Loading model ===")
    print(f"  Base model : {args.base_model}")
    print(f"  LoRA adapter: {args.adapter_path}")

    # max_lora_rank must be >= the rank used during training.
    # The hyperparam sweep used rank=16 by default; set to 64 to be safe.
    llm = LLM(
        model=args.base_model,
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

    # Sampling parameters: higher temperature to encourage diversity
    # n > 1 tells VLLM to return multiple independent completions per prompt
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
    prompts = [f"{ex['prompt']}\n" for ex in examples]

    # VLLM returns one RequestOutput per prompt; each has n completions
    outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)

    # ------------------------------------------------------------------
    # 4. Load sentence-transformer for diversity computation
    # ------------------------------------------------------------------
    print("\n=== Loading sentence-transformer for embeddings ===")
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # ------------------------------------------------------------------
    # 5. Analyse diversity and build DPO pairs
    # ------------------------------------------------------------------
    print("\n=== Analysing diversity and building DPO pairs ===")

    diversity_stats = []   # one entry per prompt
    dpo_pairs = []         # chosen / rejected pairs

    parse_failures = 0
    label_missing = 0

    for ex, output in zip(examples, outputs):
        prompt_text = ex["prompt"]
        gold_harm = ex.get("Annotator Harm")  # already binary after mapping
        true_intent = ex.get("intent", "")
        ex_id = ex.get("id", "")

        # Parse all k completions for this prompt
        parsed = []
        for completion in output.outputs:
            raw_text = completion.text.strip()
            intent, harm = extract_intent_and_harm(raw_text)
            parsed.append({
                "raw": raw_text,
                "intent": intent,
                "harm": harm,
            })
            if intent is None:
                parse_failures += 1
            if harm is None:
                label_missing += 1

        # Filter to samples that have both intent and harm
        valid = [p for p in parsed if p["intent"] and p["harm"]]

        # ---- Diversity computation ----
        diversity_score = 0.0
        if len(valid) >= 2:
            intent_texts = [p["intent"] for p in valid]
            embeddings = embedder.encode(
                intent_texts, convert_to_numpy=True, show_progress_bar=False
            )
            diversity_score = compute_pairwise_diversity(embeddings)

        # ---- Label consistency ----
        harm_labels = [p["harm"] for p in valid]
        label_counts = Counter(harm_labels)
        n_valid = len(valid)
        majority_label = label_counts.most_common(1)[0][0] if label_counts else None
        label_agreement_rate = (
            label_counts[majority_label] / n_valid if n_valid > 0 else 0.0
        )

        # ---- DPO pair construction ----
        # We only build a pair if:
        #   a) We know the gold label
        #   b) There is at least one sample matching gold (chosen candidate)
        #   c) There is at least one sample NOT matching gold (rejected candidate)
        # Strategy: use the chosen sample with the highest-quality intent
        # (longest, as a simple proxy) and the rejected sample with the
        # lowest quality to maximise the signal gap.
        n_pairs_built = 0
        if gold_harm and n_valid >= 2:
            chosen_candidates = [p for p in valid if p["harm"] == gold_harm]
            rejected_candidates = [p for p in valid if p["harm"] != gold_harm]

            if chosen_candidates and rejected_candidates:
                # Pick the longest intent as a simple quality heuristic
                chosen = max(chosen_candidates, key=lambda p: len(p["intent"]))
                rejected = max(rejected_candidates, key=lambda p: len(p["intent"]))

                dpo_pairs.append({
                    "id": ex_id,
                    "prompt": prompt_text,
                    "true_intent": true_intent,
                    "gold_harm": gold_harm,
                    "chosen": chosen["raw"],          # full raw generation (intent + harm)
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
            "prompt": prompt_text[:120],   # truncate for readability in the file
            "gold_harm": gold_harm,
            "n_valid_samples": n_valid,
            "diversity_score": diversity_score,
            "label_agreement_rate": label_agreement_rate,
            "majority_predicted_harm": majority_label,
            "majority_correct": majority_label == gold_harm if gold_harm else None,
            "all_parsed": parsed,
            "n_dpo_pairs": n_pairs_built,
        })

    # ------------------------------------------------------------------
    # 6. Save outputs
    # ------------------------------------------------------------------

    # Per-prompt diversity stats
    stats_path = output_dir / "diversity_stats.jsonl"
    with stats_path.open("w", encoding="utf-8") as f:
        for row in diversity_stats:
            # Don't write all_parsed (too verbose) — save a compact version
            compact = {k: v for k, v in row.items() if k != "all_parsed"}
            f.write(json.dumps(compact, ensure_ascii=False) + "\n")
    print(f"\nPer-prompt stats saved to: {stats_path}")

    # DPO pairs
    dpo_path = output_dir / "dpo_pairs.jsonl"
    with dpo_path.open("w", encoding="utf-8") as f:
        for pair in dpo_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    print(f"DPO pairs saved to:        {dpo_path}")

    # ------------------------------------------------------------------
    # 7. Compute and print aggregate summary
    # ------------------------------------------------------------------
    all_diversity = [r["diversity_score"] for r in diversity_stats if r["n_valid_samples"] >= 2]
    all_agreement = [r["label_agreement_rate"] for r in diversity_stats if r["n_valid_samples"] > 0]
    majority_correct = [
        r["majority_correct"]
        for r in diversity_stats
        if r["majority_correct"] is not None
    ]

    summary = {
        "adapter_path": str(args.adapter_path),
        "base_model": args.base_model,
        "n_prompts_total": len(examples),
        "n_prompts_with_valid_samples": sum(1 for r in diversity_stats if r["n_valid_samples"] > 0),
        "n_dpo_pairs": len(dpo_pairs),
        "parse_failures_total": parse_failures,
        "label_missing_total": label_missing,
        "diversity": {
            "mean": float(np.mean(all_diversity)) if all_diversity else 0.0,
            "median": float(np.median(all_diversity)) if all_diversity else 0.0,
            "std": float(np.std(all_diversity)) if all_diversity else 0.0,
            "min": float(np.min(all_diversity)) if all_diversity else 0.0,
            "max": float(np.max(all_diversity)) if all_diversity else 0.0,
            "n_prompts": len(all_diversity),
        },
        "label_agreement": {
            "mean": float(np.mean(all_agreement)) if all_agreement else 0.0,
            "fully_consistent_rate": float(
                sum(1 for a in all_agreement if a == 1.0) / len(all_agreement)
            ) if all_agreement else 0.0,
        },
        "majority_vote_accuracy": float(np.mean(majority_correct)) if majority_correct else 0.0,
        "generation_params": {
            "num_samples_k": args.num_samples,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_new_tokens": args.max_new_tokens,
        },
    }

    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to:          {summary_path}")

    # ------------------------------------------------------------------
    # 8. Print human-readable summary to stdout
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("DIVERSITY ANALYSIS SUMMARY")
    print("=" * 65)
    print(f"Total prompts analysed      : {summary['n_prompts_total']}")
    print(f"Prompts with valid samples  : {summary['n_prompts_with_valid_samples']}")
    print(f"DPO pairs constructed       : {summary['n_dpo_pairs']}")
    print(f"Parse failures (intent=None): {summary['parse_failures_total']}")
    print(f"Label missing (harm=None)   : {summary['label_missing_total']}")
    print()
    print("Intent Diversity (cosine distance, higher = more diverse):")
    d = summary["diversity"]
    print(f"  mean   : {d['mean']:.4f}")
    print(f"  median : {d['median']:.4f}")
    print(f"  std    : {d['std']:.4f}")
    print(f"  range  : [{d['min']:.4f}, {d['max']:.4f}]")
    print()
    print("Label Consistency across k samples:")
    la = summary["label_agreement"]
    print(f"  mean agreement rate        : {la['mean']:.4f}  (1.0 = all k agree)")
    print(f"  fully consistent prompts   : {la['fully_consistent_rate']:.2%}")
    print()
    print(f"Majority-vote accuracy (vs gold): {summary['majority_vote_accuracy']:.4f}")
    print("=" * 65)

    # ------------------------------------------------------------------
    # 9. Optional: plot diversity histogram
    # ------------------------------------------------------------------
    _plot_histogram(all_diversity, output_dir / "diversity_histogram.png")

    print("\nDone. Check the output directory for all saved files.")
    print(f"  {output_dir}/")
    print(f"    diversity_stats.jsonl  — per-prompt stats")
    print(f"    dpo_pairs.jsonl        — {len(dpo_pairs)} chosen/rejected pairs")
    print(f"    summary.json           — aggregate numbers")
    print(f"    diversity_histogram.png")


def _plot_histogram(diversity_scores, save_path: Path):
    """Save a histogram of per-prompt diversity scores."""
    try:
        import matplotlib
        matplotlib.use("Agg")   # non-interactive backend for cluster use
        import matplotlib.pyplot as plt

        if not diversity_scores:
            print("No diversity scores to plot.")
            return

        plt.figure(figsize=(8, 5))
        plt.hist(diversity_scores, bins=30, edgecolor="black", color="steelblue", alpha=0.85)
        plt.xlabel("Intra-prompt diversity (avg pairwise cosine distance)", fontsize=12)
        plt.ylabel("Number of prompts", fontsize=12)
        plt.title("Distribution of Intent Diversity across Test Prompts", fontsize=13)
        mean_val = float(np.mean(diversity_scores))
        plt.axvline(mean_val, color="red", linestyle="--", linewidth=1.5,
                    label=f"Mean = {mean_val:.3f}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Histogram saved to:        {save_path}")
    except ImportError:
        print("matplotlib not available — skipping histogram plot.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyse intent diversity from a fine-tuned generation model "
                    "and build DPO chosen/rejected pairs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model paths
    parser.add_argument(
        "--adapter-path",
        type=str,
        default="trained_models/causal/hyperparam_sweep/lr_0.0005_e_5_adapter",
        help="Path to the LoRA adapter directory (generation model, not classification).",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="HuggingFace path to the base model.",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/diversity_analysis",
        help="Directory to write all output files.",
    )

    # Sampling
    parser.add_argument(
        "--num-samples", "-k",
        type=int,
        default=5,
        help="Number of independent completions to generate per prompt.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (higher = more diverse outputs).",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p (nucleus) sampling parameter.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Maximum tokens to generate per completion.",
    )

    # Dataset split (must match training config)
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.1,
        help="Fraction of data held out as test set (must match training split).",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.1,
        help="Fraction of data held out as val set (must match training split).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=22,
        help="Random seed for dataset split (must match training split).",
    )

    # Misc
    parser.add_argument(
        "--max-prompts",
        type=int,
        default=None,
        help="Limit the number of prompts processed (useful for quick debugging). "
             "Default: use the full test set.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_analysis(args)