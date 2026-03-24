"""
What it does:
  1. Loads the fine-tuned Llama 3.1 8B generation LoRA adapter with VLLM.
  2. For each prompt in the test split, samples k completions at higher temperature.
  3. Parses intent text and harm label from each sample.
  4. Optionally reclassifies harm using either:
       - LLM at temperature=0 (greedy, deterministic) — --harm-source llm_t0
       - Trained ModernBERT intent classifier         — --harm-source modernbert
  5. Computes intra-prompt semantic diversity using sentence-transformers.
  6. Builds DPO-format chosen/rejected pairs from label disagreements.
  7. Saves parsed_samples.jsonl, diversity_stats.jsonl, dpo_pairs.jsonl,
     summary.json, histogram.

Usage:
    # Original mode (harm label from the same generation)
    python scripts/intent_diversity_analysis.py \
        --adapter-path trained_models/causal/hyperparam_sweep/lr_5e-05_e_5_adapter \
        --base-model meta-llama/Llama-3.1-8B-Instruct \
        --output-dir data/diversity_analysis/t0.8 \
        --num-samples 5 \
        --temperature 0.8

    # LLM-at-t0 reclassification (skips generation, loads parsed_samples.jsonl)
    python scripts/intent_diversity_analysis.py \
        --adapter-path trained_models/causal/hyperparam_sweep/lr_5e-05_e_5_adapter \
        --base-model meta-llama/Llama-3.1-8B-Instruct \
        --from-samples data/diversity_analysis/t0.8/parsed_samples.jsonl \
        --harm-source llm_t0 \
        --output-dir data/diversity_analysis_llm_t0/t0.8

    # ModernBERT reclassification (skips generation AND vLLM load)
    python scripts/intent_diversity_analysis.py \
        --from-samples data/diversity_analysis/t0.8/parsed_samples.jsonl \
        --harm-source modernbert \
        --modernbert-path trained_models/bert_intent/ModernBERT-base \
        --output-dir data/diversity_analysis_modernbert/t0.8
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

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from intention_jailbreak.model_generation.preprocessing import preprocess_data
from intention_jailbreak.model_generation.data_utils import apply_binary_harm_mapping


# ---------------------------------------------------------------------------
# Shared normaliser — used by parser and LLM-t0 classifier
# ---------------------------------------------------------------------------

def norm_harm(s: str) -> Optional[str]:
    s = s.strip().lower().rstrip(".").rstrip(";")
    if s in {"safe", "harmless", "benign", "s", "b"}:
        return "safe"
    if s in {"harmful", "unsafe", "dangerous", "h", "u"}:
        return "harmful"
    return None


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
# Harm reclassification — LLM at temperature=0
# ---------------------------------------------------------------------------

def classify_intents_llm_t0(llm, lora_request, intent_texts: list) -> list:
    """
    Classify intent texts using the LLM at temperature=0 (greedy / deterministic).

    Prompts the model with "Intent: {text}\\nHarm: " and reads the single-token
    completion.  This reuses the model's fine-tuned output format without needing
    the original user prompt.

    Returns a list of "harmful" | "safe" | None (one per input).
    """
    from vllm import SamplingParams as _SP

    prompts = [f"Intent: {t}\nHarm: " for t in intent_texts]
    params = _SP(n=1, max_tokens=5, temperature=0.0, top_p=1.0,
                 skip_special_tokens=True)
    outputs = llm.generate(prompts, params, lora_request=lora_request)
    return [norm_harm(o.outputs[0].text.strip()) for o in outputs]


# ---------------------------------------------------------------------------
# Harm reclassification — ModernBERT intent classifier
# ---------------------------------------------------------------------------

def classify_intents_modernbert(model_path: str, intent_texts: list,
                                 batch_size: int = 32) -> list:
    """
    Classify intent texts using the trained ModernBERT intent classifier.

    Loads the model once, runs batched inference, and returns a list of
    "harmful" | "safe" strings (one per input).
    """
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch

    print(f"[ModernBERT] Loading classifier from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"[ModernBERT] Running on {device}")

    # id2label keys may be strings ("0", "1") or ints depending on HF version
    id2label = {int(k): v for k, v in model.config.id2label.items()}

    results = []
    for i in range(0, len(intent_texts), batch_size):
        batch = intent_texts[i : i + batch_size]
        enc = tokenizer(batch, max_length=256, truncation=True,
                        padding=True, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
        preds = logits.argmax(dim=-1).cpu().tolist()
        results.extend([id2label[p] for p in preds])

    print(f"[ModernBERT] Classified {len(results)} intents")
    return results


# ---------------------------------------------------------------------------
# parsed_samples.jsonl helpers
# ---------------------------------------------------------------------------

def _save_parsed_samples(all_parsed_data: list, path: Path):
    """Save all k parsed samples per prompt so reclassification can reuse them."""
    with path.open("w", encoding="utf-8") as f:
        for d in all_parsed_data:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"Parsed samples   -> {path}")


def _load_parsed_samples(path: str) -> list:
    """Load parsed_samples.jsonl written by a previous generation run."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"Loaded {len(records)} prompts from {path}")
    return records


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_analysis(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Decide whether vLLM is needed
    # ------------------------------------------------------------------
    need_llm = (args.from_samples is None) or (args.harm_source == "llm_t0")

    # ------------------------------------------------------------------
    # 1. Dataset — use canonical HF test split
    # ------------------------------------------------------------------
    print("\n=== Loading dataset (HF test split) ===")
    test_dataset = preprocess_data(split="test")
    test_dataset = apply_binary_harm_mapping(test_dataset, binary_harm_mapping=True)

    if args.max_prompts is not None and args.max_prompts < len(test_dataset):
        test_dataset = test_dataset.select(range(args.max_prompts))

    print(f"Test set size: {len(test_dataset)}")

    # ------------------------------------------------------------------
    # 2. Load vLLM (only when required)
    # ------------------------------------------------------------------
    llm = None
    lora_request = None

    if need_llm:
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest

        print(f"\n=== Loading model ===")
        print(f"  Base model  : {args.base_model}")
        print(f"  LoRA adapter: {args.adapter_path}")

        llm = LLM(
            model=args.base_model,
            tokenizer=args.adapter_path,
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

    # ------------------------------------------------------------------
    # Phase 1: Generate OR load parsed samples
    # ------------------------------------------------------------------
    if args.from_samples:
        print(f"\n=== Loading parsed samples from {args.from_samples} ===")
        all_parsed_data = _load_parsed_samples(args.from_samples)
        examples = [
            {"id": d["id"], "prompt": d["prompt"], "Annotator Harm": d["gold_harm"],
             "intent": d["true_intent"]}
            for d in all_parsed_data
        ]
    else:
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            n=args.num_samples,
            max_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            skip_special_tokens=True,
        )

        print(f"\n=== Generating {args.num_samples} samples per prompt "
              f"(temperature={args.temperature}) ===")

        examples = list(test_dataset)
        prompts = [f"{ex['prompt']}\n" for ex in examples]
        outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)

        all_parsed_data = []
        parse_failures_gen = 0
        label_missing_gen = 0

        for ex, output in zip(examples, outputs):
            samples = []
            for completion in output.outputs:
                raw_text = completion.text.strip()
                intent, harm = extract_intent_and_harm(raw_text)
                if not intent:
                    parse_failures_gen += 1
                if harm is None:
                    label_missing_gen += 1
                samples.append({"raw": raw_text, "intent": intent, "harm": harm})

            all_parsed_data.append({
                "id":          ex.get("id", ""),
                "prompt":      ex["prompt"],
                "gold_harm":   ex.get("Annotator Harm"),
                "true_intent": ex.get("intent", ""),
                "samples":     samples,
            })

        _save_parsed_samples(all_parsed_data,
                             output_dir / "parsed_samples.jsonl")
        print(f"Generation parse failures : {parse_failures_gen}")
        print(f"Generation label missing  : {label_missing_gen}")

    # ------------------------------------------------------------------
    # Phase 2: Reclassify harm labels if requested
    # ------------------------------------------------------------------
    if args.harm_source in ("llm_t0", "modernbert"):
        print(f"\n=== Reclassifying harm labels using: {args.harm_source} ===")

        # Collect all intent texts with their position (prompt_idx, sample_idx)
        flat_intents = []
        flat_idx = []
        for i, d in enumerate(all_parsed_data):
            for j, s in enumerate(d["samples"]):
                if s["intent"]:
                    flat_intents.append(s["intent"])
                    flat_idx.append((i, j))

        print(f"  Classifying {len(flat_intents)} intent texts...")

        if args.harm_source == "llm_t0":
            new_labels = classify_intents_llm_t0(llm, lora_request, flat_intents)
        else:
            new_labels = classify_intents_modernbert(
                args.modernbert_path, flat_intents
            )

        # Write labels back into the parsed data structure
        for (i, j), label in zip(flat_idx, new_labels):
            all_parsed_data[i]["samples"][j]["harm"] = label

        overridden = sum(1 for l in new_labels if l is not None)
        print(f"  Harm labels assigned: {overridden} / {len(flat_intents)}")
        # Save per-sample reclassified labels so the notebook can show them
        _save_parsed_samples(all_parsed_data, output_dir / "parsed_samples.jsonl")

    # ------------------------------------------------------------------
    # Phase 3: Sentence-transformer for diversity
    # ------------------------------------------------------------------
    print("=== Loading sentence-transformer ===")
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # ------------------------------------------------------------------
    # Phase 4: Compute diversity, build DPO pairs
    # ------------------------------------------------------------------
    print("\n=== Analysing diversity and building DPO pairs ===")

    diversity_stats = []
    dpo_pairs = []
    parse_failures = 0
    label_missing = 0
    total_completions = 0

    for d in all_parsed_data:
        prompt_text = d["prompt"]
        gold_harm   = d["gold_harm"]
        true_intent = d["true_intent"]
        ex_id       = d["id"]
        parsed      = d["samples"]

        for p in parsed:
            total_completions += 1
            if not p["intent"]:
                parse_failures += 1
            if p["harm"] is None:
                label_missing += 1

        # Only keep samples where both intent and harm are present
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
        label_agreement_rate = (
            label_counts[majority_label] / n_valid if n_valid > 0 else 0.0
        )

        # DPO pair construction
        n_pairs_built = 0
        if gold_harm and n_valid >= 2:
            chosen_cands   = [p for p in valid if p["harm"] == gold_harm]
            rejected_cands = [p for p in valid if p["harm"] != gold_harm]
            if chosen_cands and rejected_cands:
                chosen   = max(chosen_cands,   key=lambda p: len(p["intent"]))
                rejected = max(rejected_cands, key=lambda p: len(p["intent"]))
                dpo_pairs.append({
                    "id":              ex_id,
                    "prompt":          prompt_text,
                    "true_intent":     true_intent,
                    "gold_harm":       gold_harm,
                    "chosen":          chosen["raw"],
                    "chosen_intent":   chosen["intent"],
                    "chosen_harm":     chosen["harm"],
                    "rejected":        rejected["raw"],
                    "rejected_intent": rejected["intent"],
                    "rejected_harm":   rejected["harm"],
                    "diversity_score": diversity_score,
                    "n_valid_samples": n_valid,
                })
                n_pairs_built = 1

        diversity_stats.append({
            "id":                     ex_id,
            "prompt":                 prompt_text[:120],
            "gold_harm":              gold_harm,
            "n_valid_samples":        n_valid,
            "n_total_samples":        len(parsed),
            "diversity_score":        diversity_score,
            "label_agreement_rate":   label_agreement_rate,
            "majority_predicted_harm": majority_label,
            "majority_correct":       majority_label == gold_harm if gold_harm else None,
            "n_dpo_pairs":            n_pairs_built,
            "sample_outputs":         [p["raw"][:200] for p in parsed[:2]],
        })

    # ------------------------------------------------------------------
    # Phase 5: Save outputs
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
    # Phase 6: Summary
    # ------------------------------------------------------------------
    all_diversity    = [r["diversity_score"]       for r in diversity_stats
                        if r["n_valid_samples"] >= 2]
    all_agreement    = [r["label_agreement_rate"]  for r in diversity_stats
                        if r["n_valid_samples"] > 0]
    majority_correct = [r["majority_correct"]      for r in diversity_stats
                        if r["majority_correct"] is not None]

    summary = {
        "harm_source":    args.harm_source,
        "adapter_path":   str(args.adapter_path) if args.adapter_path else None,
        "base_model":     args.base_model if args.base_model else None,
        "from_samples":   str(args.from_samples) if args.from_samples else None,
        "temperature":    args.temperature,
        "n_prompts_total":              len(all_parsed_data),
        "n_prompts_with_valid_samples": sum(
            1 for r in diversity_stats if r["n_valid_samples"] > 0
        ),
        "n_dpo_pairs":          len(dpo_pairs),
        "total_completions":    total_completions,
        "parse_failures_total": parse_failures,
        "label_missing_total":  label_missing,
        "diversity": {
            "mean":     float(np.mean(all_diversity))   if all_diversity else 0.0,
            "median":   float(np.median(all_diversity)) if all_diversity else 0.0,
            "std":      float(np.std(all_diversity))    if all_diversity else 0.0,
            "min":      float(np.min(all_diversity))    if all_diversity else 0.0,
            "max":      float(np.max(all_diversity))    if all_diversity else 0.0,
            "n_prompts": len(all_diversity),
        },
        "label_agreement": {
            "mean": float(np.mean(all_agreement)) if all_agreement else 0.0,
            "fully_consistent_rate": float(
                sum(1 for a in all_agreement if a == 1.0) / len(all_agreement)
            ) if all_agreement else 0.0,
        },
        "majority_vote_accuracy": float(np.mean(majority_correct))
                                  if majority_correct else 0.0,
    }

    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary          -> {summary_path}")

    # ------------------------------------------------------------------
    # Phase 7: Readable summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("DIVERSITY ANALYSIS SUMMARY")
    print("=" * 65)
    print(f"Harm source            : {summary['harm_source']}")
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
    print(f"Diversity  mean={d['mean']:.4f}  median={d['median']:.4f}  "
          f"std={d['std']:.4f}  range=[{d['min']:.4f},{d['max']:.4f}]")
    la = summary["label_agreement"]
    print(f"Label agreement  mean={la['mean']:.4f}  "
          f"fully_consistent={la['fully_consistent_rate']:.2%}")
    print(f"Majority-vote accuracy : {summary['majority_vote_accuracy']:.4f}")
    print("=" * 65)

    _plot_histogram(all_diversity, output_dir / "diversity_histogram.png",
                    args.temperature)
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
        plt.axvline(mean_val, color="red", linestyle="--", linewidth=1.5,
                    label=f"Mean={mean_val:.3f}")
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

    # Generation
    p.add_argument("--adapter-path",   type=str,
                   default="trained_models/causal/hyperparam_sweep/lr_5e-05_e_5_adapter")
    p.add_argument("--base-model",     type=str,
                   default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--num-samples", "-k", type=int, default=5)
    p.add_argument("--temperature",    type=float, default=0.8)
    p.add_argument("--top-p",          type=float, default=0.95)
    p.add_argument("--max-new-tokens", type=int, default=64)

    # Harm source
    p.add_argument("--harm-source",
                   choices=["generated", "llm_t0", "modernbert"],
                   default="generated",
                   help="Where harm labels come from: "
                        "'generated' = parsed from the same LLM output (original), "
                        "'llm_t0'    = same LLM reclassifies each intent at temperature=0, "
                        "'modernbert'= trained ModernBERT classifier.")
    p.add_argument("--modernbert-path", type=str,
                   default="trained_models/bert_intent/ModernBERT-base",
                   help="Path to trained ModernBERT intent classifier "
                        "(used when --harm-source modernbert).")

    # Skip generation
    p.add_argument("--from-samples", type=str, default=None,
                   help="Path to parsed_samples.jsonl from a previous run. "
                        "When set, skips generation entirely (and skips vLLM "
                        "load unless --harm-source llm_t0).")

    # Data
    p.add_argument("--output-dir",  type=str, default="data/diversity_analysis")
    p.add_argument("--max-prompts", type=int, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_analysis(args)
