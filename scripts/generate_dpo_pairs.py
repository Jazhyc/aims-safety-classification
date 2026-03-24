"""
Generate DPO and contrastive training pairs from the annotated training split.

Pipeline per training prompt:
  1. chosen  = "Intent: {gold_intent}; Harm: {gold_harm}"
               (ground-truth from human annotation — never a model sample)
  2. Sample k completions at T=temperature from the SFT model.
  3. Parse the intent text from each sample (the T=0.8 harm label is discarded).
  4. Re-classify each generated intent at T=0 (greedy) for a reliable harm label.
  5. rejected = samples whose T=0 harm label ≠ gold_harm.

Outputs (all in --output-dir):
  parsed_samples.jsonl   — raw k samples per prompt (for debugging / re-use)
  dpo_pairs.jsonl        — one (chosen, rejected) pair per prompt
  contrastive_pairs.jsonl— one chosen + ALL rejected per prompt (InfoNCE)
  summary.json           — run statistics

Usage (from project root):
    python scripts/generate_dpo_pairs.py \\
        --adapter-path trained_models/causal/hyperparam_sweep/lr_5e-05_e_5_adapter \\
        --base-model   meta-llama/Llama-3.1-8B-Instruct \\
        --output-dir   data/dpo_pairs/train_t0.8 \\
        --num-samples  5 \\
        --temperature  0.8

    # If you already have parsed_samples.jsonl from a previous run, skip generation:
    python scripts/generate_dpo_pairs.py \\
        --from-samples data/dpo_pairs/train_t0.8/parsed_samples.jsonl \\
        --adapter-path trained_models/causal/hyperparam_sweep/lr_5e-05_e_5_adapter \\
        --base-model   meta-llama/Llama-3.1-8B-Instruct \\
        --output-dir   data/dpo_pairs/train_t0.8
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from intention_jailbreak.model_generation.preprocessing import preprocess_data
from intention_jailbreak.model_generation.data_utils import apply_binary_harm_mapping


# ---------------------------------------------------------------------------
# Helpers shared with intent_diversity_analysis.py
# ---------------------------------------------------------------------------

def norm_harm(s: str) -> Optional[str]:
    """Normalise a raw harm string to 'harmful' | 'safe' | None."""
    s = s.strip().lower().rstrip(".").rstrip(";")
    if s in {"safe", "harmless", "benign", "s", "b"}:
        return "safe"
    if s in {"harmful", "unsafe", "dangerous", "h", "u"}:
        return "harmful"
    return None


def extract_intent_and_harm(raw_text: str):
    """
    Parse intent and harm from a model completion.

    Handles all formats observed in prediction files:
      "Intent: <text>; Harm: harmful"
      "Intent: <text>\\nHarm: harmful"
      "Intent: <text> Harm: harmful"
      "<text>; Harm: harmful"        (missing Intent: prefix)
      "<text>"                       (intent only)

    Returns (intent_str, harm_str) — either may be None.
    """
    text = raw_text.strip()

    # Strip thinking tags (reasoning models)
    if "<think>" in text and "</think>" in text:
        text = text.split("</think>")[-1].strip()

    # Pattern 1: "Intent: <text>; Harm: <label>"
    m = re.search(r"Intent:\s*(.+?);\s*Harm:\s*(\S+)", text, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip(), norm_harm(m.group(2))

    # Pattern 2: "Intent: <text>\nHarm: <label>"
    m = re.search(r"Intent:\s*(.+?)\n\s*Harm:\s*(\S+)", text, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip().rstrip("."), norm_harm(m.group(2))

    # Pattern 3: "Intent: <text> Harm: <label>"
    m = re.search(r"Intent:\s*(.+?)\s+Harm:\s*(\S+)", text, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip().rstrip("."), norm_harm(m.group(2))

    # Pattern 4: "<text>; Harm: <label>"
    m = re.search(r"^(.+?);\s*Harm:\s*(\S+)", text, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip(), norm_harm(m.group(2))

    # Pattern 5: Harm label after semicolon or newline
    m = re.search(r"[;\n]\s*Harm:\s*(\S+)", text, re.IGNORECASE)
    if m:
        harm = norm_harm(m.group(1))
        intent = text[:m.start()].strip()
        intent = re.sub(r"^Intent:\s*", "", intent, flags=re.IGNORECASE).strip()
        return intent if intent else text, harm

    # Pattern 6: Intent only — no harm label
    m = re.search(r"^Intent:\s*(.+?)(?:;|$)", text, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip(), None

    return text if text else None, None


def classify_intents_llm_t0(llm, lora_request, intent_texts: list) -> list:
    """
    Re-classify intent texts with the SFT model at T=0 (greedy).

    Prompts the model with "Intent: {text}\\nHarm: " and reads the first token.
    This gives a deterministic, reliable harm label independent of the T=0.8
    generation that produced the intent.

    Returns a list of "harmful" | "safe" | None (one per input).
    """
    from vllm import SamplingParams

    prompts = [f"Intent: {t}\nHarm: " for t in intent_texts]
    params = SamplingParams(
        n=1,
        max_tokens=5,
        temperature=0.0,
        top_p=1.0,
        skip_special_tokens=True,
    )
    outputs = llm.generate(prompts, params, lora_request=lora_request)
    return [norm_harm(o.outputs[0].text.strip()) for o in outputs]


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def save_jsonl(records: list, path: Path):
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Saved {len(records):>6} records → {path}")


def load_jsonl(path: str) -> list:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"  Loaded {len(records)} records from {path}")
    return records


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load dataset — HF train split only
    # ------------------------------------------------------------------
    print("\n=== Loading dataset (HF train split) ===")
    train_dataset = preprocess_data(split="train")
    train_dataset = apply_binary_harm_mapping(train_dataset, binary_harm_mapping=True)

    print(f"  Train: {len(train_dataset)}")
    examples = list(train_dataset)

    # ------------------------------------------------------------------
    # 2. Load vLLM (needed for generation and / or T=0 reclassification)
    # ------------------------------------------------------------------
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    print(f"\n=== Loading model ===")
    print(f"  Base model   : {args.base_model}")
    print(f"  LoRA adapter : {args.adapter_path}")

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
    lora_request = LoRARequest("sft_lora", 1, args.adapter_path)

    # ------------------------------------------------------------------
    # 3. Generate k samples per prompt OR load from previous run
    # ------------------------------------------------------------------
    if args.from_samples:
        print(f"\n=== Loading parsed samples from {args.from_samples} ===")
        all_parsed = load_jsonl(args.from_samples)
        # Re-attach examples list in the same order
        examples = [
            {
                "id":            d["id"],
                "prompt":        d["prompt"],
                "intent":        d["gold_intent"],
                "Annotator Harm": d["gold_harm"],
            }
            for d in all_parsed
        ]
    else:
        sampling_params = SamplingParams(
            n=args.num_samples,
            max_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            skip_special_tokens=True,
        )

        print(
            f"\n=== Generating {args.num_samples} samples per prompt "
            f"(T={args.temperature}) ==="
        )
        prompts = [f"{ex['prompt']}\n" for ex in examples]
        outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)

        all_parsed = []
        n_parse_fail = 0
        for ex, output in zip(examples, outputs):
            samples = []
            for completion in output.outputs:
                raw = completion.text.strip()
                intent, _ = extract_intent_and_harm(raw)  # discard generated harm label
                if intent is None:
                    n_parse_fail += 1
                samples.append({"raw": raw, "intent": intent})

            all_parsed.append({
                "id":          ex.get("id", ""),
                "prompt":      ex["prompt"],
                "gold_intent": ex.get("intent", ""),
                "gold_harm":   ex.get("Annotator Harm"),
                "samples":     samples,
            })

        print(f"  Generation parse failures: {n_parse_fail}")
        save_jsonl(all_parsed, output_dir / "parsed_samples.jsonl")

    # ------------------------------------------------------------------
    # 4. Re-classify every generated intent at T=0 for reliable labels
    # ------------------------------------------------------------------
    print("\n=== Re-classifying intents at T=0 ===")

    flat_intents: list[str] = []
    flat_idx: list[tuple[int, int]] = []   # (prompt_idx, sample_idx)

    for i, d in enumerate(all_parsed):
        for j, s in enumerate(d["samples"]):
            if s["intent"]:
                flat_intents.append(s["intent"])
                flat_idx.append((i, j))

    print(f"  Classifying {len(flat_intents)} intent texts...")
    t0_labels = classify_intents_llm_t0(llm, lora_request, flat_intents)

    for (i, j), label in zip(flat_idx, t0_labels):
        all_parsed[i]["samples"][j]["harm_t0"] = label

    assigned = sum(1 for l in t0_labels if l is not None)
    print(f"  Labels assigned: {assigned} / {len(flat_intents)}")

    # ------------------------------------------------------------------
    # 5. Build DPO and contrastive pairs
    # ------------------------------------------------------------------
    print("\n=== Building pairs ===")

    dpo_pairs         = []   # one chosen / one rejected per prompt
    contrastive_pairs = []   # one chosen / ALL rejecteds per prompt

    n_no_gold      = 0   # prompts with missing gold harm
    n_no_negative  = 0   # prompts where all samples agree with gold
    n_no_valid     = 0   # prompts where no sample parsed correctly

    for d in all_parsed:
        gold_intent = d["gold_intent"]
        gold_harm   = d["gold_harm"]
        prompt      = d["prompt"]
        ex_id       = d["id"]

        if not gold_harm or not gold_intent:
            n_no_gold += 1
            continue

        # Only keep samples with both intent and a T=0 label
        valid = [
            s for s in d["samples"]
            if s.get("intent") and s.get("harm_t0") is not None
        ]

        if not valid:
            n_no_valid += 1
            continue

        # chosen is always the ground-truth annotation
        chosen_text = f"Intent: {gold_intent}; Harm: {gold_harm}"

        # rejected = generated samples whose T=0 label disagrees with gold
        rejecteds = [s for s in valid if s["harm_t0"] != gold_harm]

        if not rejecteds:
            n_no_negative += 1
            continue

        # ── DPO pair: one rejected (pick the one most confidently wrong,
        #    i.e. longest intent — proxy for well-formed output)
        rejected_dpo = max(rejecteds, key=lambda s: len(s["intent"]))
        dpo_pairs.append({
            "id":              ex_id,
            "prompt":          prompt,
            "gold_intent":     gold_intent,
            "gold_harm":       gold_harm,
            "chosen":          chosen_text,
            "rejected":        f"Intent: {rejected_dpo['intent']}; Harm: {rejected_dpo['harm_t0']}",
            "rejected_intent": rejected_dpo["intent"],
            "rejected_harm":   rejected_dpo["harm_t0"],
            "n_rejecteds":     len(rejecteds),
        })

        # ── Contrastive pair: all rejecteds
        contrastive_pairs.append({
            "id":          ex_id,
            "prompt":      prompt,
            "gold_intent": gold_intent,
            "gold_harm":   gold_harm,
            "chosen":      chosen_text,
            "rejecteds": [
                {
                    "text":   f"Intent: {s['intent']}; Harm: {s['harm_t0']}",
                    "intent": s["intent"],
                    "harm":   s["harm_t0"],
                }
                for s in rejecteds
            ],
        })

    # ------------------------------------------------------------------
    # 6. Save outputs
    # ------------------------------------------------------------------
    print("\n=== Saving outputs ===")
    save_jsonl(dpo_pairs,         output_dir / "dpo_pairs.jsonl")
    save_jsonl(contrastive_pairs, output_dir / "contrastive_pairs.jsonl")

    # ------------------------------------------------------------------
    # 7. Summary
    # ------------------------------------------------------------------
    n_total = len(all_parsed)
    n_dpo   = len(dpo_pairs)
    neg_counts = [len(p["rejecteds"]) for p in contrastive_pairs]

    summary = {
        "adapter_path":       args.adapter_path,
        "base_model":         args.base_model,
        "temperature":        args.temperature,
        "num_samples":        args.num_samples,
        "split":              "train",
        "n_prompts_total":    n_total,
        "n_skipped_no_gold":  n_no_gold,
        "n_skipped_no_valid": n_no_valid,
        "n_skipped_no_neg":   n_no_negative,
        "n_dpo_pairs":        n_dpo,
        "n_contrastive_pairs": len(contrastive_pairs),
        "negatives_per_prompt": {
            "mean":   float(np.mean(neg_counts))   if neg_counts else 0.0,
            "median": float(np.median(neg_counts)) if neg_counts else 0.0,
            "max":    int(np.max(neg_counts))      if neg_counts else 0,
            "dist":   {
                str(k): int(sum(1 for c in neg_counts if c == k))
                for k in sorted(set(neg_counts))
            },
        },
    }

    summary_path = output_dir / "summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved summary      → {summary_path}")

    print("\n" + "=" * 65)
    print("PAIR GENERATION SUMMARY")
    print("=" * 65)
    print(f"Split              : HF train split")
    print(f"Total prompts      : {n_total}")
    print(f"Skipped (no gold)  : {n_no_gold}")
    print(f"Skipped (no valid) : {n_no_valid}")
    print(f"Skipped (no neg)   : {n_no_negative}")
    print(f"DPO pairs          : {n_dpo}  ({100*n_dpo/max(n_total,1):.1f}%)")
    print(f"Contrastive pairs  : {len(contrastive_pairs)}")
    if neg_counts:
        print(f"Negatives/prompt   : mean={np.mean(neg_counts):.2f}  "
              f"median={np.median(neg_counts):.1f}  max={np.max(neg_counts)}")
        print(f"Distribution       : {summary['negatives_per_prompt']['dist']}")
    print("=" * 65)
    print(f"\nDone. All outputs in {output_dir}/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Model
    p.add_argument(
        "--adapter-path", type=str,
        default="trained_models/causal/hyperparam_sweep/lr_5e-05_e_5_adapter",
        help="Path to the SFT LoRA adapter.",
    )
    p.add_argument(
        "--base-model", type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="HuggingFace model ID for the base model.",
    )

    # Sampling
    p.add_argument("--num-samples", "-k", type=int, default=5,
                   help="Number of completions to sample per prompt at T=temperature.")
    p.add_argument("--temperature", type=float, default=0.8,
                   help="Sampling temperature for negative generation.")
    p.add_argument("--top-p",       type=float, default=0.95)
    p.add_argument("--max-new-tokens", type=int, default=64)

    # Re-use previous generation
    p.add_argument(
        "--from-samples", type=str, default=None,
        help="Path to parsed_samples.jsonl from a previous run. "
             "Skips the T=temperature generation step.",
    )

    # Output
    p.add_argument("--output-dir", type=str, default="data/dpo_pairs/train_t0.8")

    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
