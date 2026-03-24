#!/usr/bin/env python3
"""
Check reasoning traces for ground-truth leakage using Qwen3.5-0.8B.

Leakage = the ground truth intent or harm label is explicitly stated at the
BEGINNING of the reasoning before the model has worked through the evidence.
Appearances at the END (as a conclusion) are acceptable.

Usage (from project root):
    python scripts/check_trace_leakage.py
"""

import json
from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR     = Path("data")
TRACE_DIR_V3 = DATA_DIR / "reasoning_traces_v3"
TRACE_DIR_V1 = DATA_DIR / "reasoning_traces"
OUTPUT_PATH  = DATA_DIR / "trace_leakage_results.json"

TRACE_MODELS = {
    "Gemma 3 27B (v3)":       TRACE_DIR_V3 / "RedHatAI-gemma-3-27b-it-quantized.w4a16",
    "Qwen3-32B (v3)":         TRACE_DIR_V3 / "RedHatAI-Qwen3-32B-quantized.w4a16",
    "Qwen3.5-27B BF16 (v3)":  TRACE_DIR_V3 / "Qwen-Qwen3.5-27B",
    "Qwen3.5-27B 4-bit (v3)": TRACE_DIR_V3 / "cyankiwi-Qwen3.5-27B-AWQ-4bit",
    "GPT-OSS-120B (v3)":      TRACE_DIR_V3 / "openai-gpt-oss-120b",
    "GPT-OSS-120B (v1)":      TRACE_DIR_V1 / "openai-gpt-oss-120b",
}

# ── Checker model ─────────────────────────────────────────────────────────────
CHECKER_MODEL    = "Qwen/Qwen3.5-0.8B"
REASONING_PREFIX = 800   # chars to examine for early leakage

SYSTEM_PROMPT = """\
You are a quality checker for AI reasoning traces used in model distillation.

A teacher model was given a user prompt and the ground truth harm label (and intent, \
where applicable) and asked to write a reasoning trace that justifies those labels. \
The trace is then used to train a student model — so it must read as genuine reasoning, \
not as a model that was simply told the answer.

Your task: decide whether the reasoning trace LEAKS the ground truth at the BEGINNING, \
meaning the model reveals it already knows the answer before working through the evidence.

Leakage looks like: immediately restating the ground truth label/intent verbatim, \
opening with "Since this is harmful..." or "The intent is X, therefore...", or \
referencing the answer in the first sentence without any prior analysis.

NOT leakage: the answer appearing naturally at the END after reasoning through the prompt, \
or the model noting something that is also obvious from the prompt itself (i.e. a conclusion \
that any reader would reach without knowing the ground truth in advance).

You are provided with the original user prompt so you can judge whether any early \
conclusion is self-evident from the prompt or only explicable by knowing the ground truth.

Reply with only YES (leakage detected) or NO (no leakage).\
"""


def build_user_message(entry: dict) -> str:
    gt     = entry["ground_truth"]
    prefix = entry["reasoning"][:REASONING_PREFIX]
    lines  = [
        f"User prompt:\n{entry['prompt']}",
        "",
        f"Ground truth harm label: {gt.get('prompt_harm_label', '')}",
    ]
    if entry["condition"] == "with_intent":
        lines.append(f"Ground truth intent: {gt.get('intent', '')}")
    lines += [
        "",
        "Beginning of reasoning trace:",
        prefix,
        "",
        "Does the reasoning leak the ground truth at the beginning? Answer YES or NO.",
    ]
    return "\n".join(lines)


def main():
    # ── Load traces ───────────────────────────────────────────────────────────
    all_entries = []   # list of (model_label, entry)
    for model_label, trace_dir in TRACE_MODELS.items():
        parsed_path = trace_dir / "parsed_results.json"
        if not parsed_path.exists():
            print(f"MISSING: {parsed_path}")
            continue
        with open(parsed_path) as f:
            data = json.load(f)
        before = len(all_entries)
        for entry in data:
            if entry.get("reasoning"):          # skip entries with empty reasoning
                all_entries.append((model_label, entry))
        print(f"  {model_label}: {len(all_entries) - before} entries")

    print(f"\nTotal entries to check: {len(all_entries)}")

    # ── Build prompts ─────────────────────────────────────────────────────────
    print(f"\nLoading tokenizer: {CHECKER_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(CHECKER_MODEL)

    prompts = []
    for model_label, entry in all_entries:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": build_user_message(entry)},
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,      # disable CoT in the checker itself
        )
        prompts.append(text)

    # ── Run vLLM ──────────────────────────────────────────────────────────────
    print(f"\nLoading checker model: {CHECKER_MODEL}")
    llm = LLM(
        model=CHECKER_MODEL,
        max_model_len=2048,
        dtype="bfloat16",
        enable_prefix_caching=True,
    )

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=5,
        structured_outputs=StructuredOutputsParams(choice=["YES", "NO"]),
    )

    print(f"Running inference on {len(prompts)} prompts...")
    outputs = llm.generate(prompts, sampling_params)

    # ── Aggregate results ─────────────────────────────────────────────────────
    results: dict[str, dict] = {}
    for (model_label, entry), output in zip(all_entries, outputs):
        answer  = output.outputs[0].text.strip().upper()
        leakage = answer == "YES"

        if model_label not in results:
            results[model_label] = {"entries": [], "n_leaked": 0, "n_total": 0}

        results[model_label]["entries"].append({
            "wildguard_id": entry["wildguard_id"],
            "condition":    entry["condition"],
            "leakage":      leakage,
        })
        results[model_label]["n_total"]  += 1
        results[model_label]["n_leaked"] += int(leakage)

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n=== Leakage Summary ===")
    for model_label, r in results.items():
        rate = r["n_leaked"] / r["n_total"] if r["n_total"] else 0
        print(f"  {model_label}: {r['n_leaked']}/{r['n_total']} ({rate:.1%})")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
