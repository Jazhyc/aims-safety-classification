"""
Generate DPO pairs from an annotated HF split (default: train).

Pipeline per training prompt:
  1. chosen  = "Intent: {gold_intent}; Harm: {gold_harm}"
               (ground-truth from human annotation — never a model sample)
  2. Sample k completions at T=temperature from the SFT model.
  3. Parse intent + sampled harm from each T=0.8 sample.
  4. Re-classify each generated intent at T=0 (greedy) for a reliable harm label.
  5. rejected = samples whose T=0 harm label ≠ gold_harm.

Outputs (all in --output-dir):
  samples_t0p8_raw.jsonl        — raw k samples per prompt parsed at T=0.8 (intent + harm_t08)
  samples_t0p8_relabel_t0.jsonl — same samples after deterministic T=0 harm relabeling
  parsed_samples_t0p8.jsonl     — structured snapshot after T=0.8 parsing
  parsed_samples_t0.jsonl       — structured snapshot after deterministic T=0 relabeling
  parsed_samples_t0_judged.jsonl — structured snapshot after judge verdicts (--intent-filter only)
  parsed_samples.jsonl          — canonical latest snapshot (kept for compatibility)
  pairs_wrong_only.jsonl        — harm_t0 != gold_harm
  pairs_judge_bad_only.jsonl    — harm_t0 == gold_harm and judge_verdict == bad_match
  pairs_judge_bad_decent_only.jsonl — harm_t0 == gold_harm and judge_verdict in {bad_match,decent_match}
  pairs_wrong_plus_judge_bad.jsonl — union of wrong_only and judge_bad_only
  pairs_wrong_plus_judge_bad_decent.jsonl — union of wrong_only and judge_bad_decent_only
  dpo_pairs.jsonl               — legacy alias to the active training set for this run
  dpo_pairs_decent.jsonl        — legacy alias to decent variant when available
  summary.json              — run statistics
  intent_filter_examples.jsonl — per-sample judge verdicts (only with --intent-filter)

Usage (from project root):
    python scripts/dpo/generate_dpo_pairs.py \\
        --adapter-path Jazhyc/llama-3.1-8b-sft-generation \\
        --base-model   meta-llama/Llama-3.1-8B-Instruct \\
        --output-dir   data/dpo_pairs/train_t0.8 \\
        --num-samples  5 \\
        --temperature  0.8

    # Step 1 — generate samples + T=0 labels (SFT 8B):
    python scripts/dpo/generate_dpo_pairs.py \\
        --adapter-path Jazhyc/llama-3.1-8b-sft-generation \\
        --base-model   meta-llama/Llama-3.1-8B-Instruct \\
        --output-dir   data/dpo_pairs/train_t0.8

    # Step 2 — judge-only pass with a larger model (SFT not loaded):
    python scripts/dpo/generate_dpo_pairs.py \\
        --from-samples data/dpo_pairs/train_t0.8/parsed_samples.jsonl \\
        --intent-filter \\
        --judge-model  meta-llama/Llama-3.1-70B-Instruct \\
        --output-dir   data/dpo_pairs/train_t0.8_filtered

    # Single-pass with same model as judge (original behaviour):
    python scripts/dpo/generate_dpo_pairs.py \\
        --from-samples data/dpo_pairs/train_t0.8/parsed_samples.jsonl \\
        --intent-filter \\
        --adapter-path Jazhyc/llama-3.1-8b-sft-generation \\
        --base-model   meta-llama/Llama-3.1-8B-Instruct \\
        --output-dir   data/dpo_pairs/train_t0.8
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from intention_jailbreak.model_generation.parsing import (
    extract_intent_and_harm,
    normalize_harm_label_binary,
)
from intention_jailbreak.model_generation.prompt_templates import (
    DPO_JUDGE_INTENT_SYSTEM_PROMPT,
    GENERATION_SYSTEM_PROMPT,
)
from intention_jailbreak.model_generation.preprocessing import preprocess_data
from intention_jailbreak.model_generation.data_utils import apply_binary_harm_mapping


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RAW_SAMPLES_FILENAME = "samples_t0p8_raw.jsonl"
RELABELED_SAMPLES_FILENAME = "samples_t0p8_relabel_t0.jsonl"
PARSED_T0P8_FILENAME = "parsed_samples_t0p8.jsonl"
PARSED_T0_FILENAME = "parsed_samples_t0.jsonl"
PARSED_JUDGED_FILENAME = "parsed_samples_t0_judged.jsonl"
CANONICAL_SAMPLES_FILENAME = "parsed_samples.jsonl"  # backward-compatible alias
PAIRS_WRONG_ONLY_FILENAME = "pairs_wrong_only.jsonl"
PAIRS_JUDGE_BAD_ONLY_FILENAME = "pairs_judge_bad_only.jsonl"
PAIRS_JUDGE_BAD_DECENT_ONLY_FILENAME = "pairs_judge_bad_decent_only.jsonl"
PAIRS_WRONG_PLUS_JUDGE_BAD_FILENAME = "pairs_wrong_plus_judge_bad.jsonl"
PAIRS_WRONG_PLUS_JUDGE_BAD_DECENT_FILENAME = "pairs_wrong_plus_judge_bad_decent.jsonl"
DEFAULT_SFT_ADAPTER = "Jazhyc/llama-3.1-8b-sft-generation"


@dataclass
class LoadedModels:
    judge_model_id: str
    need_sft: bool
    need_judge: bool
    llm: object | None
    lora_request: object | None
    sft_tokenizer: object | None
    judge_llm: object | None
    judge_tokenizer: object | None


def _apply_chat_template(tokenizer, messages: list[dict]) -> str:
    """Apply chat template with optional enable_thinking=False for Qwen3 models."""
    try:
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )


def classify_intents_llm_t0(llm, lora_request, intent_texts: list, tokenizer) -> list:
    """
    Re-classify intent texts with the SFT model at T=0 (greedy).

    The SFT model always generates the full "Intent: ...; Harm: <label>" format,
    so we give it enough tokens to complete that format and parse the harm label
    with extract_intent_and_harm rather than truncating to 5 tokens.

    Returns a list of "harmful" | "safe" | None (one per input).
    """
    from vllm import SamplingParams

    prompts = [
        _apply_chat_template(
            tokenizer,
            [
                {"role": "system", "content": GENERATION_SYSTEM_PROMPT},
                {"role": "user", "content": f"Intent: {t}\nHarm: "},
            ],
        )
        for t in intent_texts
    ]
    params = SamplingParams(
        n=1,
        max_tokens=64,
        temperature=0.0,
        top_p=1.0,
        skip_special_tokens=True,
    )
    outputs = llm.generate(prompts, params, lora_request=lora_request)
    results = []
    for o in outputs:
        _, harm = extract_intent_and_harm(o.outputs[0].text.strip())
        results.append(harm)
    return results


def _all_have_t0(parsed: list) -> bool:
    """Return True if T=0 classification has already run.

    Checks for key presence rather than non-None value: after classification,
    every sample with a parsed intent has a 'harm_t0' key (value may be None
    when the model output couldn't be parsed — that still counts as classified).
    """
    for d in parsed:
        for s in d.get("samples", []):
            if s.get("intent") and "harm_t0" not in s:
                return False
    return True


def _all_have_judge_verdict(parsed: list) -> bool:
    """Return True if judge verdicts are already saved in the parsed samples.

    Only checks samples that would be judged: those with intent and harm_t0
    matching gold_harm. If all such samples already have 'judge_verdict', the
    judge inference step can be skipped entirely.
    """
    for d in parsed:
        gold_harm = d.get("gold_harm")
        if not gold_harm or not d.get("gold_intent"):
            continue
        for s in d.get("samples", []):
            if s.get("intent") and s.get("harm_t0") == gold_harm:
                if "judge_verdict" not in s:
                    return False
    return True


def parse_verdict(text: str, reasoning: str = "") -> str:
    """Parse a judge verdict string into one of: good_match | decent_match | bad_match | unknown.

    Searches the full output (text + reasoning_content for thinking models like GPT-OSS)
    for the verdict keyword anywhere, not just at the start.
    """
    combined = (text + " " + reasoning).strip().lower()
    if "good_match"   in combined: return "good_match"
    if "decent_match" in combined: return "decent_match"
    if "bad_match"    in combined: return "bad_match"
    return "unknown"


def judge_intent_similarity_llm(llm, triples: list, tokenizer) -> list:
    """
    Use an LLM judge (no LoRA adapter) to assess whether each generated intent
    is a valid annotation of the prompt from a safety perspective.

    Uses tokenizer.apply_chat_template so the prompt format is model-agnostic
    (Llama, Gemma, Qwen, etc.).  Pass a separate judge model + tokenizer to
    avoid SFT adapter bias and to allow using a larger judge (e.g. 70B).

    Args:
        llm:      vLLM LLM instance for the judge model (no LoRA).
        triples:  list of (prompt, gold_intent, generated_intent).
                  gold_intent is a human-annotated reference — a valid example
                  of what a correct annotation looks like.
        tokenizer: HuggingFace tokenizer for the judge model, used to apply
                   the correct chat template.

    Returns:
        list of str — one of "good_match" | "decent_match" | "bad_match" | "unknown"
        per input triple.
    """
    from vllm import SamplingParams

    prompts = [
        _apply_chat_template(
            tokenizer,
            [
                {"role": "system", "content": DPO_JUDGE_INTENT_SYSTEM_PROMPT},
                {"role": "user", "content": (
                    f"User prompt:\n{prompt}\n\n"
                    f"Reference intent (valid example):\n{gold}\n\n"
                    f"Generated intent (evaluate this):\n{generated}\n\n"
                    f"Verdict:"
                )},
            ],
        )
        for prompt, gold, generated in triples
    ]
    params = SamplingParams(
        n=1,
        max_tokens=512,
        temperature=0.0,
        top_p=1.0,
        skip_special_tokens=True,
    )
    outputs = llm.generate(prompts, params)
    return [
        parse_verdict(
            o.outputs[0].text,
            getattr(o.outputs[0], "reasoning_content", "") or "",
        )
        for o in outputs
    ]



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


def _load_examples(args) -> list[dict]:
    """Load prompt examples from HF splits or a custom JSONL."""
    if args.from_jsonl:
        print(f"\n=== Loading prompts from {args.from_jsonl} ===")
        with open(args.from_jsonl, encoding="utf-8") as fh:
            raw_records = [json.loads(line) for line in fh if line.strip()]
        examples = [
            {"id": r["id"], "prompt": r["prompt"], "intent": None, "Annotator Harm": r["gold_harm"]}
            for r in raw_records
        ]
        print(f"  Loaded {len(examples)} prompts")
        return examples

    split = args.split
    if split == "val_and_test":
        print("\n=== Loading dataset (HF validation + test splits combined) ===")
        val_dataset  = preprocess_data(split="validation")
        test_dataset = preprocess_data(split="test")
        val_dataset  = apply_binary_harm_mapping(val_dataset,  binary_harm_mapping=True)
        test_dataset = apply_binary_harm_mapping(test_dataset, binary_harm_mapping=True)
        print(f"  Validation: {len(val_dataset)}  Test: {len(test_dataset)}")
        return list(val_dataset) + list(test_dataset)

    print(f"\n=== Loading dataset (HF {split} split) ===")
    dataset = preprocess_data(split=split)
    dataset = apply_binary_harm_mapping(dataset, binary_harm_mapping=True)
    print(f"  {split.capitalize()}: {len(dataset)}")
    return list(dataset)


def _resolve_local_model_path(model_id: str) -> str:
    """Resolve a HF model ID to a local snapshot path when available."""
    from huggingface_hub import snapshot_download

    try:
        return snapshot_download(model_id, local_files_only=True)
    except Exception:
        return model_id


def _resolve_local_adapter_path(adapter_ref: str, revision: Optional[str] = None) -> str:
    """Resolve adapter source (local path or HF repo) to a local directory."""
    p = Path(adapter_ref).expanduser()
    if p.exists():
        return str(p.resolve())

    from huggingface_hub import snapshot_download
    try:
        return snapshot_download(adapter_ref, revision=revision)
    except Exception as e:
        raise RuntimeError(
            f"Failed to resolve adapter '{adapter_ref}'"
            + (f" at revision '{revision}'" if revision else "")
            + f": {e}"
        ) from e


def _load_models(args, all_parsed: Optional[list]):
    """Load SFT and/or judge models depending on cached data and flags."""
    from vllm import LLM
    from vllm.lora.request import LoRARequest
    from transformers import AutoTokenizer

    judge_model_id = args.judge_model or args.base_model
    need_sft = (all_parsed is None) or (not _all_have_t0(all_parsed))
    need_judge = args.intent_filter and (all_parsed is None or not _all_have_judge_verdict(all_parsed))
    use_separate_judge = need_judge and (judge_model_id != args.base_model)

    llm = lora_request = sft_tokenizer = None
    if need_sft:
        local_adapter = _resolve_local_adapter_path(args.adapter_path, args.adapter_revision)
        local_base = _resolve_local_model_path(args.base_model)
        print(f"\n=== Loading SFT model ===")
        print(f"  Base model   : {args.base_model}")
        print(f"  Local path   : {local_base}")
        print(f"  LoRA adapter : {args.adapter_path}")
        print(f"  Local adapter: {local_adapter}")
        llm = LLM(
            model=local_base,
            tokenizer=local_base,
            enable_lora=True,
            max_lora_rank=64,
            max_loras=1,
            limit_mm_per_prompt={"image": 0},
            gpu_memory_utilization=0.90,
            max_model_len=args.max_model_len,
            dtype="bfloat16",
            enforce_eager=True,
        )
        lora_request = LoRARequest("sft_lora", 1, local_adapter)
        sft_tokenizer = AutoTokenizer.from_pretrained(local_base)

    judge_llm = judge_tokenizer = None
    if need_judge:
        if use_separate_judge:
            local_judge = _resolve_local_model_path(judge_model_id)
            print(f"\n=== Loading judge model ===")
            print(f"  Judge model  : {judge_model_id}")
            print(f"  Local path   : {local_judge}")
            print(f"  Judge GPUs   : {args.judge_tensor_parallel}")
            judge_llm = LLM(
                model=local_judge,
                gpu_memory_utilization=0.90,
                max_model_len=4096,
                dtype="auto",
                enforce_eager=True,
                tensor_parallel_size=args.judge_tensor_parallel,
            )
        elif llm is not None:
            judge_llm = llm
        else:
            local_judge = _resolve_local_model_path(judge_model_id)
            print(f"\n=== Loading judge model (base, no LoRA) ===")
            print(f"  Judge model  : {judge_model_id}")
            print(f"  Local path   : {local_judge}")
            judge_llm = LLM(
                model=local_judge,
                gpu_memory_utilization=0.90,
                max_model_len=4096,
                dtype="auto",
                enforce_eager=True,
                tensor_parallel_size=args.judge_tensor_parallel,
            )

        local_judge_path = _resolve_local_model_path(judge_model_id)
        judge_tokenizer = AutoTokenizer.from_pretrained(local_judge_path)

    return LoadedModels(
        judge_model_id=judge_model_id,
        need_sft=need_sft,
        need_judge=need_judge,
        llm=llm,
        lora_request=lora_request,
        sft_tokenizer=sft_tokenizer,
        judge_llm=judge_llm,
        judge_tokenizer=judge_tokenizer,
    )


def _generate_or_attach_samples(
    args,
    examples: list[dict],
    all_parsed: Optional[list],
    llm,
    lora_request,
    sft_tokenizer,
    output_dir: Path,
):
    """Generate k samples per prompt, or re-attach examples from cached parsed samples."""
    from vllm import SamplingParams

    if all_parsed is not None:
        examples = [
            {
                "id": d["id"],
                "prompt": d["prompt"],
                "intent": d["gold_intent"],
                "Annotator Harm": d["gold_harm"],
            }
            for d in all_parsed
        ]
        return examples, all_parsed

    if llm is None or lora_request is None or sft_tokenizer is None:
        raise RuntimeError(
            "SFT model + tokenizer are required to generate fresh samples. "
            "Pass --from-samples to reuse cached parsed samples."
        )

    sampling_params = SamplingParams(
        n=args.num_samples,
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        skip_special_tokens=True,
    )

    print(f"\n=== Generating {args.num_samples} samples per prompt (T={args.temperature}) ===")
    prompts = [
        _apply_chat_template(
            sft_tokenizer,
            [
                {"role": "system", "content": GENERATION_SYSTEM_PROMPT},
                {"role": "user", "content": ex["prompt"]},
            ],
        )
        for ex in examples
    ]
    outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)

    parsed = []
    n_parse_fail = 0
    for ex, output in zip(examples, outputs):
        samples = []
        for completion in output.outputs:
            raw = completion.text.strip()
            intent, sampled_harm = extract_intent_and_harm(raw)
            if intent is None:
                n_parse_fail += 1
            samples.append(
                {
                    "raw": raw,
                    "intent": intent,
                    # Keep sampled T=0.8 harm for reproducibility + switch analysis vs T=0.
                    "harm_t08": normalize_harm_label_binary(sampled_harm, safe_label="safe"),
                }
            )

        parsed.append(
            {
                "id": ex.get("id", ""),
                "prompt": ex["prompt"],
                "gold_intent": ex.get("intent", ""),
                "gold_harm": ex.get("Annotator Harm"),
                "samples": samples,
            }
        )

    print(f"  Generation parse failures: {n_parse_fail}")
    save_jsonl(parsed, output_dir / RAW_SAMPLES_FILENAME)
    save_jsonl(parsed, output_dir / PARSED_T0P8_FILENAME)
    save_jsonl(parsed, output_dir / CANONICAL_SAMPLES_FILENAME)
    return examples, parsed


def _classify_harm_t0_if_needed(
    all_parsed: list,
    need_sft: bool,
    llm,
    lora_request,
    sft_tokenizer,
    output_dir: Path,
) -> None:
    """Populate harm_t0 labels for parsed intents if needed."""
    if not need_sft:
        print("\n=== Skipping T=0 reclassification (harm_t0 already cached) ===")
        save_jsonl(all_parsed, output_dir / PARSED_T0_FILENAME)
        save_jsonl(all_parsed, output_dir / CANONICAL_SAMPLES_FILENAME)
        return

    if llm is None or lora_request is None or sft_tokenizer is None:
        raise RuntimeError(
            "SFT model + tokenizer are required for T=0 relabeling but were not loaded."
        )

    print("\n=== Re-classifying intents at T=0 ===")
    flat_intents: list[str] = []
    flat_idx: list[tuple[int, int]] = []
    for i, d in enumerate(all_parsed):
        for j, s in enumerate(d["samples"]):
            if s["intent"]:
                flat_intents.append(s["intent"])
                flat_idx.append((i, j))

    print(f"  Classifying {len(flat_intents)} intent texts...")
    t0_labels = classify_intents_llm_t0(llm, lora_request, flat_intents, sft_tokenizer)
    for (i, j), label in zip(flat_idx, t0_labels):
        all_parsed[i]["samples"][j]["harm_t0"] = label

    assigned = sum(1 for l in t0_labels if l is not None)
    print(f"  Labels assigned: {assigned} / {len(flat_intents)}")
    save_jsonl(all_parsed, output_dir / RELABELED_SAMPLES_FILENAME)
    save_jsonl(all_parsed, output_dir / PARSED_T0_FILENAME)
    save_jsonl(all_parsed, output_dir / CANONICAL_SAMPLES_FILENAME)


def _compute_relabel_stats(all_parsed: list[dict]) -> dict:
    """Compute agreement/switch stats between sampled T=0.8 harm and relabeled T=0 harm."""
    n_total_samples = 0
    n_with_intent = 0
    n_with_t08_harm = 0
    n_with_t0_harm = 0
    n_comparable = 0
    n_switched = 0
    n_unchanged = 0

    for d in all_parsed:
        for s in d.get("samples", []):
            n_total_samples += 1
            if s.get("intent"):
                n_with_intent += 1
            h08 = s.get("harm_t08")
            h0 = s.get("harm_t0")
            if h08 is not None:
                n_with_t08_harm += 1
            if h0 is not None:
                n_with_t0_harm += 1
            if h08 is not None and h0 is not None:
                n_comparable += 1
                if h08 != h0:
                    n_switched += 1
                else:
                    n_unchanged += 1

    return {
        "n_samples_total": n_total_samples,
        "n_samples_with_intent": n_with_intent,
        "n_samples_with_harm_t08": n_with_t08_harm,
        "n_samples_with_harm_t0": n_with_t0_harm,
        "n_samples_comparable_t08_t0": n_comparable,
        "n_harm_label_switched_t08_to_t0": n_switched,
        "n_harm_label_unchanged_t08_to_t0": n_unchanged,
        "pct_harm_label_switched_t08_to_t0": (100.0 * n_switched / n_comparable) if n_comparable else 0.0,
    }


def _build_dpo_pairs(all_parsed: list, args):
    """Construct explicit pair sets, plus legacy training-set selections."""
    pairs_wrong_only = []
    pairs_judge_bad_only = []
    pairs_judge_bad_decent_only = []
    pairs_wrong_plus_judge_bad = []
    pairs_wrong_plus_judge_bad_decent = []

    n_no_gold_harm = 0
    n_no_gold_intent = 0
    n_no_negative = 0
    n_no_valid = 0
    n_intent_rejecteds = 0
    n_intent_rejecteds_decent = 0
    n_hard_rejecteds = 0

    def _make_dpo_pairs(ex_id, prompt, chosen_t, chosen_i, gold_harm, rej_list):
        """Build one DPO row per rejected candidate for a prompt."""
        n_rejecteds_total = len(rej_list)
        return [
            {
                "id": ex_id,
                "prompt": prompt,
                "gold_intent": chosen_i,
                "gold_harm": gold_harm,
                "chosen": chosen_t,
                "rejected": f"Intent: {rejected['intent']}; Harm: {rejected['harm_t0']}",
                "rejected_intent": rejected["intent"],
                "rejected_harm": rejected["harm_t0"],
                "n_rejecteds": n_rejecteds_total,
            }
            for rejected in rej_list
        ]

    for d in all_parsed:
        gold_intent = d["gold_intent"]
        gold_harm = d["gold_harm"]
        prompt = d["prompt"]
        ex_id = d["id"]

        if not gold_harm:
            n_no_gold_harm += 1
            continue

        # Chosen must always come from the human annotation.
        # If gold intent is missing, skip this prompt entirely.
        if not gold_intent:
            n_no_gold_intent += 1
            continue

        valid = [s for s in d["samples"] if s.get("intent") and s.get("harm_t0") is not None]
        if not valid:
            n_no_valid += 1
            continue

        chosen_intent_str = gold_intent
        chosen_text = f"Intent: {gold_intent}; Harm: {gold_harm}"

        hard_rejecteds = [] if args.judge_only else [s for s in valid if s["harm_t0"] != gold_harm]
        n_hard_rejecteds += len(hard_rejecteds)
        if hard_rejecteds:
            pairs_wrong_only.extend(
                _make_dpo_pairs(ex_id, prompt, chosen_text, chosen_intent_str, gold_harm, hard_rejecteds)
            )

        judge_bad_rejecteds = []
        intent_rejecteds_decent = []

        if args.intent_filter:
            judge_bad_rejecteds = [
                s for s in valid if s["harm_t0"] == gold_harm and s.get("same_intent") is False
            ]
            if judge_bad_rejecteds:
                n_intent_rejecteds += len(judge_bad_rejecteds)
                pairs_judge_bad_only.extend(
                    _make_dpo_pairs(
                        ex_id, prompt, chosen_text, chosen_intent_str, gold_harm, judge_bad_rejecteds
                    )
                )

            intent_rejecteds_decent = [
                s
                for s in valid
                if s["harm_t0"] == gold_harm and s.get("judge_verdict") in {"bad_match", "decent_match"}
            ]
            if intent_rejecteds_decent:
                n_intent_rejecteds_decent += len(intent_rejecteds_decent)
                pairs_judge_bad_decent_only.extend(
                    _make_dpo_pairs(
                        ex_id, prompt, chosen_text, chosen_intent_str, gold_harm, intent_rejecteds_decent
                    )
                )

        rejecteds_union_bad = hard_rejecteds + judge_bad_rejecteds
        rejecteds_union_bad_decent = hard_rejecteds + intent_rejecteds_decent

        if rejecteds_union_bad:
            pairs_wrong_plus_judge_bad.extend(
                _make_dpo_pairs(
                    ex_id, prompt, chosen_text, chosen_intent_str, gold_harm, rejecteds_union_bad
                )
            )
        if rejecteds_union_bad_decent:
            pairs_wrong_plus_judge_bad_decent.extend(
                _make_dpo_pairs(
                    ex_id, prompt, chosen_text, chosen_intent_str, gold_harm, rejecteds_union_bad_decent
                )
            )

        if args.intent_filter:
            active_rejecteds = judge_bad_rejecteds if args.judge_only else rejecteds_union_bad
        else:
            active_rejecteds = hard_rejecteds
        if not active_rejecteds:
            n_no_negative += 1

    if args.intent_filter:
        dpo_pairs = pairs_judge_bad_only if args.judge_only else pairs_wrong_plus_judge_bad
        dpo_pairs_decent = (
            pairs_judge_bad_decent_only if args.judge_only else pairs_wrong_plus_judge_bad_decent
        )
    else:
        dpo_pairs = pairs_wrong_only
        dpo_pairs_decent = []

    pair_sets = {
        "wrong_only": pairs_wrong_only,
        "judge_bad_only": pairs_judge_bad_only,
        "judge_bad_decent_only": pairs_judge_bad_decent_only,
        "wrong_plus_judge_bad": pairs_wrong_plus_judge_bad,
        "wrong_plus_judge_bad_decent": pairs_wrong_plus_judge_bad_decent,
        "legacy_main": dpo_pairs,
        "legacy_decent": dpo_pairs_decent,
    }

    counts = {
        "n_no_gold_harm": n_no_gold_harm,
        "n_no_gold_intent": n_no_gold_intent,
        "n_no_negative": n_no_negative,
        "n_no_valid": n_no_valid,
        "n_hard_rejecteds": n_hard_rejecteds,
        "n_intent_rejecteds": n_intent_rejecteds,
        "n_intent_rejecteds_decent": n_intent_rejecteds_decent,
        "n_rows_wrong_only": len(pairs_wrong_only),
        "n_rows_judge_bad_only": len(pairs_judge_bad_only),
        "n_rows_judge_bad_decent_only": len(pairs_judge_bad_decent_only),
        "n_rows_wrong_plus_judge_bad": len(pairs_wrong_plus_judge_bad),
        "n_rows_wrong_plus_judge_bad_decent": len(pairs_wrong_plus_judge_bad_decent),
    }
    return pair_sets, counts


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    examples = _load_examples(args)
    all_parsed = None
    if args.from_samples:
        print(f"\n=== Loading parsed samples from {args.from_samples} ===")
        all_parsed = load_jsonl(args.from_samples)

    models = _load_models(args, all_parsed)

    examples, all_parsed = _generate_or_attach_samples(
        args=args,
        examples=examples,
        all_parsed=all_parsed,
        llm=models.llm,
        lora_request=models.lora_request,
        sft_tokenizer=models.sft_tokenizer,
        output_dir=output_dir,
    )

    _classify_harm_t0_if_needed(
        all_parsed=all_parsed,
        need_sft=models.need_sft,
        llm=models.llm,
        lora_request=models.lora_request,
        sft_tokenizer=models.sft_tokenizer,
        output_dir=output_dir,
    )

    # ------------------------------------------------------------------
    # 4b. Intent filter: judge correct-label samples for intent validity
    # ------------------------------------------------------------------
    filter_examples = []   # for intent_filter_examples.jsonl

    if args.intent_filter and not models.need_judge:
        print("\n=== Skipping judge inference (verdicts already cached in parsed_samples) ===")
        # Restore same_intent from cached judge_verdict
        for d in all_parsed:
            for s in d.get("samples", []):
                if "judge_verdict" in s:
                    s["same_intent"] = (s["judge_verdict"] != "bad_match")

    if models.need_judge:
        # Collect all samples where harm_t0 == gold_harm
        filter_triples = []   # (prompt, gold_intent, generated_intent)
        filter_idx     = []   # (prompt_idx, sample_idx)

        for i, d in enumerate(all_parsed):
            gold_intent = d.get("gold_intent", "")
            gold_harm   = d.get("gold_harm")
            if not gold_harm or not gold_intent:
                continue
            for j, s in enumerate(d["samples"]):
                if s.get("intent") and s.get("harm_t0") == gold_harm:
                    filter_triples.append((d["prompt"], gold_intent, s["intent"]))
                    filter_idx.append((i, j))

        print(f"\n  Judging {len(filter_triples)} correct-label samples for intent validity...")

        print(f"\n=== LLM judge ({models.judge_model_id}) ===")
        verdicts = judge_intent_similarity_llm(
            models.judge_llm,
            filter_triples,
            models.judge_tokenizer,
        )
        for (i, j), verdict in zip(filter_idx, verdicts):
            all_parsed[i]["samples"][j]["judge_verdict"] = verdict

        n_bad = sum(1 for v in verdicts if v == "bad_match")
        n_decent = sum(1 for v in verdicts if v == "decent_match")
        n_good = sum(1 for v in verdicts if v == "good_match")
        print(f"  good_match: {n_good}  decent_match: {n_decent}  bad_match: {n_bad}  "
              f"unknown: {len(verdicts) - n_good - n_decent - n_bad}")

        # bad_match → add as rejected; good/decent → keep as chosen
        for (i, j) in filter_idx:
            s = all_parsed[i]["samples"][j]
            s["same_intent"] = (s.get("judge_verdict") != "bad_match")

        # Persist judge_verdict in parsed_samples.jsonl so --union-judge-dir
        # can load it without re-running the judge.
        save_jsonl(all_parsed, output_dir / CANONICAL_SAMPLES_FILENAME)

        # Build inspection examples
        for (i, j), (prompt, gold_intent, gen_intent) in zip(filter_idx, filter_triples):
            s = all_parsed[i]["samples"][j]
            d = all_parsed[i]
            filter_examples.append({
                "id":               d["id"],
                "prompt":           prompt,
                "gold_intent":      gold_intent,
                "gold_harm":        d["gold_harm"],
                "generated_intent": gen_intent,
                "harm_t0":          s["harm_t0"],
                "judge_verdict":    s.get("judge_verdict"),
                "added_as_rejected": not s["same_intent"],
            })

    # ------------------------------------------------------------------
    # 4c. Union judge: merge bad_match verdicts from a second judge run.
    #     A sample becomes same_intent=False if EITHER judge said bad_match.
    #     Requires both runs to share the same canonical parsed_samples
    #     (same IDs in same order — guaranteed when both use --from-samples).
    # ------------------------------------------------------------------
    if args.union_judge_dir and args.intent_filter:
        union_path = Path(args.union_judge_dir) / "parsed_samples.jsonl"
        print(f"\n=== Merging judge verdicts from {union_path} ===")
        union_parsed = load_jsonl(str(union_path))

        # Verify alignment
        if len(union_parsed) != len(all_parsed):
            raise ValueError(
                f"union-judge-dir has {len(union_parsed)} records but current run has "
                f"{len(all_parsed)} — files must share the same canonical samples."
            )
        mismatched_ids = sum(1 for a, b in zip(all_parsed, union_parsed) if a["id"] != b["id"])
        if mismatched_ids:
            raise ValueError(f"{mismatched_ids} record IDs differ — files are not aligned.")

        n_union_added = 0
        for i, (main, other) in enumerate(zip(all_parsed, union_parsed)):
            for j, (sm, so) in enumerate(zip(main["samples"], other["samples"])):
                other_verdict = so.get("judge_verdict")
                if other_verdict == "bad_match" and sm.get("same_intent") is not False:
                    sm["same_intent"] = False
                    n_union_added += 1

        print(f"  Extra bad_match from second judge: {n_union_added}")

    # Persist explicit judged-stage snapshot when intent filter is active.
    if args.intent_filter:
        save_jsonl(all_parsed, output_dir / PARSED_JUDGED_FILENAME)
        save_jsonl(all_parsed, output_dir / CANONICAL_SAMPLES_FILENAME)

    # ------------------------------------------------------------------
    # 5. Build DPO pairs
    # ------------------------------------------------------------------
    print("\n=== Building pairs ===")
    pair_sets, counts = _build_dpo_pairs(all_parsed, args)
    dpo_pairs = pair_sets["legacy_main"]
    dpo_pairs_decent = pair_sets["legacy_decent"]
    n_no_gold_harm = counts["n_no_gold_harm"]
    n_no_gold_intent = counts["n_no_gold_intent"]
    n_no_negative = counts["n_no_negative"]
    n_no_valid = counts["n_no_valid"]
    n_intent_rejecteds = counts["n_intent_rejecteds"]
    n_intent_rejecteds_decent = counts["n_intent_rejecteds_decent"]
    n_hard_rejecteds = counts["n_hard_rejecteds"]
    n_rows_wrong_only = counts["n_rows_wrong_only"]
    n_rows_judge_bad_only = counts["n_rows_judge_bad_only"]
    n_rows_judge_bad_decent_only = counts["n_rows_judge_bad_decent_only"]
    n_rows_wrong_plus_judge_bad = counts["n_rows_wrong_plus_judge_bad"]
    n_rows_wrong_plus_judge_bad_decent = counts["n_rows_wrong_plus_judge_bad_decent"]

    # ------------------------------------------------------------------
    # 6. Save outputs
    # ------------------------------------------------------------------
    print("\n=== Saving outputs ===")
    save_jsonl(pair_sets["wrong_only"], output_dir / PAIRS_WRONG_ONLY_FILENAME)
    save_jsonl(pair_sets["judge_bad_only"], output_dir / PAIRS_JUDGE_BAD_ONLY_FILENAME)
    save_jsonl(pair_sets["judge_bad_decent_only"], output_dir / PAIRS_JUDGE_BAD_DECENT_ONLY_FILENAME)
    save_jsonl(pair_sets["wrong_plus_judge_bad"], output_dir / PAIRS_WRONG_PLUS_JUDGE_BAD_FILENAME)
    save_jsonl(
        pair_sets["wrong_plus_judge_bad_decent"],
        output_dir / PAIRS_WRONG_PLUS_JUDGE_BAD_DECENT_FILENAME,
    )

    # Backward-compatible aliases used by the existing pipeline.
    save_jsonl(dpo_pairs, output_dir / "dpo_pairs.jsonl")
    if args.intent_filter:
        save_jsonl(dpo_pairs_decent, output_dir / "dpo_pairs_decent.jsonl")
    if filter_examples:
        save_jsonl(filter_examples, output_dir / "intent_filter_examples.jsonl")

    # ------------------------------------------------------------------
    # 7. Summary
    # ------------------------------------------------------------------
    n_total = len(all_parsed)
    n_dpo   = len(dpo_pairs)
    relabel_stats = _compute_relabel_stats(all_parsed)

    artifacts = {
        "raw_t0p8_samples": {
            "path": str(output_dir / RAW_SAMPLES_FILENAME),
            "records": len(all_parsed),
            "exists": (output_dir / RAW_SAMPLES_FILENAME).exists(),
        },
        "parsed_t0p8_samples": {
            "path": str(output_dir / PARSED_T0P8_FILENAME),
            "records": len(all_parsed),
            "exists": (output_dir / PARSED_T0P8_FILENAME).exists(),
        },
        "relabel_t0_samples": {
            "path": str(output_dir / RELABELED_SAMPLES_FILENAME),
            "records": len(all_parsed),
            "exists": (output_dir / RELABELED_SAMPLES_FILENAME).exists(),
        },
        "parsed_t0_samples": {
            "path": str(output_dir / PARSED_T0_FILENAME),
            "records": len(all_parsed),
            "exists": (output_dir / PARSED_T0_FILENAME).exists(),
        },
        "parsed_judged_samples": {
            "path": str(output_dir / PARSED_JUDGED_FILENAME) if args.intent_filter else None,
            "records": len(all_parsed) if args.intent_filter else 0,
            "exists": (output_dir / PARSED_JUDGED_FILENAME).exists() if args.intent_filter else False,
        },
        "canonical_samples": {
            "path": str(output_dir / CANONICAL_SAMPLES_FILENAME),
            "records": len(all_parsed),
            "exists": (output_dir / CANONICAL_SAMPLES_FILENAME).exists(),
        },
        "dpo_pairs": {
            "path": str(output_dir / "dpo_pairs.jsonl"),
            "records": len(dpo_pairs),
            "exists": (output_dir / "dpo_pairs.jsonl").exists(),
        },
        "pairs_wrong_only": {
            "path": str(output_dir / PAIRS_WRONG_ONLY_FILENAME),
            "records": len(pair_sets["wrong_only"]),
            "exists": (output_dir / PAIRS_WRONG_ONLY_FILENAME).exists(),
        },
        "pairs_judge_bad_only": {
            "path": str(output_dir / PAIRS_JUDGE_BAD_ONLY_FILENAME),
            "records": len(pair_sets["judge_bad_only"]),
            "exists": (output_dir / PAIRS_JUDGE_BAD_ONLY_FILENAME).exists(),
        },
        "pairs_judge_bad_decent_only": {
            "path": str(output_dir / PAIRS_JUDGE_BAD_DECENT_ONLY_FILENAME),
            "records": len(pair_sets["judge_bad_decent_only"]),
            "exists": (output_dir / PAIRS_JUDGE_BAD_DECENT_ONLY_FILENAME).exists(),
        },
        "pairs_wrong_plus_judge_bad": {
            "path": str(output_dir / PAIRS_WRONG_PLUS_JUDGE_BAD_FILENAME),
            "records": len(pair_sets["wrong_plus_judge_bad"]),
            "exists": (output_dir / PAIRS_WRONG_PLUS_JUDGE_BAD_FILENAME).exists(),
        },
        "pairs_wrong_plus_judge_bad_decent": {
            "path": str(output_dir / PAIRS_WRONG_PLUS_JUDGE_BAD_DECENT_FILENAME),
            "records": len(pair_sets["wrong_plus_judge_bad_decent"]),
            "exists": (output_dir / PAIRS_WRONG_PLUS_JUDGE_BAD_DECENT_FILENAME).exists(),
        },
        "dpo_pairs_decent": {
            "path": str(output_dir / "dpo_pairs_decent.jsonl") if args.intent_filter else None,
            "records": len(dpo_pairs_decent) if args.intent_filter else 0,
            "exists": (output_dir / "dpo_pairs_decent.jsonl").exists() if args.intent_filter else False,
        },
        "intent_filter_examples": {
            "path": str(output_dir / "intent_filter_examples.jsonl") if args.intent_filter else None,
            "records": len(filter_examples) if args.intent_filter else 0,
            "exists": (output_dir / "intent_filter_examples.jsonl").exists() if args.intent_filter else False,
        },
    }

    summary = {
        "summary_version": 2,
        "adapter_path":    args.adapter_path,
        "adapter_revision": args.adapter_revision,
        "base_model":      args.base_model,
        "temperature":     args.temperature,
        "num_samples":     args.num_samples,
        "intent_filter":   args.intent_filter,
        "judge_only":      args.judge_only,
        "from_jsonl":      args.from_jsonl,
        "split":           "custom_jsonl" if args.from_jsonl else args.split,
        "n_prompts_total": n_total,
        "n_skipped_no_gold_harm":  n_no_gold_harm,
        "n_skipped_no_gold_intent": n_no_gold_intent,
        "n_skipped_no_valid": n_no_valid,
        "n_skipped_no_neg":   n_no_negative,
        "n_dpo_pairs":        n_dpo,
        "n_dpo_pairs_decent": len(dpo_pairs_decent),
        "pair_set_definitions": {
            "wrong_only": "harm_t0 != gold_harm",
            "judge_bad_only": "harm_t0 == gold_harm and judge_verdict == bad_match",
            "judge_bad_decent_only": "harm_t0 == gold_harm and judge_verdict in {bad_match, decent_match}",
            "wrong_plus_judge_bad": "union of wrong_only and judge_bad_only",
            "wrong_plus_judge_bad_decent": "union of wrong_only and judge_bad_decent_only",
        },
        "pair_set_rows": {
            "wrong_only": n_rows_wrong_only,
            "judge_bad_only": n_rows_judge_bad_only,
            "judge_bad_decent_only": n_rows_judge_bad_decent_only,
            "wrong_plus_judge_bad": n_rows_wrong_plus_judge_bad,
            "wrong_plus_judge_bad_decent": n_rows_wrong_plus_judge_bad_decent,
        },
        "rejecteds_counts": {
            "hard_wrong_label": n_hard_rejecteds,
            "judge_bad": n_intent_rejecteds,
            "judge_bad_decent": n_intent_rejecteds_decent,
        },
        "relabel_stats":      relabel_stats,
        "artifacts":          artifacts,
        "artifact_roles": {
            RAW_SAMPLES_FILENAME: "Raw generation parse at T=temperature with harm_t08.",
            PARSED_T0P8_FILENAME: "Structured snapshot after T=temperature parsing.",
            RELABELED_SAMPLES_FILENAME: "Snapshot after deterministic T=0 harm relabeling.",
            PARSED_T0_FILENAME: "Structured snapshot after deterministic T=0 relabeling.",
            PARSED_JUDGED_FILENAME: "Structured snapshot after judge verdict assignment (intent filter only).",
            CANONICAL_SAMPLES_FILENAME: "Backward-compatible alias to latest structured snapshot.",
            PAIRS_WRONG_ONLY_FILENAME: "Pair set: wrong_only (harm_t0 != gold_harm).",
            PAIRS_JUDGE_BAD_ONLY_FILENAME: "Pair set: judge_bad_only (correct harm, bad_match intent).",
            PAIRS_JUDGE_BAD_DECENT_ONLY_FILENAME: "Pair set: judge_bad_decent_only (correct harm, bad/decent intent).",
            PAIRS_WRONG_PLUS_JUDGE_BAD_FILENAME: "Pair set: wrong_plus_judge_bad (union).",
            PAIRS_WRONG_PLUS_JUDGE_BAD_DECENT_FILENAME: "Pair set: wrong_plus_judge_bad_decent (union).",
            "dpo_pairs.jsonl": "Legacy alias to the active training set for this run.",
            "dpo_pairs_decent.jsonl": "Legacy alias to the active decent variant for this run.",
        },
    }

    summary_path = output_dir / "summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved summary      → {summary_path}")

    print("\n" + "=" * 65)
    print("PAIR GENERATION SUMMARY")
    print("=" * 65)
    split_label = "custom_jsonl" if args.from_jsonl else args.split
    print(f"Split              : {split_label}")
    print(f"Total prompts      : {n_total}")
    print(f"Skipped (no harm)  : {n_no_gold_harm}")
    print(f"Skipped (no intent): {n_no_gold_intent}")
    print(f"Skipped (no valid) : {n_no_valid}")
    print(f"Skipped (no neg)   : {n_no_negative}")
    print(
        "Relabel switches   : "
        f"{relabel_stats['n_harm_label_switched_t08_to_t0']} / "
        f"{relabel_stats['n_samples_comparable_t08_t0']} "
        f"({relabel_stats['pct_harm_label_switched_t08_to_t0']:.1f}%)"
    )
    if args.intent_filter:
        print(f"Intent rejecteds   : {n_intent_rejecteds}  (bad_match only)")
        print(f"Intent rejecteds   : {n_intent_rejecteds_decent}  (bad+decent match)")
    print(f"DPO pairs          : {n_dpo}  ({100*n_dpo/max(n_total,1):.1f}%)")
    if args.intent_filter:
        print(f"DPO pairs (decent) : {len(dpo_pairs_decent)}  ({100*len(dpo_pairs_decent)/max(n_total,1):.1f}%)")
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
        default=DEFAULT_SFT_ADAPTER,
        help="SFT LoRA adapter source (local path or Hugging Face repo ID).",
    )
    p.add_argument(
        "--adapter-revision", type=str, default=None,
        help="Optional HF revision (branch/tag/commit) for --adapter-path.",
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
    p.add_argument("--seed",        type=int,   default=22,
                   help="Random seed for vLLM sampling (passed to SamplingParams). "
                        "Fixes pair generation across runs with the same inputs.")
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument(
        "--max-model-len", type=int, default=4096,
        help="Maximum total prompt length for vLLM generation.",
    )

    # Re-use previous generation
    p.add_argument(
        "--from-samples", type=str, default=None,
        help="Path to parsed_samples.jsonl from a previous run. "
             "Skips the T=temperature generation step.",
    )

    # Dataset split
    p.add_argument(
        "--split", type=str, default="train",
        choices=["train", "validation", "test", "val_and_test"],
        help="Which annotated HF split(s) to use as the prompt source. "
             "'val_and_test' concatenates validation + test (346 examples) — "
             "use this to build the canonical DPO validation set for early stopping.",
    )

    # Custom input dataset (for WildGuardMix augmentation)
    p.add_argument(
        "--from-jsonl", type=str, default=None,
        help="Path to a JSONL file with {id, prompt, gold_harm} records to use as "
             "the prompt source instead of the HF annotated train split. "
             "Intended for WildGuardMix augmentation (see filter_wildguard_easy.py). "
             "Note: gold_intent will be None for these records, so they will not "
             "contribute DPO pairs via the intent-filter path.",
    )

    # Intent filter
    p.add_argument(
        "--intent-filter", action="store_true", default=False,
        help="Enable intent validity filter. Correct-label samples judged as bad_match "
             "by the LLM judge are added as extra rejected pairs.",
    )
    p.add_argument(
        "--judge-only", action="store_true", default=False,
        help="Use only judge-based rejecteds (requires --intent-filter). Hard mislabels "
             "(harm_t0 != gold_harm) are excluded; rejected set contains only samples "
             "that got the harm label right but whose intent was judged as bad_match.",
    )
    p.add_argument(
        "--union-judge-dir", type=str, default=None,
        help="Path to a second judge run's output directory (requires --intent-filter). "
             "Merges bad_match verdicts: a sample is rejected if EITHER judge said bad_match. "
             "Both runs must share the same canonical parsed_samples (same IDs, same order).",
    )
    p.add_argument(
        "--judge-model", type=str, default=None,
        help="HuggingFace model ID for the intent judge. Defaults to --base-model. "
             "Set to a larger model (e.g. meta-llama/Llama-3.1-70B-Instruct or "
             "google/gemma-3-27b-it) to use a separate judge instance. "
             "When different from --base-model, the SFT model is not loaded — "
             "run without --intent-filter first to cache harm_t0 labels, then "
             "re-run with --from-samples --intent-filter --judge-model <model>.",
    )
    p.add_argument(
        "--judge-tensor-parallel", type=int, default=1,
        help="Number of GPUs for tensor-parallel inference of the judge model. "
             "Set to 2 for 27B models on 40 GB A100s (fits in bfloat16 across 2 GPUs).",
    )

    # Output
    p.add_argument("--output-dir", type=str, default="data/dpo_pairs/train_t0.8")

    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
