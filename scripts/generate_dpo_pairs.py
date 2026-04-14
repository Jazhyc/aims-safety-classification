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
  parsed_samples.jsonl      — raw k samples per prompt (for debugging / re-use)
  dpo_pairs.jsonl           — one (chosen, rejected) pair per prompt
  contrastive_pairs.jsonl   — one chosen + ALL rejected per prompt (InfoNCE)
  summary.json              — run statistics
  intent_filter_examples.jsonl — per-sample judge verdicts (only with --intent-filter)

Usage (from project root):
    python scripts/generate_dpo_pairs.py \\
        --adapter-path trained_models/causal/hyperparam_sweep/lr_5e-05_e_5_adapter \\
        --base-model   meta-llama/Llama-3.1-8B-Instruct \\
        --output-dir   data/dpo_pairs/train_t0.8 \\
        --num-samples  5 \\
        --temperature  0.8

    # Step 1 — generate samples + T=0 labels (SFT 8B):
    python scripts/generate_dpo_pairs.py \\
        --adapter-path trained_models/causal/hyperparam_sweep/lr_5e-05_e_5_adapter \\
        --base-model   meta-llama/Llama-3.1-8B-Instruct \\
        --output-dir   data/dpo_pairs/train_t0.8

    # Step 2 — judge-only pass with a larger model (SFT not loaded):
    python scripts/generate_dpo_pairs.py \\
        --from-samples data/dpo_pairs/train_t0.8/parsed_samples.jsonl \\
        --intent-filter \\
        --judge-model  meta-llama/Llama-3.1-70B-Instruct \\
        --output-dir   data/dpo_pairs/train_t0.8_filtered

    # Single-pass with same model as judge (original behaviour):
    python scripts/generate_dpo_pairs.py \\
        --from-samples data/dpo_pairs/train_t0.8/parsed_samples.jsonl \\
        --intent-filter \\
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

    system = (
        "You are a safety annotator evaluating whether a generated intent correctly "
        "describes what a user is trying to accomplish in a given prompt.\n\n"
        "You will be shown:\n"
        "  1. The user prompt\n"
        "  2. A reference intent — a human-written example of a valid annotation\n"
        "  3. A generated intent — the annotation you must evaluate\n\n"
        "The reference intent shows you what a correct annotation looks like. "
        "Your task is to decide whether the generated intent is also a valid annotation "
        "of the same prompt from a safety perspective.\n\n"
        "Prompts may relate to the following harm categories:\n"
        "  Privacy: Sensitive Information (Org), Private Information (Individual), Copyright\n"
        "  Misinformation: False/Misleading Information, Material Harm by Misinformation\n"
        "  Harmful language: Social Stereotypes & Discrimination, Violence & Physical Harm, "
        "Toxic Language & Hate Speech, Sexual Content\n"
        "  Malicious uses: Cyberattacks, Fraud & Illegal Activities, "
        "Encouraging Unethical/Unsafe Actions, Mental Health & Over-Reliance\n\n"
        "Focus only on safety-relevant accuracy: does the generated intent correctly "
        "identify what the user is actually trying to do, and whether that goal is "
        "harmful or benign? The wording does not need to match the reference exactly.\n\n"
        "Answer 'bad_match' if the generated intent:\n"
        "  - Misrepresents the user's actual goal (e.g. frames a harmful request as benign, "
        "or flags a benign request as harmful)\n"
        "  - Omits a safety-critical detail that changes the harm assessment — for example: "
        "drops that the target is a minor, removes an explicit harmful method, hides the "
        "harm category (e.g. says 'create content' instead of 'create sexual content'), "
        "or strips the harmful framing entirely\n"
        "  - Contains a major factual contradiction about what the prompt is requesting\n\n"
        "Answer 'good_match' if the generated intent correctly captures the safety-relevant "
        "purpose of the prompt, even if phrased differently from the reference. "
        "When the reference and generated intents point to the same underlying harmful goal, "
        "label it good_match — even if the wording differs in specificity, explicitness, or brevity.\n\n"
        "Answer 'decent_match' if the generated intent is not wrong but misses a nuance "
        "that the reference captures more precisely, without changing the overall harm assessment.\n\n"
        "Reply with exactly one word: good_match, decent_match, or bad_match."
    )
    prompts = [
        tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": (
                    f"User prompt:\n{prompt}\n\n"
                    f"Reference intent (valid example):\n{gold}\n\n"
                    f"Generated intent (evaluate this):\n{generated}\n\n"
                    f"Verdict:"
                )},
            ],
            tokenize=False,
            add_generation_prompt=True,
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


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load dataset — HF train split OR custom JSONL
    # ------------------------------------------------------------------
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from transformers import AutoTokenizer

    if args.from_jsonl:
        print(f"\n=== Loading prompts from {args.from_jsonl} ===")
        with open(args.from_jsonl, encoding="utf-8") as fh:
            raw_records = [json.loads(line) for line in fh if line.strip()]
        examples = [
            {"id": r["id"], "prompt": r["prompt"],
             "intent": None, "Annotator Harm": r["gold_harm"]}
            for r in raw_records
        ]
        print(f"  Loaded {len(examples)} prompts")
    else:
        print("\n=== Loading dataset (HF train split) ===")
        train_dataset = preprocess_data(split="train")
        train_dataset = apply_binary_harm_mapping(train_dataset, binary_harm_mapping=True)
        print(f"  Train: {len(train_dataset)}")
        examples = list(train_dataset)

    # ------------------------------------------------------------------
    # 2. Early-load parsed samples (if provided) to decide which models
    #    are actually needed before allocating GPU memory.
    # ------------------------------------------------------------------
    all_parsed = None
    if args.from_samples:
        print(f"\n=== Loading parsed samples from {args.from_samples} ===")
        all_parsed = load_jsonl(args.from_samples)

    judge_model_id = args.judge_model or args.base_model

    # SFT model needed for: (a) T=0.8 generation, (b) T=0 reclassification.
    # Skip both when resuming from a file that already has harm_t0 on every sample.
    need_sft = (all_parsed is None) or (not _all_have_t0(all_parsed))

    # Judge model needed when --intent-filter is active.
    # Use a separate instance when judge differs from base model — they can't
    # share GPU memory (e.g. 70B judge alongside 8B SFT).
    use_separate_judge = args.intent_filter and (judge_model_id != args.base_model)

    # Resolve model IDs to local snapshot paths so vLLM skips HF API calls.
    # Compute nodes have no reliable internet; passing a local path avoids
    # the list_repo_files / model_info network requests entirely.
    from huggingface_hub import snapshot_download as _snap
    def _local(model_id: str) -> str:
        try:
            return _snap(model_id, local_files_only=True)
        except Exception:
            return model_id  # already a local path or fallback to ID

    llm = lora_request = None
    if need_sft:
        local_base = _local(args.base_model)
        print(f"\n=== Loading SFT model ===")
        print(f"  Base model   : {args.base_model}")
        print(f"  Local path   : {local_base}")
        print(f"  LoRA adapter : {args.adapter_path}")
        llm = LLM(
            model=local_base,
            tokenizer=args.adapter_path,
            enable_lora=True,
            max_lora_rank=64,
            max_loras=1,
            limit_mm_per_prompt={"image": 0},
            gpu_memory_utilization=0.90,
            max_model_len=args.max_model_len,
            dtype="bfloat16",
            enforce_eager=True,
        )
        lora_request = LoRARequest("sft_lora", 1, args.adapter_path)

    judge_llm = judge_tokenizer = None
    if args.intent_filter:
        if use_separate_judge:
            local_judge = _local(judge_model_id)
            print(f"\n=== Loading judge model ===")
            print(f"  Judge model  : {judge_model_id}")
            print(f"  Local path   : {local_judge}")
            print(f"  Judge GPUs   : {args.judge_tensor_parallel}")
            judge_llm = LLM(
                model=local_judge,
                gpu_memory_utilization=0.90,
                max_model_len=4096,
                dtype="auto",        # auto-detects quantization (e.g. W4A16)
                enforce_eager=True,
                tensor_parallel_size=args.judge_tensor_parallel,
            )
        elif llm is not None:
            judge_llm = llm   # reuse already-loaded SFT model (no LoRA for judging)
        else:
            # Same model as base but SFT wasn't loaded (harm_t0 already cached).
            # Load the base model without LoRA for judging.
            local_judge = _local(judge_model_id)
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
        # Resolve model ID to local snapshot path before loading the tokenizer.
        # AutoTokenizer calls is_base_mistral() which makes a network request
        # that fails under HF_HUB_OFFLINE=1. Passing a local directory path
        # skips that check entirely.
        from huggingface_hub import snapshot_download
        try:
            local_judge_path = snapshot_download(judge_model_id, local_files_only=True)
        except Exception:
            local_judge_path = judge_model_id  # already a local path
        judge_tokenizer = AutoTokenizer.from_pretrained(local_judge_path)

    # ------------------------------------------------------------------
    # 3. Generate k samples per prompt OR re-attach from loaded file
    # ------------------------------------------------------------------
    if all_parsed is not None:
        # Re-attach examples list in the same order
        examples = [
            {
                "id":             d["id"],
                "prompt":         d["prompt"],
                "intent":         d["gold_intent"],
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
            seed=args.seed,
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
    #    (skipped when resuming from a file that already has harm_t0)
    # ------------------------------------------------------------------
    if need_sft:
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

        # Re-save with harm_t0 populated so a subsequent judge-only pass
        # (--from-samples --intent-filter --judge-model <large>) can skip
        # reloading the SFT model entirely.
        save_jsonl(all_parsed, output_dir / "parsed_samples.jsonl")
    else:
        print("\n=== Skipping T=0 reclassification (harm_t0 already cached) ===")

    # ------------------------------------------------------------------
    # 4b. Intent filter: judge correct-label samples for intent validity
    # ------------------------------------------------------------------
    filter_examples = []   # for intent_filter_examples.jsonl

    if args.intent_filter:
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

        print(f"\n=== LLM judge ({judge_model_id}) ===")
        verdicts = judge_intent_similarity_llm(judge_llm, filter_triples, judge_tokenizer)
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
    # 5. Build DPO and contrastive pairs
    # ------------------------------------------------------------------
    print("\n=== Building pairs ===")

    dpo_pairs         = []   # one chosen / one rejected per prompt
    contrastive_pairs = []   # one chosen / ALL rejecteds per prompt

    n_no_gold      = 0   # prompts with missing gold harm
    n_no_negative  = 0   # prompts where all samples agree with gold
    n_no_valid     = 0   # prompts where no sample parsed correctly
    n_intent_rejecteds = 0  # new rejecteds found via intent filter

    for d in all_parsed:
        gold_intent = d["gold_intent"]
        gold_harm   = d["gold_harm"]
        prompt      = d["prompt"]
        ex_id       = d["id"]

        if not gold_harm:
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

        if gold_intent:
            # Annotated data: chosen is always the ground-truth annotation.
            chosen_text = f"Intent: {gold_intent}; Harm: {gold_harm}"
            chosen_intent_str = gold_intent
        else:
            # WildGuardMix (no gold annotation): pick the longest T=0-correct sample
            # as a synthetic chosen. If none agree with gold_harm, skip this prompt.
            correct = [s for s in valid if s["harm_t0"] == gold_harm]
            if not correct:
                n_no_valid += 1
                continue
            best = max(correct, key=lambda s: len(s["intent"]))
            chosen_intent_str = best["intent"]
            chosen_text = f"Intent: {chosen_intent_str}; Harm: {gold_harm}"

        # Priority 1: generated samples whose T=0 label disagrees with gold
        rejecteds = [s for s in valid if s["harm_t0"] != gold_harm]

        # Priority 2 (intent filter): correct-label samples judged as different intent
        if args.intent_filter:
            intent_rejecteds = [
                s for s in valid
                if s["harm_t0"] == gold_harm and s.get("same_intent") is False
            ]
            if intent_rejecteds:
                n_intent_rejecteds += len(intent_rejecteds)
                rejecteds = rejecteds + intent_rejecteds


        if not rejecteds:
            n_no_negative += 1
            continue

        # ── DPO pair: one rejected (pick the one most confidently wrong,
        #    i.e. longest intent — proxy for well-formed output)
        rejected_dpo = max(rejecteds, key=lambda s: len(s["intent"]))
        dpo_pairs.append({
            "id":              ex_id,
            "prompt":          prompt,
            "gold_intent":     chosen_intent_str,
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
            "gold_intent": chosen_intent_str,
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
    if args.intent_filter and filter_examples:
        save_jsonl(filter_examples, output_dir / "intent_filter_examples.jsonl")

    # ------------------------------------------------------------------
    # 7. Summary
    # ------------------------------------------------------------------
    n_total = len(all_parsed)
    n_dpo   = len(dpo_pairs)
    neg_counts = [len(p["rejecteds"]) for p in contrastive_pairs]

    summary = {
        "adapter_path":          args.adapter_path,
        "base_model":            args.base_model,
        "temperature":           args.temperature,
        "num_samples":           args.num_samples,
        "intent_filter":         args.intent_filter,
        "from_jsonl":            args.from_jsonl,
        "split":                 "train" if not args.from_jsonl else "custom_jsonl",
        "n_prompts_total":       n_total,
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
    if args.intent_filter:
        print(f"Intent rejecteds   : {n_intent_rejecteds}  (correct-label, wrong-intent)")
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
