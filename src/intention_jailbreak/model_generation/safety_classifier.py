"""
Condition runners and parsing utilities for the safety classifier evaluation.

Each run_* function corresponds to one evaluation condition defined in VALID_CONDITIONS.
All functions accept a vLLM LLM instance and return a list of result dicts with at least:
  id, prompt, true_harm, true_harm_binary, predicted_harm, raw_generation, condition
"""

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Tuple

from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from vllm.sampling_params import StructuredOutputsParams as GuidedDecodingParams
from transformers import AutoTokenizer

from intention_jailbreak.model_generation.prompt_templates import build_student_prompt, build_student_messages
from .safety_prompts import (
    BINARY_LABELS,
    CLASSIFICATION_SYSTEM_PROMPT,
    CLASSIFICATION_WITH_INTENT_SYSTEM_PROMPT,
    GENERATION_JSON_SYSTEM_PROMPT,
    COT_CLASSIFICATION_SYSTEM_PROMPT,
    COT_GENERATION_SYSTEM_PROMPT,
    COT_CLASSIFICATION_WITH_INTENT_SYSTEM_PROMPT,
    COT_CLASSIFICATION_SCHEMA,
    COT_GENERATION_SCHEMA,
    _WILDGUARD_TEMPLATE,
    _SAFEGUARD_SYSTEM_PROMPT,
    _GUARDREASONER_INSTRUCT,
    _SHIELDGEMMA_PROMPT,
    _NEMOTRON_SAFETY_PROMPT,
)

# ---------------------------------------------------------------------------
# Parsing / extraction helpers
# ---------------------------------------------------------------------------

def extract_intent_and_harm(generated_text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract intent and harm label from generated text.

    Expected format: "Intent: <text>; Harm: <label>" or "Intent: <text> Harm: <label>"
    """
    generated_text = generated_text.strip()

    if "<think>" in generated_text and "</think>" in generated_text:
        generated_text = generated_text.split("</think>")[-1].strip()

    def normalize_harm(harm_str: str) -> Optional[str]:
        harm = harm_str.strip().lower().rstrip(".")
        if harm in ["safe", "harmless", "benign", "s", "b"]:
            return "safe"
        elif harm in ["harmful", "unsafe", "dangerous", "h", "u"]:
            return "harmful"
        return None

    # "Intent: <text>; Harm: <label>"
    m = re.search(r"Intent:\s*(.+?);\s*Harm:\s*(\S+)", generated_text, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip(), normalize_harm(m.group(2))

    # "Intent: <text>\nHarm: <label>"
    m = re.search(r"Intent:\s*(.+?)\n\s*Harm:\s*(\S+)", generated_text, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip().rstrip("."), normalize_harm(m.group(2))

    # "Intent: <text> Harm: <label>"
    m = re.search(r"Intent:\s*(.+?)\s+Harm:\s*(\S+)", generated_text, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip().rstrip("."), normalize_harm(m.group(2))

    # Intent only
    m = re.search(r"Intent:\s*(.+?)(?:\n|$)", generated_text, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip(), None

    return generated_text, None


def extract_harm_label(generated_text: str, valid_labels: List[str] = BINARY_LABELS) -> Optional[str]:
    """Extract harm label from generated text using structured matching."""
    text = generated_text.strip()

    if "<think>" in text and "</think>" in text:
        text = text.split("</think>")[-1].strip()

    text_lower = text.lower()
    first_word = text_lower.split()[0] if text_lower.split() else ""
    first_line = text_lower.split("\n")[0].strip()

    for label in valid_labels:
        if first_word == label.lower() or first_line == label.lower():
            return label.lower()

    for label in valid_labels:
        if label.lower() in text_lower:
            return label.lower()

    if any(w in text_lower for w in ["unsafe", "dangerous", "malicious", "harmful"]):
        return "harmful"
    if any(w in text_lower for w in ["safe", "benign", "harmless", "legitimate"]):
        return "safe"

    return None


def map_harm_to_binary(harm_value: Optional[str]) -> Optional[str]:
    """Map various dataset harm labels to binary (safe/harmful)."""
    if harm_value is None:
        return None

    harm_lower = str(harm_value).lower().strip()

    # XSTest labels
    if harm_lower == "unsafe":
        return "harmful"
    if harm_lower == "safe":
        return "safe"

    # WildGuardMix labels
    if harm_lower == "unharmful":
        return "safe"
    if harm_lower == "harmful":
        return "harmful"

    # 4-category labels from annotated-intents
    if "harmful" in harm_lower:
        return "harmful"
    if "safe" in harm_lower:
        return "safe"

    return None


def _apply_chat_template(tokenizer: AutoTokenizer, messages: list) -> str:
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


def _parse_cot_json(
    generated_text: str, has_intent: bool = False
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Parse JSON output from CoT conditions.

    Returns:
        (reasoning, intent, harm) — intent is None for classification-only schemas.
    """
    reasoning = intent = harm = None
    try:
        parsed = json.loads(generated_text)
        reasoning = parsed.get("reasoning")
        harm_raw = parsed.get("harm", "").lower()
        harm = harm_raw if harm_raw in BINARY_LABELS else None
        if has_intent:
            intent = parsed.get("intent")
    except json.JSONDecodeError:
        m = re.search(r'"reasoning"\s*:\s*"((?:[^"\\]|\\.)*)', generated_text)
        if m:
            reasoning = m.group(1)
        m = re.search(r'"harm"\s*:\s*"(harmful|safe)"', generated_text, re.IGNORECASE)
        if m:
            harm = m.group(1).lower()
        if has_intent:
            m = re.search(r'"intent"\s*:\s*"((?:[^"\\]|\\.)*)', generated_text)
            if m:
                intent = m.group(1)
    return reasoning, intent, harm


def _parse_harmony_final(text: str) -> str:
    """Extract the content of the final channel from a harmony-format response."""
    m = re.search(
        r"<\|channel\|>final<\|message\|>(.*?)(?:<\|return\|>|<\|end\|>|$)", text, re.DOTALL
    )
    return m.group(1).strip() if m else text.strip()


def _parse_reasoning_output(
    raw_text: str, with_intent: bool = False
) -> Tuple[str, Optional[str], Optional[str]]:
    """Parse Reasoning: / [Intent:] / Prompt harm: format produced by distilled models."""
    text = raw_text.strip()

    reasoning = ""
    m = re.search(r"Reasoning:\s*(.+?)(?=Intent:|Prompt harm:|$)", text, re.IGNORECASE | re.DOTALL)
    if m:
        reasoning = m.group(1).strip()

    intent = None
    if with_intent:
        m = re.search(r"Intent:\s*(.+?)(?=Prompt harm:|$)", text, re.IGNORECASE | re.DOTALL)
        if m:
            intent = m.group(1).strip()

    predicted_harm = None
    m = re.search(r"Prompt harm:\s*([^\n]+)", text, re.IGNORECASE)
    if m:
        label = m.group(1).strip().lower()
        if "unharmful" in label or "safe" in label:
            predicted_harm = "safe"
        elif "harmful" in label:
            predicted_harm = "harmful"

    return reasoning, intent, predicted_harm


# ---------------------------------------------------------------------------
# Condition runners — main model
# ---------------------------------------------------------------------------

def run_finetuned_generation(
    llm: LLM,
    lora_request: Optional[LoRARequest],
    tokenizer: AutoTokenizer,
    test_dataset,
    sampling_params: SamplingParams,
    harm_column: str,
) -> List[dict]:
    """Fine-tuned model generates both intent and harm label (no system prompt)."""
    print("\n=== Running: Fine-tuned Generation (intent + harm) ===")
    examples = list(test_dataset)
    prompts = [f"{ex['prompt']}\n" for ex in examples]
    outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)

    results = []
    for ex, output in zip(examples, outputs):
        generated_text = output.outputs[0].text.strip()
        predicted_intent, predicted_harm = extract_intent_and_harm(generated_text)
        true_harm = ex.get(harm_column)
        results.append({
            "id": ex.get("id", ""),
            "prompt": ex["prompt"],
            "true_intent": ex.get("intent"),
            "generated_intent": predicted_intent,
            "true_harm": true_harm,
            "true_harm_binary": map_harm_to_binary(true_harm),
            "predicted_harm": predicted_harm,
            "raw_generation": generated_text,
            "num_tokens": len(output.outputs[0].token_ids),
            "condition": "finetuned_generation",
        })
    return results


def run_finetuned_classification(
    llm: LLM,
    lora_request: Optional[LoRARequest],
    tokenizer: AutoTokenizer,
    test_dataset,
    sampling_params: SamplingParams,
    harm_column: str,
) -> List[dict]:
    """Fine-tuned model directly predicts harm label only (no system prompt)."""
    print("\n=== Running: Fine-tuned Classification (harm only) ===")
    examples = list(test_dataset)
    prompts = [f"{ex['prompt']}\n" for ex in examples]
    outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)

    results = []
    for ex, output in zip(examples, outputs):
        generated_text = output.outputs[0].text.strip()
        true_harm = ex.get(harm_column)
        results.append({
            "id": ex.get("id", ""),
            "prompt": ex["prompt"],
            "true_harm": true_harm,
            "true_harm_binary": map_harm_to_binary(true_harm),
            "predicted_harm": extract_harm_label(generated_text),
            "raw_generation": generated_text,
            "num_tokens": len(output.outputs[0].token_ids),
            "condition": "finetuned_classification",
        })
    return results


def run_vanilla_classification(
    llm: LLM,
    tokenizer: AutoTokenizer,
    test_dataset,
    sampling_params: SamplingParams,
    harm_column: str,
) -> List[dict]:
    """Vanilla model with system prompt classifies harm via guided decoding."""
    print("\n=== Running: Vanilla Classification ===")
    examples = list(test_dataset)

    guided_params = SamplingParams(
        max_tokens=sampling_params.max_tokens,
        temperature=sampling_params.temperature,
        top_p=sampling_params.top_p,
        top_k=sampling_params.top_k,
        skip_special_tokens=True,
        structured_outputs=GuidedDecodingParams(choice=BINARY_LABELS),
    )
    formatted_prompts = [
        _apply_chat_template(tokenizer, [
            {"role": "system", "content": CLASSIFICATION_SYSTEM_PROMPT},
            {"role": "user", "content": ex["prompt"]},
        ])
        for ex in examples
    ]
    outputs = llm.generate(formatted_prompts, guided_params)

    results = []
    for ex, output in zip(examples, outputs):
        generated_text = output.outputs[0].text.strip()
        predicted_harm = (
            generated_text.lower()
            if generated_text.lower() in BINARY_LABELS
            else extract_harm_label(generated_text)
        )
        true_harm = ex.get(harm_column)
        results.append({
            "id": ex.get("id", ""),
            "prompt": ex["prompt"],
            "true_harm": true_harm,
            "true_harm_binary": map_harm_to_binary(true_harm),
            "predicted_harm": predicted_harm,
            "raw_generation": generated_text,
            "num_tokens": len(output.outputs[0].token_ids),
            "condition": "vanilla_classification",
        })
    return results


def run_vanilla_generation(
    llm: LLM,
    tokenizer: AutoTokenizer,
    test_dataset,
    sampling_params: SamplingParams,
    harm_column: str,
) -> List[dict]:
    """Vanilla model generates both intent and harm label via JSON structured output."""
    print("\n=== Running: Vanilla Generation (intent + harm) ===")
    examples = list(test_dataset)

    json_schema = {
        "type": "object",
        "properties": {
            "intent": {"type": "string", "description": "The inferred intent behind the user's prompt"},
            "harm": {"type": "string", "enum": BINARY_LABELS, "description": "Safety classification: harmful or safe"},
        },
        "required": ["intent", "harm"],
        "additionalProperties": False,
    }
    guided_params = SamplingParams(
        max_tokens=max(sampling_params.max_tokens, 512),
        temperature=sampling_params.temperature,
        top_p=sampling_params.top_p,
        top_k=sampling_params.top_k,
        skip_special_tokens=True,
        structured_outputs=GuidedDecodingParams(json=json_schema),
    )
    formatted_prompts = [
        _apply_chat_template(tokenizer, [
            {"role": "system", "content": GENERATION_JSON_SYSTEM_PROMPT},
            {"role": "user", "content": ex["prompt"]},
        ])
        for ex in examples
    ]
    outputs = llm.generate(formatted_prompts, guided_params)

    results = []
    for ex, output in zip(examples, outputs):
        generated_text = output.outputs[0].text.strip()
        predicted_intent = predicted_harm = None
        try:
            parsed = json.loads(generated_text)
            predicted_intent = parsed.get("intent")
            harm_value = parsed.get("harm", "").lower()
            predicted_harm = harm_value if harm_value in BINARY_LABELS else None
        except json.JSONDecodeError:
            m = re.search(r'"intent"\s*:\s*"([^"]*)', generated_text)
            if m:
                predicted_intent = m.group(1)
            m = re.search(r'"harm"\s*:\s*"(harmful|safe)"', generated_text, re.IGNORECASE)
            if m:
                predicted_harm = m.group(1).lower()
            else:
                _, predicted_harm = extract_intent_and_harm(generated_text)
                if predicted_harm is None:
                    predicted_harm = extract_harm_label(generated_text)

        true_harm = ex.get(harm_column)
        results.append({
            "id": ex.get("id", ""),
            "prompt": ex["prompt"],
            "true_intent": ex.get("intent"),
            "generated_intent": predicted_intent,
            "true_harm": true_harm,
            "true_harm_binary": map_harm_to_binary(true_harm),
            "predicted_harm": predicted_harm,
            "raw_generation": generated_text,
            "num_tokens": len(output.outputs[0].token_ids),
            "condition": "vanilla_generation",
        })
    return results


def run_vanilla_generation_openrouter(
    model_name: str,
    test_dataset,
    harm_column: str,
    max_tokens: int = 2048,
    temperature: float = 0.0,
    max_workers: int = 8,
    max_retries: int = 3,
    site_url: Optional[str] = None,
    app_name: Optional[str] = None,
) -> List[dict]:
    """
    Vanilla generation via OpenRouter: prompt -> intent + harm (JSON).

    Mirrors `run_vanilla_generation` but uses OpenRouter's OpenAI-compatible API
    instead of vLLM. JSON output is requested via `response_format={"type":"json_object"}`;
    parsing falls back to regex on malformed output (same behavior as the vLLM path).

    Requires the `OPENROUTER_API_KEY` environment variable.
    """
    from openai import OpenAI

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Export it in your shell or add it to your .env file."
        )

    extra_headers: dict = {}
    if site_url:
        extra_headers["HTTP-Referer"] = site_url
    if app_name:
        extra_headers["X-Title"] = app_name

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        **({"default_headers": extra_headers} if extra_headers else {}),
    )

    examples = list(test_dataset)
    print(f"\n=== Running: Vanilla Generation via OpenRouter ({model_name}, {len(examples)} prompts) ===")
    print(f"  max_workers={max_workers}  max_retries={max_retries}  max_tokens={max_tokens}")

    def call_api(idx: int, ex: dict):
        messages = [
            {"role": "system", "content": GENERATION_JSON_SYSTEM_PROMPT},
            {"role": "user", "content": ex["prompt"]},
        ]
        last_exc: Optional[Exception] = None
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    response_format={"type": "json_object"},
                )
                content = response.choices[0].message.content or ""
                completion_tokens = 0
                if response.usage and getattr(response.usage, "completion_tokens", None):
                    completion_tokens = response.usage.completion_tokens
                return idx, content, completion_tokens
            except Exception as e:
                last_exc = e
                if attempt < max_retries - 1:
                    wait = 2 ** attempt
                    print(f"  [retry {attempt + 1}/{max_retries}] idx={idx}: {e} — {wait}s")
                    time.sleep(wait)
        raise RuntimeError(f"All {max_retries} attempts failed for idx={idx}") from last_exc

    results: List[Optional[dict]] = [None] * len(examples)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(call_api, i, ex): i for i, ex in enumerate(examples)}
        with tqdm(total=len(examples), desc=f"OpenRouter [{model_name}]", unit="req") as pbar:
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    idx, content, num_tokens = future.result()
                except Exception as e:
                    print(f"  FAILED idx={idx}: {e}")
                    content, num_tokens = "", 0

                ex = examples[idx]
                predicted_intent = predicted_harm = None
                try:
                    parsed = json.loads(content)
                    predicted_intent = parsed.get("intent")
                    harm_value = (parsed.get("harm") or "").lower()
                    predicted_harm = harm_value if harm_value in BINARY_LABELS else None
                except (json.JSONDecodeError, AttributeError):
                    m = re.search(r'"intent"\s*:\s*"([^"]*)', content)
                    if m:
                        predicted_intent = m.group(1)
                    m = re.search(r'"harm"\s*:\s*"(harmful|safe)"', content, re.IGNORECASE)
                    if m:
                        predicted_harm = m.group(1).lower()
                    else:
                        _, predicted_harm = extract_intent_and_harm(content)
                        if predicted_harm is None:
                            predicted_harm = extract_harm_label(content)

                true_harm = ex.get(harm_column)
                results[idx] = {
                    "id": ex.get("id", ""),
                    "prompt": ex["prompt"],
                    "true_intent": ex.get("intent"),
                    "generated_intent": predicted_intent,
                    "true_harm": true_harm,
                    "true_harm_binary": map_harm_to_binary(true_harm),
                    "predicted_harm": predicted_harm,
                    "raw_generation": content,
                    "num_tokens": num_tokens,
                    "condition": "vanilla_generation",
                    "openrouter_model": model_name,
                }
                pbar.update(1)

    return [r for r in results if r is not None]


def run_vanilla_classification_with_intent(
    llm: LLM,
    tokenizer: AutoTokenizer,
    test_dataset,
    sampling_params: SamplingParams,
    harm_column: str,
    intent_source: str = "human",
    model_intents: Optional[List[str]] = None,
) -> List[dict]:
    """Vanilla model classifies harm using prompt + intent (human or model-generated)."""
    condition_name = f"vanilla_with_{intent_source}_intent"
    print(f"\n=== Running: Vanilla Classification with {intent_source.title()} Intent ===")
    examples = list(test_dataset)

    guided_params = SamplingParams(
        max_tokens=sampling_params.max_tokens,
        temperature=sampling_params.temperature,
        top_p=sampling_params.top_p,
        top_k=sampling_params.top_k,
        skip_special_tokens=True,
        structured_outputs=GuidedDecodingParams(choice=BINARY_LABELS),
    )
    formatted_prompts = []
    for i, ex in enumerate(examples):
        intent = (
            model_intents[i] if intent_source == "model" and model_intents and i < len(model_intents)
            else ex.get("intent", "Unknown")
        )
        formatted_prompts.append(_apply_chat_template(tokenizer, [
            {"role": "system", "content": CLASSIFICATION_WITH_INTENT_SYSTEM_PROMPT},
            {"role": "user", "content": f"Prompt: {ex['prompt']}\n\nInferred Intent: {intent}"},
        ]))

    outputs = llm.generate(formatted_prompts, guided_params)

    results = []
    for i, (ex, output) in enumerate(zip(examples, outputs)):
        generated_text = output.outputs[0].text.strip()
        predicted_harm = (
            generated_text.lower()
            if generated_text.lower() in BINARY_LABELS
            else extract_harm_label(generated_text)
        )
        true_harm = ex.get(harm_column)
        result = {
            "id": ex.get("id", ""),
            "prompt": ex["prompt"],
            "true_harm": true_harm,
            "true_harm_binary": map_harm_to_binary(true_harm),
            "predicted_harm": predicted_harm,
            "raw_generation": generated_text,
            "num_tokens": len(output.outputs[0].token_ids),
            "condition": condition_name,
        }
        if intent_source == "model" and model_intents:
            result["model_intent"] = model_intents[i] if i < len(model_intents) else None
        if "intent" in ex:
            result["human_intent"] = ex.get("intent")
        results.append(result)
    return results


def run_zeroshot_cot_classification(
    llm: LLM,
    tokenizer: AutoTokenizer,
    test_dataset,
    sampling_params: SamplingParams,
    harm_column: str,
) -> List[dict]:
    """CoT Classification: prompt -> reasoning + harm (JSON)."""
    print("\n=== Running: Zero-shot CoT Classification (reasoning + harm) ===")
    examples = list(test_dataset)

    cot_params = SamplingParams(
        max_tokens=max(sampling_params.max_tokens, 1024),
        temperature=sampling_params.temperature,
        top_p=sampling_params.top_p,
        top_k=sampling_params.top_k,
        skip_special_tokens=True,
        structured_outputs=GuidedDecodingParams(json=COT_CLASSIFICATION_SCHEMA),
    )
    formatted_prompts = [
        _apply_chat_template(tokenizer, [
            {"role": "system", "content": COT_CLASSIFICATION_SYSTEM_PROMPT},
            {"role": "user", "content": ex["prompt"]},
        ])
        for ex in examples
    ]
    outputs = llm.generate(formatted_prompts, cot_params)

    results = []
    for ex, output in zip(examples, outputs):
        generated_text = output.outputs[0].text.strip()
        reasoning, _, predicted_harm = _parse_cot_json(generated_text, has_intent=False)
        true_harm = ex.get(harm_column)
        results.append({
            "id": ex.get("id", ""),
            "prompt": ex["prompt"],
            "true_harm": true_harm,
            "true_harm_binary": map_harm_to_binary(true_harm),
            "predicted_harm": predicted_harm,
            "reasoning": reasoning,
            "raw_generation": generated_text,
            "num_tokens": len(output.outputs[0].token_ids),
            "condition": "zeroshot_cot_classification",
        })
    return results


def run_zeroshot_cot_generation(
    llm: LLM,
    tokenizer: AutoTokenizer,
    test_dataset,
    sampling_params: SamplingParams,
    harm_column: str,
) -> List[dict]:
    """CoT Generation: prompt -> reasoning + intent + harm (JSON)."""
    print("\n=== Running: Zero-shot CoT Generation (reasoning + intent + harm) ===")
    examples = list(test_dataset)

    cot_params = SamplingParams(
        max_tokens=max(sampling_params.max_tokens, 1024),
        temperature=sampling_params.temperature,
        top_p=sampling_params.top_p,
        top_k=sampling_params.top_k,
        skip_special_tokens=True,
        structured_outputs=GuidedDecodingParams(json=COT_GENERATION_SCHEMA),
    )
    formatted_prompts = [
        _apply_chat_template(tokenizer, [
            {"role": "system", "content": COT_GENERATION_SYSTEM_PROMPT},
            {"role": "user", "content": ex["prompt"]},
        ])
        for ex in examples
    ]
    outputs = llm.generate(formatted_prompts, cot_params)

    results = []
    for ex, output in zip(examples, outputs):
        generated_text = output.outputs[0].text.strip()
        reasoning, predicted_intent, predicted_harm = _parse_cot_json(generated_text, has_intent=True)
        true_harm = ex.get(harm_column)
        results.append({
            "id": ex.get("id", ""),
            "prompt": ex["prompt"],
            "true_intent": ex.get("intent"),
            "generated_intent": predicted_intent,
            "true_harm": true_harm,
            "true_harm_binary": map_harm_to_binary(true_harm),
            "predicted_harm": predicted_harm,
            "reasoning": reasoning,
            "raw_generation": generated_text,
            "num_tokens": len(output.outputs[0].token_ids),
            "condition": "zeroshot_cot_generation",
        })
    return results


def run_zeroshot_cot_classification_with_intent(
    llm: LLM,
    tokenizer: AutoTokenizer,
    test_dataset,
    sampling_params: SamplingParams,
    harm_column: str,
) -> List[dict]:
    """CoT Classification with Intent: prompt + human intent -> reasoning + harm (JSON)."""
    print("\n=== Running: Zero-shot CoT Classification with Intent (reasoning + harm) ===")
    examples = list(test_dataset)

    cot_params = SamplingParams(
        max_tokens=max(sampling_params.max_tokens, 1024),
        temperature=sampling_params.temperature,
        top_p=sampling_params.top_p,
        top_k=sampling_params.top_k,
        skip_special_tokens=True,
        structured_outputs=GuidedDecodingParams(json=COT_CLASSIFICATION_SCHEMA),
    )
    formatted_prompts = [
        _apply_chat_template(tokenizer, [
            {"role": "system", "content": COT_CLASSIFICATION_WITH_INTENT_SYSTEM_PROMPT},
            {"role": "user", "content": f"Prompt: {ex['prompt']}\n\nInferred Intent: {ex.get('intent', 'Unknown')}"},
        ])
        for ex in examples
    ]
    outputs = llm.generate(formatted_prompts, cot_params)

    results = []
    for ex, output in zip(examples, outputs):
        generated_text = output.outputs[0].text.strip()
        reasoning, _, predicted_harm = _parse_cot_json(generated_text, has_intent=False)
        true_harm = ex.get(harm_column)
        results.append({
            "id": ex.get("id", ""),
            "prompt": ex["prompt"],
            "human_intent": ex.get("intent"),
            "true_harm": true_harm,
            "true_harm_binary": map_harm_to_binary(true_harm),
            "predicted_harm": predicted_harm,
            "reasoning": reasoning,
            "raw_generation": generated_text,
            "num_tokens": len(output.outputs[0].token_ids),
            "condition": "zeroshot_cot_classification_with_intent",
        })
    return results


def run_finetuned_reasoning_classification(
    llm: LLM,
    lora_request: Optional[LoRARequest],
    tokenizer: AutoTokenizer,
    test_dataset,
    sampling_params: SamplingParams,
    harm_column: str,
) -> List[dict]:
    """Distilled model (MODE D): prompt -> Reasoning: ... / Prompt harm: harmful|safe."""
    print("\n=== Running: Fine-tuned Reasoning Classification (reasoning + harm) ===")
    examples = list(test_dataset)
    prompts = [
        _apply_chat_template(tokenizer, build_student_messages(ex["prompt"], condition="no_intent"))
        for ex in examples
    ]
    outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)

    results = []
    for ex, output in zip(examples, outputs):
        generated_text = output.outputs[0].text.strip()
        reasoning, _, predicted_harm = _parse_reasoning_output(generated_text, with_intent=False)
        true_harm = ex.get(harm_column)
        results.append({
            "id": ex.get("id", ""),
            "prompt": ex["prompt"],
            "true_harm": true_harm,
            "true_harm_binary": map_harm_to_binary(true_harm),
            "predicted_harm": predicted_harm,
            "reasoning": reasoning,
            "raw_generation": generated_text,
            "num_tokens": len(output.outputs[0].token_ids),
            "condition": "finetuned_reasoning_classification",
        })
    return results


def run_finetuned_reasoning_human_intent(
    llm: LLM,
    lora_request: Optional[LoRARequest],
    tokenizer: AutoTokenizer,
    test_dataset,
    sampling_params: SamplingParams,
    harm_column: str,
) -> List[dict]:
    """Distilled model (MODE E): prompt -> Reasoning: ... / Intent: ... / Prompt harm: harmful|safe.

    Intent is from human-annotated ground truth."""
    print("\n=== Running: Fine-tuned Reasoning Human Intent (reasoning + human intent + harm) ===")
    examples = list(test_dataset)
    prompts = [
        _apply_chat_template(tokenizer, build_student_messages(ex["prompt"], condition="human_intent"))
        for ex in examples
    ]
    outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)

    results = []
    for ex, output in zip(examples, outputs):
        generated_text = output.outputs[0].text.strip()
        reasoning, generated_intent, predicted_harm = _parse_reasoning_output(generated_text, with_intent=True)
        true_harm = ex.get(harm_column)
        results.append({
            "id": ex.get("id", ""),
            "prompt": ex["prompt"],
            "true_intent": ex.get("intent"),
            "generated_intent": generated_intent,
            "true_harm": true_harm,
            "true_harm_binary": map_harm_to_binary(true_harm),
            "predicted_harm": predicted_harm,
            "reasoning": reasoning,
            "raw_generation": generated_text,
            "num_tokens": len(output.outputs[0].token_ids),
            "condition": "finetuned_reasoning_human_intent",
        })
    return results


def run_finetuned_reasoning_synthetic_intent(
    llm: LLM,
    lora_request: Optional[LoRARequest],
    tokenizer: AutoTokenizer,
    test_dataset,
    sampling_params: SamplingParams,
    harm_column: str,
) -> List[dict]:
    """Distilled model (MODE F): prompt -> Reasoning: ... / Intent: ... / Prompt harm: harmful|safe

    Same as finetuned_reasoning_generation but trained on model-generated (synthetic) intents
    rather than ground-truth intents.
    """
    print("\n=== Running: Fine-tuned Reasoning Synthetic Intent (reasoning + synthetic intent + harm) ===")
    examples = list(test_dataset)
    prompts = [
        _apply_chat_template(tokenizer, build_student_messages(ex["prompt"], condition="synthetic_intent"))
        for ex in examples
    ]
    outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)

    results = []
    for ex, output in zip(examples, outputs):
        generated_text = output.outputs[0].text.strip()
        reasoning, generated_intent, predicted_harm = _parse_reasoning_output(generated_text, with_intent=True)
        true_harm = ex.get(harm_column)
        results.append({
            "id": ex.get("id", ""),
            "prompt": ex["prompt"],
            "true_harm": true_harm,
            "true_harm_binary": map_harm_to_binary(true_harm),
            "generated_intent": generated_intent,
            "predicted_harm": predicted_harm,
            "reasoning": reasoning,
            "raw_generation": generated_text,
            "num_tokens": len(output.outputs[0].token_ids),
            "condition": "finetuned_reasoning_synthetic_intent",
        })
    return results


# ---------------------------------------------------------------------------
# Prior-work baseline runners (each loads its own model externally)
# ---------------------------------------------------------------------------

def run_llamaguard_classification(
    llm: LLM,
    tokenizer: AutoTokenizer,
    test_dataset,
    sampling_params: SamplingParams,
    harm_column: str,
) -> List[dict]:
    """
    LlamaGuard 4 baseline: native safety classification.

    LG4 requires content as a list of typed dicts (multimodal format);
    plain strings are not accepted by its chat template.
    """
    print("\n=== Running: LlamaGuard Classification ===")
    examples = list(test_dataset)

    llama_guard_params = SamplingParams(
        max_tokens=sampling_params.max_tokens,
        temperature=sampling_params.temperature,
        top_p=sampling_params.top_p,
        top_k=sampling_params.top_k,
        skip_special_tokens=True,
    )
    formatted_prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": [{"type": "text", "text": ex["prompt"]}]}],
            add_generation_prompt=True,
            tokenize=False,
        )
        for ex in examples
    ]
    outputs = llm.generate(formatted_prompts, llama_guard_params)

    results = []
    for ex, output in zip(examples, outputs):
        generated_text = output.outputs[0].text.strip()
        first_line = generated_text.split("\n")[0].lower().strip()
        if first_line == "safe":
            predicted_harm = "safe"
        elif first_line == "unsafe":
            predicted_harm = "harmful"
        else:
            predicted_harm = extract_harm_label(generated_text)

        true_harm = ex.get(harm_column)
        results.append({
            "id": ex.get("id", ""),
            "prompt": ex["prompt"],
            "true_harm": true_harm,
            "true_harm_binary": map_harm_to_binary(true_harm),
            "predicted_harm": predicted_harm,
            "raw_generation": generated_text,
            "num_tokens": len(output.outputs[0].token_ids),
            "condition": "llamaguard_classification",
        })
    return results


def run_wildguard_classification(
    llm: LLM,
    test_dataset,
    sampling_params: SamplingParams,
    harm_column: str,
) -> List[dict]:
    """
    WildGuard baseline: native safety classification.

    WildGuard outputs three labelled lines; we extract 'Harmful request: yes/no'.
    """
    print("\n=== Running: WildGuard Classification ===")
    examples = list(test_dataset)

    wildguard_params = SamplingParams(
        max_tokens=32,
        temperature=sampling_params.temperature,
        top_p=sampling_params.top_p,
        top_k=sampling_params.top_k,
        skip_special_tokens=True,
    )
    formatted_prompts = [
        _WILDGUARD_TEMPLATE.format(prompt=ex["prompt"], response="")
        for ex in examples
    ]
    outputs = llm.generate(formatted_prompts, wildguard_params)

    results = []
    for ex, output in zip(examples, outputs):
        generated_text = output.outputs[0].text.strip()
        predicted_harm = None
        for line in generated_text.splitlines():
            line_lower = line.lower()
            if line_lower.startswith("harmful request:"):
                answer = line_lower.split(":", 1)[1].strip()
                predicted_harm = "harmful" if answer == "yes" else ("safe" if answer == "no" else None)
                break
        if predicted_harm is None:
            predicted_harm = extract_harm_label(generated_text)

        true_harm = ex.get(harm_column)
        results.append({
            "id": ex.get("id", ""),
            "prompt": ex["prompt"],
            "true_harm": true_harm,
            "true_harm_binary": map_harm_to_binary(true_harm),
            "predicted_harm": predicted_harm,
            "raw_generation": generated_text,
            "num_tokens": len(output.outputs[0].token_ids),
            "condition": "wildguard_classification",
        })
    return results


def run_safeguard_classification(
    llm: LLM,
    tokenizer: AutoTokenizer,
    test_dataset,
    sampling_params: SamplingParams,
    harm_column: str,
) -> List[dict]:
    """
    GPT-OSS-Safeguard baseline: harmony response format.

    Writes reasoning in the `analysis` channel, then outputs 0 (safe) or 1 (harmful)
    in the `final` channel. skip_special_tokens=False is required to parse channel markers.
    """
    print("\n=== Running: GPT-OSS-Safeguard Classification ===")
    examples = list(test_dataset)

    safeguard_params = SamplingParams(
        max_tokens=sampling_params.max_tokens,
        temperature=sampling_params.temperature,
        top_p=sampling_params.top_p,
        top_k=sampling_params.top_k,
        skip_special_tokens=False,
    )
    formatted_prompts = [
        tokenizer.apply_chat_template(
            [
                {"role": "system", "content": _SAFEGUARD_SYSTEM_PROMPT},
                {"role": "user", "content": ex["prompt"]},
            ],
            add_generation_prompt=True,
            tokenize=False,
        )
        for ex in examples
    ]
    outputs = llm.generate(formatted_prompts, safeguard_params)

    results = []
    for ex, output in zip(examples, outputs):
        generated_text = output.outputs[0].text
        final_content = _parse_harmony_final(generated_text)

        if final_content == "0":
            predicted_harm = "safe"
        elif final_content == "1":
            predicted_harm = "harmful"
        else:
            has_zero = bool(re.search(r"\b0\b", final_content))
            has_one = bool(re.search(r"\b1\b", final_content))
            if has_zero and not has_one:
                predicted_harm = "safe"
            elif has_one and not has_zero:
                predicted_harm = "harmful"
            else:
                predicted_harm = extract_harm_label(generated_text)

        true_harm = ex.get(harm_column)
        results.append({
            "id": ex.get("id", ""),
            "prompt": ex["prompt"],
            "true_harm": true_harm,
            "true_harm_binary": map_harm_to_binary(true_harm),
            "predicted_harm": predicted_harm,
            "raw_generation": generated_text,
            "num_tokens": len(output.outputs[0].token_ids),
            "condition": "safeguard_classification",
        })
    return results


def run_guardreasoner_classification(
    llm: LLM,
    test_dataset,
    sampling_params: SamplingParams,
    harm_column: str,
) -> List[dict]:
    """
    GuardReasoner-8B baseline: prompt-only safety classification.

    Uses the model's native instruction format (plain string, no chat template).
    Passes response="None" since we evaluate prompt harm only (Task 1).
    Parses Task 1 answer: "unharmful" -> safe, "harmful" -> harmful.
    """
    print("\n=== Running: GuardReasoner Classification ===")
    examples = list(test_dataset)

    guardreasoner_params = SamplingParams(
        max_tokens=sampling_params.max_tokens,
        temperature=sampling_params.temperature,
        top_p=sampling_params.top_p,
        top_k=sampling_params.top_k,
        skip_special_tokens=True,
    )
    formatted_prompts = [
        _GUARDREASONER_INSTRUCT + f"Human user:\n{ex['prompt']}\n\nAI assistant:\nNone\n\n"
        for ex in examples
    ]
    outputs = llm.generate(formatted_prompts, guardreasoner_params)

    results = []
    for ex, output in zip(examples, outputs):
        generated_text = output.outputs[0].text.strip()

        # Extract Task 1 answer from the reasoning output.
        # Search within the Task 1 section (before Task 2, if present).
        predicted_harm = None
        task1_section = re.split(r"(?i)task\s*2", generated_text)[0]
        # Check for explicit answer lines first (e.g. "Answer: unharmful")
        m = re.search(r"(?i)answer[:\s]+(\w+)", task1_section)
        if m:
            answer = m.group(1).lower()
            if "unharmful" in answer:
                predicted_harm = "safe"
            elif "harmful" in answer:
                predicted_harm = "harmful"
        # Fall back to searching the Task 1 section for the label words
        if predicted_harm is None:
            section_lower = task1_section.lower()
            if "unharmful" in section_lower:
                predicted_harm = "safe"
            elif "harmful" in section_lower:
                predicted_harm = "harmful"

        true_harm = ex.get(harm_column)
        results.append({
            "id": ex.get("id", ""),
            "prompt": ex["prompt"],
            "true_harm": true_harm,
            "true_harm_binary": map_harm_to_binary(true_harm),
            "predicted_harm": predicted_harm,
            "raw_generation": generated_text,
            "num_tokens": len(output.outputs[0].token_ids),
            "condition": "guardreasoner_classification",
        })
    return results


def run_shieldgemma_classification(
    llm: LLM,
    test_dataset,
    sampling_params: SamplingParams,
    harm_column: str,
) -> List[dict]:
    """
    ShieldGemma-27B baseline: prompt-only safety classification.

    Uses the model's native prompt format (preamble + user turn + policy + epilogue).
    Generates 1 token; "Yes" = policy violation (harmful), "No" = not violating (safe).
    """
    print("\n=== Running: ShieldGemma Classification ===")
    examples = list(test_dataset)

    shieldgemma_params = SamplingParams(
        max_tokens=1,
        temperature=sampling_params.temperature,
        top_p=sampling_params.top_p,
        top_k=sampling_params.top_k,
        skip_special_tokens=True,
    )
    formatted_prompts = [
        _SHIELDGEMMA_PROMPT.format(prompt=ex["prompt"])
        for ex in examples
    ]
    outputs = llm.generate(formatted_prompts, shieldgemma_params)

    results = []
    for ex, output in zip(examples, outputs):
        generated_text = output.outputs[0].text.strip()
        first_token = generated_text.split()[0].lower() if generated_text.split() else ""
        if first_token == "yes":
            predicted_harm = "harmful"
        elif first_token == "no":
            predicted_harm = "safe"
        else:
            predicted_harm = extract_harm_label(generated_text)

        true_harm = ex.get(harm_column)
        results.append({
            "id": ex.get("id", ""),
            "prompt": ex["prompt"],
            "true_harm": true_harm,
            "true_harm_binary": map_harm_to_binary(true_harm),
            "predicted_harm": predicted_harm,
            "raw_generation": generated_text,
            "num_tokens": len(output.outputs[0].token_ids),
            "condition": "shieldgemma_classification",
        })
    return results


def run_nemotron_classification(
    llm: LLM,
    tokenizer: AutoTokenizer,
    test_dataset,
    sampling_params: SamplingParams,
    harm_column: str,
    thinking: bool = True,
) -> List[dict]:
    """
    Nemotron-Content-Safety-Reasoning-4B baseline: prompt-only safety classification.

    Appends /think or /no_think to toggle reasoning mode. Applies the model's chat
    template with multimodal-style content (required by the Gemma-3 base architecture).
    Parses "Prompt harm: harmful/unharmful" from the output.
    """
    think_token = "/think" if thinking else "/no_think"
    print(f"\n=== Running: Nemotron Classification (thinking={thinking}) ===")
    examples = list(test_dataset)

    formatted_prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": [{"type": "text", "text": _NEMOTRON_SAFETY_PROMPT.format(prompt=ex["prompt"], think_token=think_token)}]}],
            add_generation_prompt=True,
            tokenize=False,
        )
        for ex in examples
    ]
    outputs = llm.generate(formatted_prompts, sampling_params)

    results = []
    for ex, output in zip(examples, outputs):
        generated_text = output.outputs[0].text.strip()

        # Strip reasoning block if present
        text = generated_text
        if "<think>" in text and "</think>" in text:
            text = text.split("</think>")[-1].strip()

        predicted_harm = None
        for line in text.splitlines():
            line_lower = line.lower().strip()
            if line_lower.startswith("prompt harm:"):
                label = line_lower.split(":", 1)[1].strip()
                if "unharmful" in label:
                    predicted_harm = "safe"
                elif "harmful" in label:
                    predicted_harm = "harmful"
                break
        if predicted_harm is None:
            predicted_harm = extract_harm_label(generated_text)

        true_harm = ex.get(harm_column)
        results.append({
            "id": ex.get("id", ""),
            "prompt": ex["prompt"],
            "true_harm": true_harm,
            "true_harm_binary": map_harm_to_binary(true_harm),
            "predicted_harm": predicted_harm,
            "raw_generation": generated_text,
            "num_tokens": len(output.outputs[0].token_ids),
            "condition": "nemotron_classification",
        })
    return results


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def generate_model_intents(
    llm: LLM,
    lora_request: Optional[LoRARequest],
    test_dataset,
    sampling_params: SamplingParams,
) -> List[str]:
    """Generate intents using the fine-tuned model for later use with vanilla classifier."""
    print("\n=== Generating Model Intents ===")
    examples = list(test_dataset)
    prompts = [f"{ex['prompt']}\n" for ex in examples]
    outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)

    intents = []
    for output in outputs:
        generated_text = output.outputs[0].text.strip()
        intent, _ = extract_intent_and_harm(generated_text)
        intents.append(intent)

    print(f"Generated {len(intents)} intents")
    return intents


def run_condition_on_dataset(
    condition: str,
    llm: LLM,
    tokenizer: AutoTokenizer,
    test_dataset,
    sampling_params: SamplingParams,
    harm_column: str,
    has_intent: bool,
    model_intents: Optional[List[str]] = None,
    generation_lora_request: Optional[LoRARequest] = None,
    classification_lora_request: Optional[LoRARequest] = None,
    reasoning_classification_lora_request: Optional[LoRARequest] = None,
    reasoning_human_intent_lora_request: Optional[LoRARequest] = None,
    reasoning_synthetic_lora_request: Optional[LoRARequest] = None,
) -> List[dict]:
    """Dispatch a condition name to its corresponding run_* function."""
    if condition == "finetuned_generation":
        return run_finetuned_generation(
            llm, generation_lora_request, tokenizer, test_dataset, sampling_params, harm_column
        )
    elif condition == "finetuned_classification":
        return run_finetuned_classification(
            llm, classification_lora_request, tokenizer, test_dataset, sampling_params, harm_column
        )
    elif condition == "vanilla_classification":
        return run_vanilla_classification(llm, tokenizer, test_dataset, sampling_params, harm_column)
    elif condition == "vanilla_generation":
        return run_vanilla_generation(llm, tokenizer, test_dataset, sampling_params, harm_column)
    elif condition == "vanilla_with_human_intent":
        if not has_intent:
            print(f"  WARNING: Dataset has no intent column, skipping {condition}")
            return []
        return run_vanilla_classification_with_intent(
            llm, tokenizer, test_dataset, sampling_params, harm_column, intent_source="human"
        )
    elif condition == "vanilla_with_model_intent":
        if model_intents is None:
            print(f"  WARNING: No model intents provided, skipping {condition}")
            return []
        return run_vanilla_classification_with_intent(
            llm, tokenizer, test_dataset, sampling_params, harm_column,
            intent_source="model", model_intents=model_intents,
        )
    elif condition in (
        "llamaguard_classification", "wildguard_classification",
        "safeguard_classification", "guardreasoner_classification",
        "shieldgemma_classification", "nemotron_classification",
    ):
        return []  # handled separately in main with dedicated models
    elif condition == "zeroshot_cot_classification":
        return run_zeroshot_cot_classification(llm, tokenizer, test_dataset, sampling_params, harm_column)
    elif condition == "zeroshot_cot_generation":
        return run_zeroshot_cot_generation(llm, tokenizer, test_dataset, sampling_params, harm_column)
    elif condition == "zeroshot_cot_classification_with_intent":
        if not has_intent:
            print(f"  WARNING: Dataset has no intent column, skipping {condition}")
            return []
        return run_zeroshot_cot_classification_with_intent(
            llm, tokenizer, test_dataset, sampling_params, harm_column
        )
    elif condition == "finetuned_reasoning_classification":
        return run_finetuned_reasoning_classification(
            llm, reasoning_classification_lora_request, tokenizer, test_dataset, sampling_params, harm_column
        )
    elif condition == "finetuned_reasoning_human_intent":
        return run_finetuned_reasoning_human_intent(
            llm, reasoning_human_intent_lora_request, tokenizer, test_dataset, sampling_params, harm_column
        )
    elif condition == "finetuned_reasoning_synthetic_intent":
        return run_finetuned_reasoning_synthetic_intent(
            llm, reasoning_synthetic_lora_request, tokenizer, test_dataset, sampling_params, harm_column
        )
    else:
        raise ValueError(f"Unknown condition: {condition}")
