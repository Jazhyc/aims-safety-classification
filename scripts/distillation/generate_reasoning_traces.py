"""
Generate reasoning traces for WildGuard samples.

Supports two backends, selected via ``model.backend``:
  - ``"vllm"``       (default) — loads the model locally via vLLM
  - ``"openrouter"`` — calls the OpenRouter API; requires the
                       ``OPENROUTER_API_KEY`` environment variable

All three Hub splits (train, validation, test) are processed in a single pass
to maximise throughput.  Outputs are saved to split-specific subdirectories so
consumers (distillation pipeline, safety experiment, etc.) can load the correct
split.

Outputs:
  - data/reasoning_traces/<model>/train/raw_outputs.json
  - data/reasoning_traces/<model>/train/parsed_results.json
  - data/reasoning_traces/<model>/validation/raw_outputs.json
  - data/reasoning_traces/<model>/validation/parsed_results.json
  - data/reasoning_traces/<model>/test/raw_outputs.json
  - data/reasoning_traces/<model>/test/parsed_results.json

Usage:
    python scripts/distillation/generate_reasoning_traces.py
    python scripts/distillation/generate_reasoning_traces.py model.name=<model> model.backend=openrouter
    python scripts/distillation/generate_reasoning_traces.py dataset.num_samples=100
    python scripts/distillation/generate_reasoning_traces.py conditions=[zeroshot_cot]
"""

import json
import re
import time
import warnings
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

from datasets import load_dataset
import hydra
from omegaconf import DictConfig, OmegaConf

from dotenv import load_dotenv
load_dotenv()

try:
    import wandb as _wandb
except ImportError:
    _wandb = None

from intention_jailbreak.model_generation.prompt_templates import (
    PREAMBLE,
    PREAMBLE_NO_INTENT,
    OUTPUT_FORMAT_NO_INTENT,
    OUTPUT_FORMAT_WITH_INTENT,
    TEACHER_GROUND_TRUTH,
    build_student_prompt,
)


# ── Response schema for structured output parsing ────────────────────────────
# Defines how to extract thinking blocks, reasoning, intent, and harm predictions
# from the model output using HF transformers' parse_response schema-based parser.
REASONING_TRACE_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "role": {"const": "assistant"},
        "thinking": {
            "type": "string",
            "x-regex": r"^(.*?)(?=Reasoning:|$)",
        },
        "reasoning": {
            "type": "string",
            "x-regex": r"Reasoning:\s*(.+?)(?=(?:Prompt )?intent:|Prompt harm:|$)",
        },
        "prompt_intent": {
            "type": "string",
            "x-regex": r"(?:Prompt intent|Intent):\s*(.+?)(?=Prompt harm:|$)",
        },
        "prompt_harm": {
            "type": "string",
            "x-regex": r"Prompt harm:\s*([^\n]+)",
        },
    },
}


# ── Prompt builders ────────────────────────────────────────────────────────────

def _build_teacher_prompt(user_prompt: str, condition: str, **labels) -> str:
    """
    Build the teacher-facing prompt that includes ground-truth labels.

    Args:
        user_prompt: The raw human request.
        condition:   'no_intent', 'synthetic_intent', or 'human_intent'.
        **labels:    annotator_harmful_label (all conditions), intent (human_intent only).

    Returns:
        Formatted prompt string.
    """
    if condition == "no_intent":
        preamble = PREAMBLE_NO_INTENT
        output_fmt = OUTPUT_FORMAT_NO_INTENT
        gt_block = TEACHER_GROUND_TRUTH["no_intent"].format(
            annotator_harmful_label=labels["annotator_harmful_label"]
        )
    elif condition == "synthetic_intent":
        preamble = PREAMBLE
        output_fmt = OUTPUT_FORMAT_WITH_INTENT
        gt_block = TEACHER_GROUND_TRUTH["synthetic_intent"].format(
            annotator_harmful_label=labels["annotator_harmful_label"]
        )
    elif condition == "human_intent":
        preamble = PREAMBLE
        output_fmt = OUTPUT_FORMAT_WITH_INTENT
        gt_block = TEACHER_GROUND_TRUTH["human_intent"].format(
            intent=labels.get("intent", ""),
            annotator_harmful_label=labels["annotator_harmful_label"],
        )
    else:
        raise ValueError(f"Unknown teacher condition: {condition!r}")

    return f"{preamble}\n\nHuman user:\n{user_prompt}\n\n{gt_block}\n{output_fmt}"


def _build_zeroshot_cot_prompt(user_prompt: str, condition: str = "human_intent") -> str:
    return build_student_prompt(user_prompt, condition=condition)


def _user_content_for_condition(condition: str, sample: dict) -> str:
    """Return the user-facing prompt string for a given condition and sample."""
    if condition == "no_intent":
        return _build_teacher_prompt(
            sample["prompt"], "no_intent",
            annotator_harmful_label=sample["prompt_harm_label"],
        )
    if condition == "synthetic_intent":
        return _build_teacher_prompt(
            sample["prompt"], "synthetic_intent",
            annotator_harmful_label=sample["prompt_harm_label"],
        )
    if condition == "human_intent":
        return _build_teacher_prompt(
            sample["prompt"], "human_intent",
            intent=sample["intent"],
            annotator_harmful_label=sample["prompt_harm_label"],
        )
    if condition == "zeroshot_cot":
        return _build_zeroshot_cot_prompt(sample["prompt"], condition="human_intent")
    raise ValueError(f"Unknown condition: {condition!r}")


# ── Thinking token detection ───────────────────────────────────────────────────

def setup_response_schema(tokenizer) -> None:
    """
    Set up the response schema on the tokenizer for structured output parsing.

    This schema enables tokenizer.parse_response() to extract thinking blocks,
    reasoning, intent, and harm predictions from model output using regex patterns.
    """
    tokenizer.response_schema = REASONING_TRACE_RESPONSE_SCHEMA




# ── Output parser ──────────────────────────────────────────────────────────────

def parse_model_output(
    raw_text: str,
    tokenizer=None,
    pre_extracted_thinking: str | None = None,
) -> dict:
    """
    Parse model output into structured fields (intent, harm, reasoning, thinking).

    Uses the tokenizer's schema-based parse_response method (HF transformers ≥ 5.5.0)
    which applies regex patterns to extract structured fields from the raw text.

    Args:
        raw_text:               Raw model output text.
        tokenizer:              Tokenizer with response_schema set. If None, uses
                                fallback regex-based extraction.
        pre_extracted_thinking: Pre-extracted thinking block (from vLLM reasoning_content
                                or OpenRouter). Bypasses text-based extraction.

    Returns:
        dict with keys: prompt_intent, prompt_harm, reasoning, thinking_trace.
    """
    # Resolve the thinking block, then parse the post-thinking text with
    # _parse_content_fields (LAST-match, IGNORECASE — robust against residual
    # "Intent:"/"Reasoning:" markers that the model may emit during its
    # thinking phase, e.g. gpt-oss harmony "analysis...assistantfinal..." traces).
    #
    # We deliberately do NOT use tokenizer.parse_response here: HF's schema
    # parser matches FIRST occurrence per regex and applies no IGNORECASE flag,
    # which causes "Intent:" markers inside residual thinking to be picked up
    # as the prompt_intent field and reasoning to over-capture.
    if pre_extracted_thinking is None:
        # Auto-detect thinking as everything before the first "Reasoning:" marker.
        # Works for gpt-oss harmony format (analysis section terminated by
        # assistantfinal then "Reasoning:") and for non-thinking models (no
        # leading content → empty thinking, identical to the fallback path).
        m = re.search(r'Reasoning:', raw_text, re.IGNORECASE)
        if m and m.start() > 0:
            thinking = raw_text[:m.start()]
            content = raw_text[m.start():]
        else:
            thinking = ""
            content = raw_text
    else:
        thinking = pre_extracted_thinking
        content = raw_text.replace(pre_extracted_thinking, "", 1).strip()

    parsed = _parse_content_fields(content)
    parsed["thinking_trace"] = thinking.strip()
    return parsed


def _parse_content_fields(text: str) -> dict:
    """
    Extract structured fields (intent, harm, reasoning) from content using regex.

    Fallback parser for when tokenizer.parse_response is unavailable.

    If multiple instances exist (e.g., template placeholder + actual output),
    prefers the LAST match to avoid capturing unfilled template values.
    """
    # Extract Reasoning field — stops before Intent:, Prompt intent:, or Prompt harm:
    # Use LAST match to skip template placeholders if multiple "Reasoning:" exist.
    reasoning = ""
    reasoning_matches = list(re.finditer(
        r'Reasoning:\s*(.+?)(?=(?:Prompt )?intent:|Prompt harm:|$)',
        text, re.IGNORECASE | re.DOTALL,
    ))
    if reasoning_matches:
        reasoning = reasoning_matches[-1].group(1).strip()

    # Extract Intent field (ends when Prompt harm: begins)
    # Use LAST match to avoid template placeholders.
    prompt_intent = None
    intent_matches = list(re.finditer(
        r'(?:Prompt intent|Intent):\s*(.+?)(?=Prompt harm:|$)',
        text, re.IGNORECASE | re.DOTALL,
    ))
    if intent_matches:
        prompt_intent = intent_matches[-1].group(1).strip()

    # Extract Prompt harm field with binary normalisation
    # Use LAST match to avoid template placeholders.
    prompt_harm = None
    ph_matches = list(re.finditer(r'Prompt harm:\s*([^\n]+)', text, re.IGNORECASE))
    if ph_matches:
        prompt_harm = _normalize_harm_label(ph_matches[-1].group(1))

    return {
        "prompt_intent":  prompt_intent,
        "prompt_harm":    prompt_harm,
        "reasoning":      reasoning,
        "thinking_trace": "",
    }


def _normalize_harm_label(label: str) -> str | None:
    """Normalize harm label to binary: 'harmful', 'unharmful', or None."""
    if not label:
        return None
    label = label.strip().lower()
    if 'unharmful' in label or 'safe' in label:
        return 'unharmful'
    elif 'harmful' in label:
        return 'harmful'
    return None


# ── Dataset loading ────────────────────────────────────────────────────────────

def load_samples_for_split(data_cfg: dict, split: str) -> list:
    """
    Load samples from one Hub split of the annotated intents dataset.

    Each annotation row is treated as a separate training sample.  The train
    split may contain multiple annotations (different annotators) for the same
    WildGuard prompt; all are included so their distinct harm labels and intents
    are represented in the traces.

    Args:
        data_cfg: The ``dataset`` block from reasoning_traces.yaml.
        split:    Hub split name: "train", "validation", or "test".

    Returns:
        List of dicts with keys: annotation_id, wildguard_id, prompt,
        prompt_harm_label, intent, split.
    """
    import random

    num_samples = data_cfg.get("num_samples")
    seed = data_cfg.get("seed", 42)

    ai_cfg = data_cfg.get("annotated_intents", {})
    ai_name = ai_cfg.get("dataset_name", "Jazhyc/wildguard-annotated-intents")
    ai_subset = ai_cfg.get("subset")
    harm_column = ai_cfg.get("harm_column", "Annotator Harm")

    print(f"  Loading split='{split}' from {ai_name} ...")
    if ai_subset:
        annotated_ds = load_dataset(ai_name, ai_subset, split=split)
    else:
        annotated_ds = load_dataset(ai_name, split=split)
    print(f"    Raw size: {len(annotated_ds)}")

    # Keep every annotation as a separate sample (no wildguard_id deduplication).
    # The annotation_id (dataset "ID" column) uniquely identifies each row and
    # is included in output entries so consumers can distinguish multiple
    # annotations for the same prompt.
    available = []
    for row in annotated_ds:
        ann_id = row.get("ID")
        wg_id = row.get("Wildguard ID")
        prompt = row.get("Prompt", "").strip()
        harm_label = row.get(harm_column)
        if prompt and harm_label is not None:
            available.append({
                "annotation_id": ann_id,
                "wildguard_id": wg_id,
                "prompt": prompt,
                "prompt_harm_label": harm_label,
                "intent": row.get("Intent", ""),
                "split": split,
            })

    print(f"    Samples with harm labels: {len(available)}")

    rng = random.Random(seed)
    rng.shuffle(available)
    samples = available if num_samples is None else available[:num_samples]
    return samples


def load_all_samples(data_cfg: dict) -> list:
    """
    Load samples from all three Hub splits and return them as a single list,
    each sample tagged with its split name.
    """
    all_samples = []
    for split in ("train", "validation", "test"):
        split_samples = load_samples_for_split(data_cfg, split)
        all_samples.extend(split_samples)
        print(f"    → {len(split_samples)} samples from '{split}'")
    print(f"\nTotal samples across all splits: {len(all_samples)}")
    return all_samples


# ── Result builders (shared schema) ───────────────────────────────────────────

def _make_raw_entry(sample: dict, condition: str, raw_text: str, parsed: dict) -> dict:
    return {
        "annotation_id": sample.get("annotation_id"),
        "wildguard_id": sample["wildguard_id"],
        "prompt": sample["prompt"],
        "intent": sample["intent"],
        "prompt_harm_label": sample["prompt_harm_label"],
        "split": sample["split"],
        "condition": condition,
        "raw_output": raw_text,
        "thinking_trace": parsed["thinking_trace"],
    }


def _make_parsed_entry(sample: dict, condition: str, parsed: dict) -> dict:
    return {
        "annotation_id": sample.get("annotation_id"),
        "wildguard_id": sample["wildguard_id"],
        "prompt": sample["prompt"],
        "split": sample["split"],
        "condition": condition,
        "ground_truth": {
            "intent": sample["intent"],
            "prompt_harm_label": sample["prompt_harm_label"],
        },
        "predicted": {
            "prompt_intent": parsed["prompt_intent"],
            "prompt_harm": parsed["prompt_harm"],
        },
        "reasoning": parsed["reasoning"],
    }


# ── vLLM backend ───────────────────────────────────────────────────────────────

def _generate_with_vllm(
    samples: list,
    conditions: list[str],
    config: dict,
) -> tuple[list, list]:
    """Generate reasoning traces using a local vLLM instance."""
    # Must be set before any vLLM module is imported so get_mp_context() picks it up.
    # vLLM defaults to 'fork' on Linux, which crashes when CUDA is initialized
    # in the parent before the EngineCore subprocess is forked.
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    import torch

    # ── Monkey-patch 1: allow MIG / GPU UUIDs in CUDA_VISIBLE_DEVICES ─────
    #
    # vLLM's Platform.device_id_to_physical_device_id() calls int() on the
    # value from CUDA_VISIBLE_DEVICES, which crashes when it contains a UUID
    # such as CUDA_VISIBLE_DEVICES=MIG-ec73ed57-6aa0-541d-9909-e4f6518cbd33.
    # The patch detects UUID-format entries and resolves them to the correct
    # physical GPU index via nvml before vLLM tries to cast them to int.
    def _uuid_aware_device_id_to_physical(cls, device_id: int) -> int:
        device_control = getattr(cls, "device_control_env_var", "CUDA_VISIBLE_DEVICES")
        raw = os.environ.get(device_control, "")
        if not raw:
            return device_id
        entries = raw.split(",")
        entry = entries[device_id]
        if entry.lstrip("-").isdigit():
            return int(entry)
        from vllm.third_party import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByUUID(entry)
        try:
            parent = pynvml.nvmlDeviceGetDeviceHandleFromMigDeviceHandle(handle)
            return pynvml.nvmlDeviceGetIndex(parent)
        except pynvml.NVMLError:
            return pynvml.nvmlDeviceGetIndex(handle)

    import vllm.platforms.interface as _vllm_platform_iface
    _vllm_platform_iface.Platform.device_id_to_physical_device_id = classmethod(
        _uuid_aware_device_id_to_physical
    )

    # ── Monkey-patch 2: hide MIG UUID from vLLM's model-registry subprocess ─
    #
    # vLLM inspects model architectures in a fresh subprocess
    # (python -m vllm.model_executor.models.registry) which inherits
    # CUDA_VISIBLE_DEVICES=MIG-UUID but has no patch, so it still crashes.
    # Temporarily hide the var during the subprocess call.
    import vllm.model_executor.models.registry as _vllm_registry
    _original_run_in_subprocess = _vllm_registry._run_in_subprocess

    def _patched_run_in_subprocess(fn):
        cuda_vis = os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        try:
            return _original_run_in_subprocess(fn)
        finally:
            if cuda_vis is not None:
                os.environ["CUDA_VISIBLE_DEVICES"] = cuda_vis

    _vllm_registry._run_in_subprocess = _patched_run_in_subprocess
    # ──────────────────────────────────────────────────────────────────────

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    model_cfg    = config.get("model", {})
    gen_cfg      = config.get("generation", {})
    vllm_cfg     = config.get("vllm", {})
    thinking_mode = bool(config.get("thinking_mode", False))

    model_name = model_cfg["name"]
    print(f"\n=== Loading Model: {model_name} ===")

    tp_size = vllm_cfg.get("tensor_parallel_size") or torch.cuda.device_count()
    print(f"Tensor parallel size: {tp_size} GPU(s)")

    llm_kwargs = dict(
        model=model_name,
        gpu_memory_utilization=vllm_cfg.get("gpu_memory_utilization", 0.95),
        max_model_len=vllm_cfg.get("max_model_len", 8192),
        max_num_seqs=vllm_cfg.get("max_num_seqs", 256),
        dtype=vllm_cfg.get("dtype", "float16"),
        enforce_eager=vllm_cfg.get("enforce_eager", True),
        tensor_parallel_size=tp_size,
    )
    flash_attn_version = vllm_cfg.get("flash_attn_version", None)
    if flash_attn_version is not None:
        llm_kwargs["attention_config"] = {"flash_attn_version": flash_attn_version}
    llm = LLM(**llm_kwargs)

    # Load tokenizer and set up response schema for structured parsing
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    setup_response_schema(tokenizer)
    if thinking_mode:
        print(f"Thinking mode: enabled. Using schema-based parsing for structured output")

    sampling_params = SamplingParams(
        max_tokens=gen_cfg.get("max_new_tokens", 4096),
        temperature=gen_cfg.get("temperature", 0.0),
        top_p=gen_cfg.get("top_p", 1.0),
        top_k=gen_cfg.get("top_k", -1),
        skip_special_tokens=True,
    )

    def format_prompt(user_content: str) -> str:
        messages = [{"role": "user", "content": user_content}]
        try:
            return tokenizer.apply_chat_template(
                messages, add_generation_prompt=True,
                tokenize=False, enable_thinking=thinking_mode,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False,
            )

    # Build prompts for each enabled condition
    print("\n=== Formatting prompts ===")
    condition_prompts = {}
    for condition in conditions:
        prompts = []
        for sample in samples:
            try:
                user_content = _user_content_for_condition(condition, sample)
            except ValueError:
                print(f"Warning: Unknown condition '{condition}', skipping.")
                break
            prompts.append(format_prompt(user_content))
        else:
            condition_prompts[condition] = prompts
            print(f"  {condition}: {len(prompts)} prompts")

    # Combine all prompts for a single vLLM batch
    all_prompts: list[str] = []
    prompt_to_condition: dict[int, str] = {}
    for condition in conditions:
        if condition not in condition_prompts:
            continue
        start_idx = len(all_prompts)
        all_prompts.extend(condition_prompts[condition])
        end_idx = len(all_prompts)
        for idx in range(start_idx, end_idx):
            prompt_to_condition[idx] = condition

    total_prompts = len(all_prompts)
    print(f"  Total batch size: {total_prompts}")

    print(f"\n=== Generating {total_prompts} reasoning traces ===")
    all_outputs = llm.generate(all_prompts, sampling_params)

    raw_outputs: list[dict] = []
    parsed_results: list[dict] = []

    for output_idx, (prompt, output) in enumerate(zip(all_prompts, all_outputs)):
        condition = prompt_to_condition[output_idx]
        sample = samples[output_idx % len(samples)]

        completion = output.outputs[0]
        raw_text = completion.text.strip()
        # Use vLLM's built-in reasoning extraction when available (requires
        # reasoning_backend set on the LLM), otherwise use schema-based parsing.
        pre_extracted = getattr(completion, "reasoning_content", None)
        parsed = parse_model_output(raw_text, tokenizer=tokenizer, pre_extracted_thinking=pre_extracted)

        raw_outputs.append(_make_raw_entry(sample, condition, raw_text, parsed))
        parsed_results.append(_make_parsed_entry(sample, condition, parsed))

    return raw_outputs, parsed_results


# ── OpenRouter backend ─────────────────────────────────────────────────────────

def _generate_with_openrouter(
    samples: list,
    conditions: list[str],
    config: dict,
) -> tuple[list, list]:
    """
    Generate reasoning traces via the OpenRouter API.

    Requires the ``OPENROUTER_API_KEY`` environment variable.  Uses the
    standard OpenAI-compatible endpoint at https://openrouter.ai/api/v1.

    Thinking extraction (in priority order):
      1. ``reasoning`` field on the API response message object — some providers
         return the thinking block separately via this OpenRouter extension.
      2. Text-based delimiter detection — inspects the response content for
         known thinking tokens (<think>/</think>, <|thinking|>/<|/thinking|>).
    """
    from openai import OpenAI

    model_cfg = config.get("model", {})
    gen_cfg   = config.get("generation", {})
    or_cfg    = config.get("openrouter", {})

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. "
            "Export it in your shell or add it to your .env file."
        )

    extra_headers: dict[str, str] = {}
    if or_cfg.get("site_url"):
        extra_headers["HTTP-Referer"] = or_cfg["site_url"]
    if or_cfg.get("app_name"):
        extra_headers["X-Title"] = or_cfg["app_name"]

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        **({"default_headers": extra_headers} if extra_headers else {}),
    )

    model_name  = model_cfg["name"]
    max_tokens  = gen_cfg.get("max_new_tokens", 4096)
    temperature = gen_cfg.get("temperature", 0.0)
    max_workers = or_cfg.get("max_workers", 8)
    max_retries = or_cfg.get("max_retries", 3)

    # Build flat task list: one entry per (condition, sample) pair
    tasks = [
        (condition, sample)
        for condition in conditions
        for sample in samples
    ]
    total = len(tasks)
    print(f"\n=== Generating {total} reasoning traces via OpenRouter ({model_name}) ===")
    print(f"  max_workers={max_workers}  max_retries={max_retries}")

    def call_api(task_idx: int, condition: str, sample: dict):
        messages = [{"role": "user", "content": _user_content_for_condition(condition, sample)}]
        last_exc: Exception | None = None
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                msg = response.choices[0].message
                content = msg.content or ""

                # Try OpenRouter's extended reasoning field before falling back to
                # schema-based parsing.
                reasoning = getattr(msg, "reasoning", None)
                if reasoning is None and hasattr(msg, "model_extra"):
                    reasoning = (msg.model_extra or {}).get("reasoning")

                parsed = parse_model_output(
                    content, tokenizer=None, pre_extracted_thinking=reasoning,
                )
                return task_idx, condition, sample, content, parsed

            except Exception as e:
                last_exc = e
                if attempt < max_retries - 1:
                    wait = 2 ** attempt
                    print(
                        f"  [retry {attempt + 1}/{max_retries}] "
                        f"{sample['wildguard_id']}/{condition}: {e} — {wait}s"
                    )
                    time.sleep(wait)

        raise RuntimeError(
            f"All {max_retries} attempts failed for "
            f"{sample['wildguard_id']}/{condition}"
        ) from last_exc

    raw_outputs: list[dict] = []
    parsed_results: list[dict] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(call_api, i, cond, samp): i
            for i, (cond, samp) in enumerate(tasks)
        }
        with tqdm(total=total, desc="OpenRouter requests", unit="req") as pbar:
            for future in as_completed(futures):
                _, condition, sample, raw_text, parsed = future.result()
                raw_outputs.append(_make_raw_entry(sample, condition, raw_text, parsed))
                parsed_results.append(_make_parsed_entry(sample, condition, parsed))
                pbar.update(1)

    return raw_outputs, parsed_results


# ── Entry point ────────────────────────────────────────────────────────────────

@hydra.main(version_base=None, config_path="../../configs/experiments", config_name="reasoning_traces")
def main(cfg: DictConfig):
    """Main function to generate reasoning traces."""
    config = OmegaConf.to_container(cfg, resolve=True)

    model_cfg   = config.get("model", {})
    dataset_cfg = config.get("dataset", {})
    paths_cfg   = config.get("paths", {})
    wandb_cfg   = config.get("wandb", {})
    conditions  = config.get("conditions", ["no_intent"])
    if isinstance(conditions, str):
        conditions = [conditions]

    backend = model_cfg.get("backend", "vllm")

    if wandb_cfg.get("enabled", False) and _wandb is not None:
        _wandb.init(
            entity=wandb_cfg.get("entity"),
            project=wandb_cfg.get("project", "reasoning-traces-generation"),
            name=wandb_cfg.get("run_name"),
            tags=wandb_cfg.get("tags", []),
            dir=str(Path(hydra.utils.get_original_cwd()) / wandb_cfg.get("dir", "logs/wandb")),
            mode=wandb_cfg.get("mode", "online"),
            config=config,
        )

    output_dir = Path(paths_cfg.get("output_dir", "data/reasoning_traces"))
    # Per-model slug overrides. Must stay in sync with _TEACHER_SLUG_OVERRIDES in
    # run_distillation_pipeline.py. Models not listed fall back to replace("/", "-").
    _TEACHER_SLUG_OVERRIDES = {
        "cyankiwi/gemma-4-31B-it-AWQ-4bit": "cyankiwi-gemma-4-31b",
    }
    model_name_clean = _TEACHER_SLUG_OVERRIDES.get(
        model_cfg["name"], model_cfg["name"].replace("/", "-")
    )
    output_dir = output_dir / model_name_clean
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nConditions : {conditions}")
    print(f"Backend    : {backend}")

    samples_json = dataset_cfg.get("samples_json")
    if samples_json:
        print(f"\n=== Loading samples from {samples_json} (overrides Hub loading) ===")
        with open(samples_json) as f:
            samples = json.load(f)
        print(f"  Loaded {len(samples)} samples")
        if samples:
            missing = [k for k in ("prompt", "prompt_harm_label", "split") if k not in samples[0]]
            if missing:
                raise ValueError(
                    f"samples_json records are missing required keys: {missing}. "
                    f"Found keys: {list(samples[0].keys())}"
                )
    else:
        print("\n=== Loading samples from all splits ===")
        samples = load_all_samples(dataset_cfg)
    if not samples:
        print("No samples loaded. Exiting.")
        return
    print(f"\nSample keys: {list(samples[0].keys())}")

    # ── Dispatch to backend ────────────────────────────────────────────────
    if backend == "openrouter":
        raw_outputs, parsed_results = _generate_with_openrouter(samples, conditions, config)
    else:
        raw_outputs, parsed_results = _generate_with_vllm(samples, conditions, config)

    # ── Save results partitioned by split (identical for both backends) ────
    print("\n=== Saving results ===")
    for split in ("train", "validation", "test"):
        split_raw    = [r for r in raw_outputs    if r["split"] == split]
        split_parsed = [r for r in parsed_results if r["split"] == split]

        split_dir = output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)

        with open(split_dir / "raw_outputs.json", "w") as f:
            json.dump(split_raw, f, indent=2, ensure_ascii=False)
        with open(split_dir / "parsed_results.json", "w") as f:
            json.dump(split_parsed, f, indent=2, ensure_ascii=False)

        print(f"  [{split}] {len(split_parsed)} samples → {split_dir}")

    # ── Per-condition summary ──────────────────────────────────────────────
    for condition in conditions:
        for split in ("train", "validation", "test"):
            cond_results = [
                r for r in parsed_results
                if r["condition"] == condition and r["split"] == split
            ]
            total = len(cond_results)
            if total == 0:
                continue
            harm_ok   = sum(1 for r in cond_results if r["predicted"]["prompt_harm"] is not None)
            intent_ok = sum(1 for r in cond_results if r["predicted"]["prompt_intent"] is not None)
            print(f"\n=== Summary ({condition} / {split}) ===")
            print(f"Total: {total}  harm parsed: {harm_ok}  intent parsed: {intent_ok}")


if __name__ == "__main__":
    main()
