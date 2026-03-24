"""
Generate reasoning traces for WildGuard samples using vLLM.

All three Hub splits (train, validation, test) are processed in a single vLLM pass
to maximise GPU utilisation.  Outputs are saved to split-specific subdirectories so
consumers (distillation pipeline, safety experiment, etc.) can load the correct split.

Outputs:
  - data/reasoning_traces/<model>/train/raw_outputs.json
  - data/reasoning_traces/<model>/train/parsed_results.json
  - data/reasoning_traces/<model>/validation/raw_outputs.json
  - data/reasoning_traces/<model>/validation/parsed_results.json
  - data/reasoning_traces/<model>/test/raw_outputs.json
  - data/reasoning_traces/<model>/test/parsed_results.json

Usage:
    python scripts/generate_reasoning_traces.py
    python scripts/generate_reasoning_traces.py model.name=<other_model>
    python scripts/generate_reasoning_traces.py dataset.num_samples=100
    python scripts/generate_reasoning_traces.py conditions=[zeroshot_cot]
"""

import json
import re
import warnings
import os
from pathlib import Path

# Must be set before vLLM is imported so get_mp_context() picks it up.
# vLLM defaults to 'fork' on Linux, which crashes when CUDA is initialized
# in the parent before the EngineCore subprocess is forked.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import torch

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

# ---------------------------------------------------------------------------
# Monkey-patch: allow MIG / GPU UUIDs in CUDA_VISIBLE_DEVICES with vLLM.
#
# vLLM's Platform.device_id_to_physical_device_id() calls int() on the value
# from CUDA_VISIBLE_DEVICES, which crashes when it contains a UUID such as
#   CUDA_VISIBLE_DEVICES=MIG-ec73ed57-6aa0-541d-9909-e4f6518cbd33
#
# The patch detects UUID-format entries and resolves them to the correct
# physical GPU index via nvml before vLLM tries to cast them to int.
# ---------------------------------------------------------------------------
def _uuid_aware_device_id_to_physical(cls, device_id: int) -> int:
    device_control = getattr(cls, "device_control_env_var", "CUDA_VISIBLE_DEVICES")
    raw = os.environ.get(device_control, "")
    if not raw:
        return device_id
    entries = raw.split(",")
    entry = entries[device_id]
    if entry.lstrip("-").isdigit():
        return int(entry)
    # UUID path (MIG-* or GPU-*)
    from vllm.third_party import pynvml
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByUUID(entry)
    try:
        # MIG device → get the parent physical GPU
        parent = pynvml.nvmlDeviceGetDeviceHandleFromMigDeviceHandle(handle)
        return pynvml.nvmlDeviceGetIndex(parent)
    except pynvml.NVMLError:
        # Regular GPU UUID
        return pynvml.nvmlDeviceGetIndex(handle)

import vllm.platforms.interface as _vllm_platform_iface
_vllm_platform_iface.Platform.device_id_to_physical_device_id = classmethod(
    _uuid_aware_device_id_to_physical
)

# ---------------------------------------------------------------------------
# Second patch: vLLM inspects model architectures in a *fresh subprocess*
# (python -m vllm.model_executor.models.registry) to avoid CUDA init in the
# parent.  That subprocess inherits CUDA_VISIBLE_DEVICES=MIG-UUID but has no
# monkey-patch, so it still crashes.  The inspection only reads Python class
# metadata — it doesn't need a GPU at all — so we temporarily hide
# CUDA_VISIBLE_DEVICES from the subprocess env and restore it afterward.
# ---------------------------------------------------------------------------
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
# ---------------------------------------------------------------------------

from datasets import load_dataset
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
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
    OUTPUT_FORMAT_WITHOUT_INTENT,
    OUTPUT_FORMAT_WITH_INTENT,
    TEACHER_GROUND_TRUTH,
    build_student_prompt,
)


def _build_teacher_prompt(user_prompt: str, condition: str, **labels) -> str:
    """
    Build the teacher-facing prompt that includes ground-truth labels.

    Args:
        user_prompt: The raw human request.
        condition:   'without_intent' or 'with_intent'.
        **labels:    annotator_harmful_label (both conditions), intent (with_intent only).

    Returns:
        Formatted prompt string.
    """
    if condition == "without_intent":
        output_fmt = OUTPUT_FORMAT_WITHOUT_INTENT
        gt_block = TEACHER_GROUND_TRUTH["without_intent"].format(
            annotator_harmful_label=labels["annotator_harmful_label"]
        )
    elif condition == "with_intent":
        output_fmt = OUTPUT_FORMAT_WITH_INTENT
        gt_block = TEACHER_GROUND_TRUTH["with_intent"].format(
            intent=labels.get("intent", ""),
            annotator_harmful_label=labels["annotator_harmful_label"],
        )
    else:
        raise ValueError(f"Unknown teacher condition: {condition!r}")

    return f"{PREAMBLE}\n\nHuman user:\n{user_prompt}\n\n{gt_block}\n{output_fmt}"


# Condition 3: Zero-shot CoT — no ground truth labels, model must reason to the answer.
# Reuses build_student_prompt (same input structure the student model sees at inference).
def _build_zeroshot_cot_prompt(user_prompt: str, with_intent: bool = True) -> str:
    return build_student_prompt(user_prompt, with_intent=with_intent)


def parse_model_output(raw_text: str) -> dict:
    """
    Parse the model output to extract the intent, harm classification, and reasoning.

    Output format (in order): Reasoning → Prompt intent → Prompt harm

    If thinking mode is enabled the model wraps its internal chain-of-thought in
    <think>...</think> tags before emitting the structured output.  The think
    block is extracted and preserved in the returned dict, then stripped so
    the downstream regex only sees the structured portion.

    Returns a dict with:
        - prompt_intent: str or None
        - prompt_harm: str or None  ('harmful' | 'unharmful')
        - reasoning: str
        - thinking_trace: str  (empty string when thinking mode was not used)
    """
    # Extract and strip thinking CoT.
    # When thinking mode is on, vLLM strips the <think> *opening* token via
    # skip_special_tokens=True but leaves the </think> closing tag in the text.
    # So the output looks like: "<CoT text>...</think>\n\n<structured output>"
    # We also handle the (unlikely) case where both tags are present.
    thinking_trace = ""
    if '<think>' in raw_text:
        # Both tags present
        think_match = re.search(r'<think>(.*?)</think>', raw_text, flags=re.DOTALL)
        if think_match:
            thinking_trace = think_match.group(1).strip()
        text = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL).strip()
    elif '</think>' in raw_text:
        # Opening tag stripped as special token; everything before </think> is the CoT
        parts = raw_text.split('</think>', 1)
        thinking_trace = parts[0].strip()
        text = parts[1].strip()
    elif 'assistantfinal' in raw_text:
        # GPT-OSS-120B wraps its CoT with 'analysis' (open) and 'assistantfinal' (close).
        # Neither is stripped by skip_special_tokens, so split on the closing marker.
        parts = raw_text.split('assistantfinal', 1)
        thinking_trace = parts[0].removeprefix('analysis').strip()
        text = parts[1].strip()
    else:
        text = raw_text

    # Extract Reasoning field — stops before Intent:, Prompt intent:, or Prompt harm:
    reasoning = ""
    reasoning_match = re.search(
        r'Reasoning:\s*(.+?)(?=(?:Prompt )?intent:|Prompt harm:|$)',
        text, re.IGNORECASE | re.DOTALL,
    )
    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()

    # Extract Intent field (ends when Prompt harm: begins).
    # The output template uses "Intent:" — match that but also accept the legacy
    # "Prompt intent:" prefix in case an older teacher produced it.
    prompt_intent = None
    intent_match = re.search(
        r'(?:Prompt intent|Intent):\s*(.+?)(?=Prompt harm:|$)',
        text, re.IGNORECASE | re.DOTALL,
    )
    if intent_match:
        prompt_intent = intent_match.group(1).strip()

    # Extract Prompt harm field.
    # The model may echo the 4-category ground-truth labels ("Completely Harmful",
    # "Uncertain Safe", etc.) instead of the binary labels requested.  Capture the
    # full label on that line and apply substring-based binary mapping so that
    # "Uncertain Harmful" -> "harmful", "Completely Safe" -> "unharmful", etc.
    # "unharmful" must be checked before "harmful" because the latter is a substring.
    prompt_harm = None
    ph_match = re.search(r'Prompt harm:\s*([^\n]+)', text, re.IGNORECASE)
    if ph_match:
        label = ph_match.group(1).strip().lower()
        if 'unharmful' in label or 'safe' in label:
            prompt_harm = 'unharmful'
        elif 'harmful' in label:
            prompt_harm = 'harmful'

    return {
        "prompt_intent": prompt_intent,
        "prompt_harm": prompt_harm,
        "reasoning": reasoning,
        "thinking_trace": thinking_trace,
    }


def load_samples_for_split(data_cfg: dict, split: str) -> list:
    """
    Load samples from one Hub split of the annotated intents dataset.

    Args:
        data_cfg: The ``dataset`` block from reasoning_traces.yaml.
        split:    Hub split name: "train", "validation", or "test".

    Returns:
        List of dicts with keys: wildguard_id, prompt, prompt_harm_label, intent, split.
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

    # Deduplicate by Wildguard ID (train split may have multiple annotations per prompt)
    seen_ids = {}
    for row in annotated_ds:
        wg_id = row.get("Wildguard ID")
        if wg_id is not None and wg_id not in seen_ids:
            prompt = row.get("Prompt", "").strip()
            harm_label = row.get(harm_column)
            if prompt and harm_label is not None:
                seen_ids[wg_id] = {
                    "wildguard_id": wg_id,
                    "prompt": prompt,
                    "prompt_harm_label": harm_label,
                    "intent": row.get("Intent", ""),
                    "split": split,
                }

    available = list(seen_ids.values())
    print(f"    Unique samples with harm labels: {len(available)}")

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

@hydra.main(version_base=None, config_path="../configs/model_generation", config_name="reasoning_traces")
def main(cfg: DictConfig):
    """Main function to generate reasoning traces."""
    config = OmegaConf.to_container(cfg, resolve=True)

    model_cfg    = config.get("model", {})
    dataset_cfg  = config.get("dataset", {})
    gen_cfg      = config.get("generation", {})
    vllm_cfg     = config.get("vllm", {})
    paths_cfg    = config.get("paths", {})
    wandb_cfg    = config.get("wandb", {})
    thinking_mode = bool(config.get("thinking_mode", False))

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

    # Create model-specific subdirectory (use model name, replace slashes with dashes)
    model_name_clean = model_cfg["name"].replace("/", "-")
    output_dir = output_dir / model_name_clean
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get conditions to run
    conditions = config.get("conditions", ["without_intent"])
    if isinstance(conditions, str):
        conditions = [conditions]

    print(f"\nConditions to run: {conditions}")

    # Load samples from all splits (single vLLM pass covers everything)
    print("\n=== Loading samples from all splits ===")
    samples = load_all_samples(dataset_cfg)

    if not samples:
        print("No samples loaded. Exiting.")
        return

    # Print sample structure for verification
    print(f"\nSample keys: {list(samples[0].keys())}")

    # Load model
    model_name = model_cfg["name"]
    print(f"\n=== Loading Model: {model_name} ===")

    tp_size = vllm_cfg.get("tensor_parallel_size") or torch.cuda.device_count()
    print(f"Tensor parallel size: {tp_size} GPU(s)")

    llm = LLM(
        model=model_name,
        gpu_memory_utilization=vllm_cfg.get("gpu_memory_utilization", 0.95),
        max_model_len=vllm_cfg.get("max_model_len", 8192),
        max_num_seqs=vllm_cfg.get("max_num_seqs", 256),
        dtype=vllm_cfg.get("dtype", "float16"),
        enforce_eager=vllm_cfg.get("enforce_eager", True),
        tensor_parallel_size=tp_size,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Sampling parameters
    sampling_params = SamplingParams(
        max_tokens=gen_cfg.get("max_new_tokens", 4096),
        temperature=gen_cfg.get("temperature", 0.0),
        top_p=gen_cfg.get("top_p", 1.0),
        top_k=gen_cfg.get("top_k", -1),
        skip_special_tokens=True,
    )

    # Build prompts for each enabled condition
    print("\n=== Formatting prompts ===")

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

    condition_prompts = {}
    for condition in conditions:
        prompts = []
        if condition == "without_intent":
            for sample in samples:
                user_content = _build_teacher_prompt(
                    sample["prompt"], "without_intent",
                    annotator_harmful_label=sample["prompt_harm_label"],
                )
                prompts.append(format_prompt(user_content))
        elif condition == "with_intent":
            for sample in samples:
                user_content = _build_teacher_prompt(
                    sample["prompt"], "with_intent",
                    intent=sample["intent"],
                    annotator_harmful_label=sample["prompt_harm_label"],
                )
                prompts.append(format_prompt(user_content))
        elif condition == "zeroshot_cot":
            for sample in samples:
                user_content = _build_zeroshot_cot_prompt(sample["prompt"], with_intent=True)
                prompts.append(format_prompt(user_content))
        else:
            print(f"Warning: Unknown condition '{condition}', skipping.")
            continue

        condition_prompts[condition] = prompts
        print(f"  {condition}: {len(prompts)} prompts")

    # Combine all prompts for a single VLLM batch
    all_prompts = []
    prompt_to_condition = {}
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

    # Generate all at once
    print(f"\n=== Generating reasoning traces for {total_prompts} prompts ===")
    all_outputs = llm.generate(all_prompts, sampling_params)

    # Process results for each condition
    raw_outputs = []
    parsed_results = []

    for output_idx, (prompt, output) in enumerate(zip(all_prompts, all_outputs)):
        condition = prompt_to_condition[output_idx]
        sample_idx = output_idx % len(samples)
        sample = samples[sample_idx]

        raw_text = output.outputs[0].text.strip()
        parsed = parse_model_output(raw_text)

        raw_outputs.append({
            "wildguard_id": sample["wildguard_id"],
            "prompt": sample["prompt"],
            "intent": sample["intent"],
            "prompt_harm_label": sample["prompt_harm_label"],
            "split": sample["split"],
            "condition": condition,
            "raw_output": raw_text,
            "thinking_trace": parsed["thinking_trace"],
        })

        parsed_results.append({
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
        })

    # Save results partitioned by split
    print("\n=== Saving results ===")
    for split in ("train", "validation", "test"):
        split_raw     = [r for r in raw_outputs    if r["split"] == split]
        split_parsed  = [r for r in parsed_results if r["split"] == split]

        split_dir = output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)

        raw_path    = split_dir / "raw_outputs.json"
        parsed_path = split_dir / "parsed_results.json"

        with open(raw_path, "w") as f:
            json.dump(split_raw, f, indent=2, ensure_ascii=False)
        with open(parsed_path, "w") as f:
            json.dump(split_parsed, f, indent=2, ensure_ascii=False)

        print(f"  [{split}] {len(split_parsed)} samples → {split_dir}")

    # Print summary per condition × split
    for condition in conditions:
        for split in ("train", "validation", "test"):
            condition_results = [
                r for r in parsed_results
                if r["condition"] == condition and r["split"] == split
            ]
            total = len(condition_results)
            if total == 0:
                continue
            parsed_harm_count = sum(
                1 for r in condition_results
                if r["predicted"]["prompt_harm"] is not None
            )
            parsed_intent_count = sum(
                1 for r in condition_results
                if r["predicted"]["prompt_intent"] is not None
            )
            print(f"\n=== Summary ({condition} / {split}) ===")
            print(f"Total samples: {total}")
            print(f"Prompt harm parsed:   {parsed_harm_count}/{total}")
            print(f"Prompt intent parsed: {parsed_intent_count}/{total}")


if __name__ == "__main__":
    main()