"""
Re-evaluate distillation sweep adapters and update W&B summaries in-place.

Fixes the bug where synthetic_intent and human_intent adapters were evaluated
with the classification prompt (no intent asked) instead of the generation
prompt, causing their val_harm_f1 scores to be artificially similar.

For each finished distillation sweep run, this script:
  1. Reads the val traces path and condition from the W&B run config.
  2. Loads the val dataset and formats prompts with the correct intent-aware template.
  3. Runs vLLM inference using the saved LoRA adapter.
  4. Computes harm F1 and updates run.summary["val_harm_f1"] via the W&B API.

Runs are grouped by student model so vLLM is loaded once per student.

Usage:
    python scripts/distillation/revalidate_sweep.py --dry-run
    python scripts/distillation/revalidate_sweep.py
    python scripts/distillation/revalidate_sweep.py --condition no_intent synthetic_intent human_intent
    python scripts/distillation/revalidate_sweep.py --traces-version v7
"""

import argparse
import multiprocessing
import os
import sys
from collections import defaultdict
from pathlib import Path

# Must be set before any CUDA / vLLM / torch imports so that vLLM's engine
# core subprocess uses spawn rather than fork (fork + CUDA = crash on Linux).
multiprocessing.set_start_method("spawn", force=True)

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import wandb

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ── Sweep dimensions (must match submit_distillation_sweep.py) ────────────────

TEACHER_MODELS = [
    #("cyankiwi/Mistral-Small-4-119B-2603-AWQ-4bit",  "cyankiwi-Mistral-Small-4-119B-2603-AWQ-4bit"),
    #("cyankiwi/Qwen3.5-122B-A10B-AWQ-4bit",          "cyankiwi-Qwen3.5-122B-A10B-AWQ-4bit"),
    ("openai/gpt-oss-120b",                           "openai-gpt-oss-120b"),
    #("RedHatAI/gemma-3-27b-it-quantized.w4a16",       "RedHatAI-gemma-3-27b-it-quantized.w4a16"),
]

STUDENT_MODELS = [
    # ("meta-llama/Llama-3.1-8B-Instruct",              "llama-3.1-8b"),
    ("google/gemma-3-12b-it",                          "gemma-3-12b"),
    #("mistralai/Ministral-3-14B-Reasoning-2512",       "ministral-3-14b-reasoning"),
    #("Qwen/Qwen3.5-9B",                                "qwen3.5-9b"),
]

DEFAULT_CONDITIONS = ["no_intent", "synthetic_intent", "human_intent"]
DEFAULT_TRACES_VERSION = "v7"

# Intent-producing conditions that require the generation eval prompt
INTENT_CONDITIONS = {"synthetic_intent", "human_intent"}


# ── Dataset helpers ───────────────────────────────────────────────────────────

def build_val_dataset(val_traces_path: str, condition: str):
    """
    Load val traces and format prompts with the condition-correct template.
    Mirrors the create_prompt_completion logic in run_causal_flow.
    """
    from intention_jailbreak.model_generation.causal import load_reasoning_traces_dataset
    from intention_jailbreak.model_generation.prompt_templates import build_student_prompt

    data_cfg = {
        "reasoning_traces_path": val_traces_path,
        "reasoning_traces_condition": condition,
    }
    raw = load_reasoning_traces_dataset(data_cfg)

    def apply_template(examples):
        prompts = [
            build_student_prompt(p, condition=condition) + "\n"
            for p in examples["prompt"]
        ]
        return {"prompt": prompts}

    return raw.map(apply_template, batched=True)


# ── vLLM inference ────────────────────────────────────────────────────────────

def run_vllm_eval(llm, lora_request, val_dataset, condition: str) -> float | None:
    """
    Generate predictions for val_dataset and return harm F1.
    Returns None if no valid harm pairs can be parsed.
    """
    from vllm import SamplingParams
    from intention_jailbreak.model_generation.causal import parse_reasoning_output
    from intention_jailbreak.model_generation.evaluate_generations import compute_harm_f1

    with_intent = condition in INTENT_CONDITIONS

    sampling_params = SamplingParams(
        max_tokens=512,
        temperature=0.0,
        top_p=1.0,
        skip_special_tokens=True,
    )

    examples = list(val_dataset)
    prompt_texts = [ex["prompt"] for ex in examples]

    outputs = llm.generate(prompt_texts, sampling_params, lora_request=lora_request)

    refs, preds = [], []
    for ex, output in zip(examples, outputs):
        raw = output.outputs[0].text.strip()
        _, _, predicted_harm = parse_reasoning_output(raw, with_intent=with_intent)
        true_harm = ex.get("harm_label", "")
        refs.append(true_harm)
        preds.append(predicted_harm or "")

    result = compute_harm_f1(refs, preds)
    return result["f1"] if result else None


# ── W&B query ─────────────────────────────────────────────────────────────────

def collect_runs(
    teacher_slug: str,
    student_slug: str,
    conditions: list[str],
    traces_version: str,
) -> list[dict]:
    """
    Query W&B for finished runs matching the given conditions and version.
    Returns a list of dicts with keys: run_id, run_path, condition,
    val_traces_path, adapter_path, student_hf, lora_rank, current_f1.
    """
    project = f"distillation-sweep-{student_slug}--{teacher_slug}"
    api = wandb.Api()

    try:
        runs = api.runs(path=project, filters={"state": "finished"})
    except Exception as e:
        print(f"    [warn] Could not query '{project}': {e}")
        return []

    results = []
    for run in runs:
        run_condition = (
            run.summary.get("reasoning_traces_condition")
            or run.config.get("data", {}).get("reasoning_traces_condition")
        )
        if run_condition not in conditions:
            continue

        # Filter by traces version: prefer config key, fall back to run name suffix
        run_version = run.config.get("data", {}).get("traces_version")
        if run_version is not None:
            if run_version != traces_version:
                continue
        elif not run.name.endswith(f"--{traces_version}"):
            continue

        val_traces_path = run.config.get("data", {}).get("reasoning_traces_val_path")
        if not val_traces_path:
            print(f"    [skip] {run.name}: no reasoning_traces_val_path in config")
            continue

        # Resolve adapter path
        adapter_path_str = run.summary.get("adapter_path")
        if not adapter_path_str:
            model_save_dir = run.config.get("paths", {}).get("model_save_dir")
            if not model_save_dir:
                continue
            adapter_path_str = model_save_dir + "_adapter"

        adapter_path = Path(adapter_path_str)
        if not adapter_path.is_absolute():
            adapter_path = PROJECT_ROOT / adapter_path

        if not adapter_path.exists():
            print(f"    [skip] {run.name}: adapter not found at {adapter_path}")
            continue

        student_hf = run.config.get("model", {}).get("name", "")
        lora_rank = run.config.get("peft", {}).get("lora_rank", 16)
        current_f1 = run.summary.get("val_harm_f1")

        results.append({
            "run_id":          run.id,
            "run_path":        f"{project}/{run.id}",
            "run_name":        run.name,
            "condition":       run_condition,
            "val_traces_path": val_traces_path,
            "adapter_path":    adapter_path,
            "student_hf":      student_hf,
            "lora_rank":       lora_rank,
            "current_f1":      current_f1,
            "_api_run":        run,
        })

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dry-run", "-n", action="store_true",
        help="Print planned actions without running inference or updating W&B.",
    )
    parser.add_argument(
        "--condition", nargs="+", default=DEFAULT_CONDITIONS, metavar="COND",
        help=f"Conditions to re-evaluate (default: {DEFAULT_CONDITIONS}).",
    )
    parser.add_argument(
        "--traces-version", default=DEFAULT_TRACES_VERSION, metavar="VERSION",
        help=f"Traces version to filter runs by (default: {DEFAULT_TRACES_VERSION}).",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Distillation Sweep — Val Re-evaluation")
    print(f"Conditions: {args.condition}  |  Traces: {args.traces_version}"
          + ("  [DRY RUN]" if args.dry_run else ""))
    print("=" * 60)

    # Collect all runs and group by student HF model ID
    runs_by_student: dict[str, list[dict]] = defaultdict(list)

    for teacher_hf, teacher_slug in TEACHER_MODELS:
        for student_hf, student_slug in STUDENT_MODELS:
            print(f"\n  Querying {teacher_slug} × {student_slug}...")
            runs = collect_runs(teacher_slug, student_slug, args.condition, args.traces_version)
            print(f"    Found {len(runs)} qualifying run(s)")
            for r in runs:
                runs_by_student[student_hf].append(r)

    total = sum(len(v) for v in runs_by_student.values())
    print(f"\nTotal runs to re-evaluate: {total}")
    if total == 0:
        print("Nothing to do.")
        return

    if args.dry_run:
        print("\n[dry-run] Would re-evaluate:")
        for student_hf, runs in runs_by_student.items():
            print(f"\n  Student: {student_hf}")
            for r in runs:
                print(f"    {r['run_name']}  condition={r['condition']}"
                      f"  current_f1={r['current_f1']}  adapter={r['adapter_path'].name}")
        return

    # Process student by student to share vLLM instance
    from vllm import LLM
    from vllm.lora.request import LoRARequest

    updated, failed = 0, 0

    for student_hf, runs in runs_by_student.items():
        print(f"\n{'='*60}")
        print(f"Student: {student_hf}  ({len(runs)} runs)")
        print("=" * 60)

        max_lora_rank = max(r["lora_rank"] for r in runs)

        print(f"Loading vLLM with enable_lora=True (max_lora_rank={max_lora_rank})...")
        llm = LLM(
            model=student_hf,
            enable_lora=True,
            max_lora_rank=max_lora_rank,
            max_loras=1,
            limit_mm_per_prompt={"image": 0},
            gpu_memory_utilization=0.90,
            max_model_len=2048,
            dtype="bfloat16",
            enforce_eager=True,
        )

        for r in runs:
            print(f"\n  Run: {r['run_name']}  condition={r['condition']}")
            print(f"  Adapter: {r['adapter_path'].name}")
            print(f"  Current val_harm_f1: {r['current_f1']}")

            try:
                val_dataset = build_val_dataset(r["val_traces_path"], r["condition"])
                print(f"  Val examples: {len(val_dataset)}")
            except Exception as e:
                print(f"  [error] Could not load val dataset: {e}")
                failed += 1
                continue

            lora_request = LoRARequest("adapter", 1, str(r["adapter_path"]))

            try:
                new_f1 = run_vllm_eval(llm, lora_request, val_dataset, r["condition"])
            except Exception as e:
                print(f"  [error] vLLM eval failed: {e}")
                failed += 1
                continue

            if new_f1 is None:
                print("  [warn] Could not compute harm F1 — no valid predictions parsed")
                failed += 1
                continue

            print(f"  New val_harm_f1: {new_f1:.4f}  (was {r['current_f1']})")
            r["_api_run"].summary.update({"val_harm_f1": new_f1})
            print("  [ok] W&B summary updated")
            updated += 1

        del llm
        import torch, gc
        torch.cuda.empty_cache()
        gc.collect()

    print(f"\n{'='*60}")
    print(f"Done. Updated: {updated}  Failed/skipped: {failed}")
    print("=" * 60)


if __name__ == "__main__":
    main()
