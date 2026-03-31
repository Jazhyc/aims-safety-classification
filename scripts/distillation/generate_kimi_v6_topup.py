"""
Kimi K2.5 v6 Trace Top-up

Builds v6 reasoning traces for moonshotai/kimi-k2.5 by:

  1. Loading the existing v5 traces (generated per unique wildguard_id).
  2. Identifying the 175 secondary annotations in the train split that were
     dropped during v5 generation (wildguard_ids with multiple annotators).
  3. Generating traces for those 175 annotations via the OpenRouter API.
  4. Adding annotation_id to all v5 entries (back-filled from the dataset).
  5. Merging v5 + new entries and saving to data/reasoning_traces_v6/.

Val and test splits are identical to v5 (no multi-annotation entries) but
are copied with annotation_id added.

Requirements:
    OPENROUTER_API_KEY in environment (or .env file).

Usage:
    cd /scratch/s4626451/intention-jailbreak
    python scripts/distillation/generate_kimi_v6_topup.py

    # Dry run: show what would be generated without making API calls
    python scripts/distillation/generate_kimi_v6_topup.py --dry-run
"""

import argparse
import importlib.util
import json
import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from datasets import load_dataset
from tqdm import tqdm

# Load helper functions from generate_reasoning_traces.py via importlib
# (scripts/ has no __init__.py so it can't be imported as a package).
_traces_spec = importlib.util.spec_from_file_location(
    "generate_reasoning_traces",
    Path(__file__).parent / "generate_reasoning_traces.py",
)
_traces_mod = importlib.util.module_from_spec(_traces_spec)
_traces_spec.loader.exec_module(_traces_mod)
_detect_thinking_from_text = _traces_mod._detect_thinking_from_text
parse_model_output = _traces_mod.parse_model_output


TEACHER_MODEL   = "moonshotai/kimi-k2.5"
TEACHER_SLUG    = "moonshotai-kimi-k2.5"
V5_BASE         = Path("data/reasoning_traces_v5") / TEACHER_SLUG
V6_BASE         = Path("data/reasoning_traces_v6") / TEACHER_SLUG
CONDITIONS      = ["without_intent", "with_intent"]
DATASET_NAME    = "Jazhyc/wildguard-annotated-intents"
HARM_COLUMN     = "Annotator Harm"


# ── Prompt building (mirrors generate_reasoning_traces.py) ─────────────────

def _build_teacher_prompt(user_prompt: str, condition: str, **labels) -> str:
    from intention_jailbreak.model_generation.prompt_templates import (
        PREAMBLE, OUTPUT_FORMAT_WITHOUT_INTENT, OUTPUT_FORMAT_WITH_INTENT,
        TEACHER_GROUND_TRUTH,
    )
    if condition == "without_intent":
        output_fmt = OUTPUT_FORMAT_WITHOUT_INTENT
        gt_block = TEACHER_GROUND_TRUTH["without_intent"].format(
            annotator_harmful_label=labels["annotator_harmful_label"]
        )
    else:
        output_fmt = OUTPUT_FORMAT_WITH_INTENT
        gt_block = TEACHER_GROUND_TRUTH["with_intent"].format(
            intent=labels.get("intent", ""),
            annotator_harmful_label=labels["annotator_harmful_label"],
        )
    return f"{PREAMBLE}\n\nHuman user:\n{user_prompt}\n\n{gt_block}\n{output_fmt}"


# ── OpenRouter API call ────────────────────────────────────────────────────

def call_openrouter(
    client,
    sample: dict,
    condition: str,
    max_tokens: int = 16384,
    max_retries: int = 3,
) -> tuple[str, dict]:
    """
    Make one OpenRouter call.

    Returns:
        (raw_content, parsed_fields) where parsed_fields has keys
        prompt_intent, prompt_harm, reasoning, thinking_trace.
    """
    user_content = _build_teacher_prompt(
        sample["prompt"], condition,
        annotator_harmful_label=sample["prompt_harm_label"],
        intent=sample.get("intent", ""),
    )
    messages = [{"role": "user", "content": user_content}]

    last_exc = None
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=TEACHER_MODEL,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.0,
            )
            msg = response.choices[0].message
            content = msg.content or ""
            reasoning = getattr(msg, "reasoning", None)
            if reasoning is None and hasattr(msg, "model_extra"):
                reasoning = (msg.model_extra or {}).get("reasoning")
            if reasoning is None:
                thinking_start, thinking_end = _detect_thinking_from_text(content)
            else:
                thinking_start, thinking_end = "", ""
            parsed = parse_model_output(content, thinking_start, thinking_end,
                                        pre_extracted_thinking=reasoning)
            return content, parsed
        except Exception as e:
            last_exc = e
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"  [retry {attempt + 1}/{max_retries}] "
                      f"ann={sample['annotation_id']}/{condition}: {e} — {wait}s")
                time.sleep(wait)

    raise RuntimeError(
        f"All {max_retries} attempts failed for ann={sample['annotation_id']}/{condition}"
    ) from last_exc


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", "-n", action="store_true",
                        help="Show what would be generated without making API calls.")
    parser.add_argument("--max-workers", type=int, default=16,
                        help="Concurrent OpenRouter requests (default 16).")
    args = parser.parse_args()

    # ── Load annotation dataset for all splits ─────────────────────────────
    print("Loading annotation dataset ...")
    ann_by_split = {}
    for split in ("train", "validation", "test"):
        ds = load_dataset(DATASET_NAME, split=split)
        ann_by_split[split] = list(ds)
        print(f"  {split}: {len(ds)} annotations")

    # Build annotation_id lookup: (split, wildguard_id) → list[annotation_id]
    # We need the FIRST annotation per wildguard_id (the one used in v5) and
    # all SECONDARY ones (missing from v5).
    first_ann_id: dict[tuple, int] = {}   # (split, wg_id) → ann_id of first occurrence
    secondary: dict[str, list] = defaultdict(list)  # split → list of secondary sample dicts

    for split, rows in ann_by_split.items():
        for row in rows:
            wg_id  = row["Wildguard ID"]
            ann_id = row["ID"]
            key = (split, wg_id)
            if key not in first_ann_id:
                first_ann_id[key] = ann_id
            else:
                # This is a secondary annotation — not covered by v5
                secondary[split].append({
                    "annotation_id":    ann_id,
                    "wildguard_id":     wg_id,
                    "prompt":           row["Prompt"].strip(),
                    "prompt_harm_label": row[HARM_COLUMN],
                    "intent":           row.get("Intent", ""),
                    "split":            split,
                })

    print(f"\nSecondary annotations (not in v5): "
          f"train={len(secondary['train'])}, "
          f"val={len(secondary.get('validation', []))}, "
          f"test={len(secondary.get('test', []))}")

    if args.dry_run:
        print(f"\n[DRY RUN] Would generate {len(secondary['train'])} × {len(CONDITIONS)} "
              f"= {len(secondary['train']) * len(CONDITIONS)} API calls for train.")
        print("[DRY RUN] Would back-fill annotation_id on all v5 entries.")
        print("[DRY RUN] Exiting without making any API calls.")
        return

    # ── Back-fill annotation_id on v5 entries and save to v6 ─────────────
    print("\n=== Processing v5 traces → v6 (adding annotation_id) ===")
    v5_parsed_by_split: dict[str, list] = {}
    v5_raw_by_split:    dict[str, list] = {}

    for split in ("train", "validation", "test"):
        parsed_path = V5_BASE / split / "parsed_results.json"
        raw_path    = V5_BASE / split / "raw_outputs.json"

        if not parsed_path.exists():
            print(f"  WARNING: {parsed_path} not found — skipping {split}")
            v5_parsed_by_split[split] = []
            v5_raw_by_split[split] = []
            continue

        with open(parsed_path) as f:
            parsed_entries = json.load(f)
        raw_entries = json.load(open(raw_path)) if raw_path.exists() else []

        # Back-fill annotation_id from first_ann_id mapping
        for entry in parsed_entries:
            wg_id = entry.get("wildguard_id")
            if entry.get("annotation_id") is None:
                entry["annotation_id"] = first_ann_id.get((split, wg_id))
        for entry in raw_entries:
            wg_id = entry.get("wildguard_id")
            if entry.get("annotation_id") is None:
                entry["annotation_id"] = first_ann_id.get((split, wg_id))

        v5_parsed_by_split[split] = parsed_entries
        v5_raw_by_split[split]    = raw_entries
        print(f"  {split}: loaded {len(parsed_entries)} v5 parsed entries")

    # ── Generate new traces for secondary train annotations ────────────────
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. "
            "Export it in your shell or add it to your .env file."
        )

    from openai import OpenAI
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    secondary_train = secondary["train"]
    tasks = [
        (sample, condition)
        for condition in CONDITIONS
        for sample in secondary_train
    ]
    total = len(tasks)
    print(f"\n=== Generating {total} new traces via OpenRouter ===")
    print(f"  ({len(secondary_train)} annotations × {len(CONDITIONS)} conditions)")

    new_parsed: list[dict] = []
    new_raw:    list[dict] = []

    def _call(task_idx, sample, condition):
        raw_content, parsed_fields = call_openrouter(client, sample, condition)
        return task_idx, sample, condition, raw_content, parsed_fields

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(_call, i, samp, cond): i
            for i, (samp, cond) in enumerate(tasks)
        }
        with tqdm(total=total, desc="OpenRouter requests", unit="req") as pbar:
            for future in as_completed(futures):
                try:
                    _, sample, condition, raw_content, parsed_fields = future.result()

                    new_parsed.append({
                        "annotation_id": sample["annotation_id"],
                        "wildguard_id":  sample["wildguard_id"],
                        "prompt":        sample["prompt"],
                        "split":         sample["split"],
                        "condition":     condition,
                        "ground_truth": {
                            "intent":            sample["intent"],
                            "prompt_harm_label": sample["prompt_harm_label"],
                        },
                        "predicted": {
                            "prompt_intent": parsed_fields.get("prompt_intent"),
                            "prompt_harm":   parsed_fields.get("prompt_harm"),
                        },
                        "reasoning": parsed_fields.get("reasoning", ""),
                    })
                    new_raw.append({
                        "annotation_id":    sample["annotation_id"],
                        "wildguard_id":     sample["wildguard_id"],
                        "prompt":           sample["prompt"],
                        "intent":           sample["intent"],
                        "prompt_harm_label": sample["prompt_harm_label"],
                        "split":            sample["split"],
                        "condition":        condition,
                        "raw_output":       raw_content,
                        "thinking_trace":   parsed_fields.get("thinking_trace", ""),
                    })
                except Exception as e:
                    print(f"  ERROR: {e}")
                pbar.update(1)

    print(f"Generated {len(new_parsed)} new trace entries.")

    # ── Merge and save ─────────────────────────────────────────────────────
    print("\n=== Saving v6 traces ===")
    all_parsed = {
        "train":      v5_parsed_by_split["train"] + new_parsed,
        "validation": v5_parsed_by_split["validation"],
        "test":       v5_parsed_by_split["test"],
    }
    all_raw = {
        "train":      v5_raw_by_split["train"] + new_raw,
        "validation": v5_raw_by_split["validation"],
        "test":       v5_raw_by_split["test"],
    }

    for split in ("train", "validation", "test"):
        out_dir = V6_BASE / split
        out_dir.mkdir(parents=True, exist_ok=True)

        with open(out_dir / "parsed_results.json", "w") as f:
            json.dump(all_parsed[split], f, indent=2, ensure_ascii=False)
        if all_raw[split]:
            with open(out_dir / "raw_outputs.json", "w") as f:
                json.dump(all_raw[split], f, indent=2, ensure_ascii=False)

        cond_counts = {}
        for cond in CONDITIONS:
            cond_counts[cond] = sum(
                1 for r in all_parsed[split] if r.get("condition") == cond
            )
        print(f"  [{split}] {len(all_parsed[split])} entries → {out_dir}")
        for cond, n in cond_counts.items():
            print(f"    {cond}: {n}")

    print("\nDone.")


if __name__ == "__main__":
    main()
