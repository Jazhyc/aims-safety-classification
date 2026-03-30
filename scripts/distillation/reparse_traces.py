"""
Re-parse raw reasoning trace outputs using the updated parse_response function.

Reads raw_outputs.json from a source version directory, applies proper thinking /
content separation (including the GPT-OSS-120B analysis/assistantfinal pattern that
was missed in v4), re-extracts intent and harm fields, then writes raw_outputs.json
and parsed_results.json to the destination version directory.

Usage:
    # Re-parse all models from v4 into v5
    python scripts/distillation/reparse_traces.py

    # Choose a different source / destination version
    python scripts/distillation/reparse_traces.py --src v3 --dst v4

    # Process only specific models (substring match against directory name)
    python scripts/distillation/reparse_traces.py --models gpt kimi
"""

import argparse
import json
from pathlib import Path

# Re-use parse_response and _extract_fields from the generation script
import sys
_repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_repo_root))
sys.path.insert(0, str(_repo_root / "src"))

from scripts.distillation.generate_reasoning_traces import (  # noqa: E402
    parse_response,
    _extract_fields,
)

SPLITS = ("train", "validation", "test")


def _reparse_entry(entry: dict) -> tuple[dict, dict]:
    """
    Re-parse a single raw_output entry into an updated raw entry and a parsed entry.

    Thinking extraction strategy (in order):
      1. If ``thinking_trace`` is non-empty in the stored entry: the upstream code
         already extracted thinking via vLLM reasoning_content or the OpenRouter
         ``reasoning`` field.  Use it directly as ``pre_extracted_thinking``.
      2. Otherwise: run ``parse_response`` in auto-detect mode so it can find
         delimiters (including the GPT-OSS-120B ``analysis``/``assistantfinal``
         pattern) inside ``raw_output``.
    """
    raw_output      = entry.get("raw_output", "")
    stored_thinking = entry.get("thinking_trace", "")

    if stored_thinking:
        parsed_resp = parse_response(raw_output, pre_extracted_thinking=stored_thinking)
    else:
        parsed_resp = parse_response(raw_output)  # auto-detect

    fields = _extract_fields(parsed_resp)

    new_raw = {
        "wildguard_id":      entry["wildguard_id"],
        "prompt":            entry["prompt"],
        "intent":            entry["intent"],
        "prompt_harm_label": entry["prompt_harm_label"],
        "split":             entry["split"],
        "condition":         entry["condition"],
        "raw_output":        parsed_resp["content"],   # content only (thinking stripped)
        "thinking_trace":    parsed_resp["thinking"],
    }

    new_parsed = {
        "wildguard_id": entry["wildguard_id"],
        "prompt":       entry["prompt"],
        "split":        entry["split"],
        "condition":    entry["condition"],
        "ground_truth": {
            "intent":           entry["intent"],
            "prompt_harm_label": entry["prompt_harm_label"],
        },
        "predicted": {
            "prompt_intent": fields["prompt_intent"],
            "prompt_harm":   fields["prompt_harm"],
        },
        "reasoning": fields["reasoning"],
    }

    return new_raw, new_parsed


def reparse_model(src_model_dir: Path, dst_model_dir: Path) -> None:
    for split in SPLITS:
        src_file = src_model_dir / split / "raw_outputs.json"
        if not src_file.exists():
            print(f"  [{split}] MISSING {src_file} — skipping")
            continue

        with open(src_file) as f:
            entries = json.load(f)

        raw_list    = []
        parsed_list = []
        fixed       = 0
        for entry in entries:
            old_thinking = entry.get("thinking_trace", "")
            new_raw, new_parsed = _reparse_entry(entry)
            raw_list.append(new_raw)
            parsed_list.append(new_parsed)
            if new_raw["thinking_trace"] != old_thinking:
                fixed += 1

        dst_split_dir = dst_model_dir / split
        dst_split_dir.mkdir(parents=True, exist_ok=True)

        with open(dst_split_dir / "raw_outputs.json", "w") as f:
            json.dump(raw_list, f, indent=2, ensure_ascii=False)
        with open(dst_split_dir / "parsed_results.json", "w") as f:
            json.dump(parsed_list, f, indent=2, ensure_ascii=False)

        harm_ok   = sum(1 for r in parsed_list if r["predicted"]["prompt_harm"] is not None)
        intent_ok = sum(1 for r in parsed_list if r["predicted"]["prompt_intent"] is not None)
        print(
            f"  [{split}] {len(parsed_list):4d} entries  "
            f"harm={harm_ok}/{len(parsed_list)}  intent={intent_ok}/{len(parsed_list)}  "
            f"thinking_fixed={fixed}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src",    default="v4",  help="Source version tag (default: v4)")
    parser.add_argument("--dst",    default="v5",  help="Destination version tag (default: v5)")
    parser.add_argument("--models", nargs="*",     help="Filter models by substring (default: all)")
    parser.add_argument(
        "--data-dir", default="data",
        help="Root data directory relative to repo root (default: data)",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    data_dir  = repo_root / args.data_dir

    src_root = data_dir / f"reasoning_traces_{args.src}"
    dst_root = data_dir / f"reasoning_traces_{args.dst}"

    if not src_root.exists():
        raise FileNotFoundError(f"Source directory not found: {src_root}")

    model_dirs = sorted(p for p in src_root.iterdir() if p.is_dir())
    if args.models:
        model_dirs = [p for p in model_dirs if any(f in p.name for f in args.models)]

    if not model_dirs:
        print("No model directories found.")
        return

    print(f"Source : {src_root}")
    print(f"Dest   : {dst_root}")
    print(f"Models : {[p.name for p in model_dirs]}\n")

    for model_dir in model_dirs:
        dst_model_dir = dst_root / model_dir.name
        print(f"=== {model_dir.name} ===")
        reparse_model(model_dir, dst_model_dir)
        print()

    print("Done.")


if __name__ == "__main__":
    main()
