#!/usr/bin/env python3
"""
Build the train-sample JSON for the data-scaling experiment.

Output: a single JSON file containing every WildGuardMix (`wildguardtrain`)
prompt with a non-null `prompt_harm_label`. Annotated-intents prompts are NOT
excluded — annotated-intents was originally used as a training/validation
source, not as a primary held-out test, so the 1.4k overlap is acceptable.
The held-out mean in the eval notebook already excludes annotated-intents,
and the 5 genuinely OOD eval datasets (ToxicChat, Aegis, XSTest,
OpenAI Moderation, WildGuardMix-test) remain valid.

Output schema per record matches generate_reasoning_traces.load_samples_for_split,
so generate_reasoning_traces.py can consume it via `dataset.samples_json`:
    annotation_id, wildguard_id, prompt, prompt_harm_label, intent, split

`split` is set to "train" (the trace generator partitions output by this field).

Usage:
    python scripts/hpc/scaling/prepare_scaling_samples.py
    python scripts/hpc/scaling/prepare_scaling_samples.py --seed 42 --output-dir data/reasoning_traces_scaling/samples
"""

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from intention_jailbreak.dataset import load_wildguardmix


def build_full_wg_samples(rng: random.Random) -> list[dict]:
    print(f"\n  Loading allenai/wildguardmix (wildguardtrain) ...")
    ds_dict = load_wildguardmix("wildguardtrain")
    split_name = "train" if "train" in ds_dict else list(ds_dict.keys())[0]
    ds = ds_dict[split_name]
    print(f"    Raw size: {len(ds)}")

    samples: list[dict] = []
    skipped_no_label = 0
    for idx, row in enumerate(ds):
        prompt = (row.get("prompt") or "").strip()
        if not prompt:
            continue
        harm = row.get("prompt_harm_label")
        if harm is None:
            skipped_no_label += 1
            continue
        samples.append({
            "annotation_id":     None,
            # WG mix has no stable per-prompt ID; the row index is stable within
            # a given dataset version and avoids collision with annotated-intents IDs.
            "wildguard_id":      f"wgmix-{idx}",
            "prompt":            prompt,
            "prompt_harm_label": harm,
            "intent":            "",
            "split":             "train",
        })

    rng.shuffle(samples)
    print(f"    Kept: {len(samples)}  no-label: {skipped_no_label}")
    print(f"    Label distribution: {Counter(s['prompt_harm_label'] for s in samples)}")
    return samples


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed (only affects record order in the JSON; default: 42).")
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "data" / "reasoning_traces_scaling" / "samples",
        help="Directory to write the sample JSON file.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    print("=" * 60)
    print("Building full_wg samples (entire wildguardtrain)")
    print("=" * 60)
    samples = build_full_wg_samples(rng)

    out_path = args.output_dir / "full_wg_train_samples.json"
    with open(out_path, "w") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {out_path}  ({len(samples)} samples)")


if __name__ == "__main__":
    main()
