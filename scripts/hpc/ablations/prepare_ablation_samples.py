#!/usr/bin/env python3
"""
Prepare train-sample JSON files for the label-source ablation.

Two output files, each shaped to be loaded by generate_reasoning_traces.py via
``dataset.samples_json`` (records mirror what load_all_samples() returns):

  hard_original_train_samples.json
      Annotated-intents `train` split, deduped to one row per Prompt (first row),
      with the per-row WG `Dataset Harm` label substituted in
      `prompt_harm_label` (instead of the human-relabeled "Annotator Harm").
      Mirrors the size of the human-relabeled hard subset after dedupe.

  random_original_train_samples.json
      Random sample from `allenai/wildguardmix` (wildguardtrain), excluding any
      prompt whose text appears in the annotated-intents train split. Stratified
      on (prompt_harm_label, adversarial). Size-matched to hard_original.

Output schema per record (matches generate_reasoning_traces.load_samples_for_split):
    annotation_id, wildguard_id, prompt, prompt_harm_label, intent, split

The `split` field is set to "train" so generate_reasoning_traces.py writes
results to the train/ subdirectory.

Usage:
    python scripts/hpc/ablations/prepare_ablation_samples.py
    python scripts/hpc/ablations/prepare_ablation_samples.py --seed 42 --output-dir data/reasoning_traces_v7_ablations/samples
"""

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

from datasets import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from intention_jailbreak.dataset import load_wildguardmix


ANNOTATED_DATASET = "Jazhyc/wildguard-annotated-intents"

# Lower-case map for the `Dataset Harm` field on the annotated dataset, which
# encodes the original WG label as "Harmful" / "Safe". WG itself uses
# "harmful" / "unharmful". We standardise to WG's own format so the teacher
# template substitution and downstream parsers see a familiar string.
DATASET_HARM_TO_WG = {
    "Harmful": "harmful",
    "Safe": "unharmful",
}


def build_hard_original(rng: random.Random) -> list[dict]:
    """One sample per unique prompt in annotated-intents `train`, with WG label."""
    ds = load_dataset(ANNOTATED_DATASET, split="train")

    seen_prompts: set[str] = set()
    samples: list[dict] = []
    skipped_no_label = 0
    for row in ds:
        prompt = (row.get("Prompt") or "").strip()
        if not prompt or prompt in seen_prompts:
            continue
        ds_harm = row.get("Dataset Harm")
        wg_label = DATASET_HARM_TO_WG.get(ds_harm)
        if wg_label is None:
            skipped_no_label += 1
            continue
        seen_prompts.add(prompt)
        samples.append({
            "annotation_id": row.get("ID"),
            "wildguard_id":  row.get("Wildguard ID"),
            "prompt":        prompt,
            # WG-format label fed to the teacher (replaces the 5-level human label).
            "prompt_harm_label": wg_label,
            "intent":        row.get("Intent", "") or "",
            "split":         "train",
        })

    rng.shuffle(samples)
    print(f"  hard_original: {len(samples)} samples (skipped no-label rows: {skipped_no_label})")
    print(f"    label distribution: {Counter(s['prompt_harm_label'] for s in samples)}")
    return samples


def build_random_original(
    rng: random.Random,
    target_size: int,
    excluded_prompts: set[str],
    excluded_wg_ids: set,
    seed: int,
) -> list[dict]:
    """Stratified random sample of WG prompts, excluding those in the hard subset."""
    print(f"\n  Loading allenai/wildguardmix (wildguardtrain) ...")
    ds_dict = load_wildguardmix("wildguardtrain")
    split_name = "train" if "train" in ds_dict else list(ds_dict.keys())[0]
    ds = ds_dict[split_name]
    print(f"    Raw size: {len(ds)}")

    # Bucket by (prompt_harm_label, adversarial) for stratified sampling.
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for idx, row in enumerate(ds):
        prompt = (row.get("prompt") or "").strip()
        if not prompt or prompt in excluded_prompts:
            continue
        harm = row.get("prompt_harm_label")
        adv = row.get("adversarial")
        if harm is None or adv is None:
            continue
        buckets[(harm, adv)].append({
            "wildguard_id":      idx,  # WG mix has no stable ID; use row index.
            "prompt":            prompt,
            "prompt_harm_label": harm,
            "adversarial":       bool(adv),
        })

    bucket_sizes = {k: len(v) for k, v in buckets.items()}
    total_pool = sum(bucket_sizes.values())
    print(f"    Pool after exclusion: {total_pool} prompts across "
          f"{len(buckets)} (harm, adversarial) buckets")
    for k, n in sorted(bucket_sizes.items()):
        print(f"      {k}: {n}")

    if total_pool < target_size:
        raise RuntimeError(
            f"Pool size {total_pool} < target {target_size}; cannot sample without replacement."
        )

    # Allocate target_size proportionally to bucket sizes (largest-remainder method).
    raw_alloc = {k: target_size * (n / total_pool) for k, n in bucket_sizes.items()}
    floors = {k: int(v) for k, v in raw_alloc.items()}
    remainder = target_size - sum(floors.values())
    leftovers = sorted(raw_alloc.items(), key=lambda kv: kv[1] - floors[kv[0]], reverse=True)
    for i in range(remainder):
        floors[leftovers[i][0]] += 1
    alloc = floors

    print(f"    Allocation:")
    for k in sorted(alloc):
        print(f"      {k}: take {alloc[k]} of {bucket_sizes[k]}")

    samples: list[dict] = []
    for key, n_take in alloc.items():
        bucket = buckets[key]
        chosen = rng.sample(bucket, n_take)
        for c in chosen:
            samples.append({
                "annotation_id":     None,
                "wildguard_id":      f"wgmix-{c['wildguard_id']}",
                "prompt":            c["prompt"],
                "prompt_harm_label": c["prompt_harm_label"],
                "intent":            "",
                "split":             "train",
            })

    rng.shuffle(samples)
    print(f"  random_original: {len(samples)} samples")
    print(f"    label distribution: {Counter(s['prompt_harm_label'] for s in samples)}")
    return samples


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for shuffling and random subset sampling (default: 42).")
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "data" / "reasoning_traces_v7_ablations" / "samples",
        help="Directory to write the two sample JSON files.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng_hard = random.Random(args.seed)
    rng_rand = random.Random(args.seed)

    print("=" * 60)
    print("Building hard_original (annotated-intents train, deduped, WG labels)")
    print("=" * 60)
    hard = build_hard_original(rng_hard)

    excluded_prompts = {s["prompt"] for s in hard}
    excluded_wg_ids = {s["wildguard_id"] for s in hard if s["wildguard_id"] is not None}

    print("\n" + "=" * 60)
    print(f"Building random_original (size-matched to hard_original n={len(hard)})")
    print("=" * 60)
    rand = build_random_original(
        rng_rand,
        target_size=len(hard),
        excluded_prompts=excluded_prompts,
        excluded_wg_ids=excluded_wg_ids,
        seed=args.seed,
    )

    hard_path = args.output_dir / "hard_original_train_samples.json"
    rand_path = args.output_dir / "random_original_train_samples.json"
    with open(hard_path, "w") as f:
        json.dump(hard, f, indent=2, ensure_ascii=False)
    with open(rand_path, "w") as f:
        json.dump(rand, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print(f"Wrote {hard_path}  ({len(hard)} samples)")
    print(f"Wrote {rand_path}  ({len(rand)} samples)")
    print("=" * 60)


if __name__ == "__main__":
    main()
