#!/usr/bin/env python3
"""
Build the merged train traces JSON for the broader-intent-mix ablation.

Concatenates two existing trace pools into a single ``parsed_results.json``
that the distillation training script consumes:

  1. All ``human_intent`` records from the annotated-intents train traces
     (``data/reasoning_traces_v7/<teacher>/train/parsed_results.json``).

  2. A harm-stratified random sample from the scaling pool's
     ``synthetic_intent`` traces
     (``data/reasoning_traces_scaling/<teacher>/full_wg/train/parsed_results.json``),
     size-matched to the human-intent half on binary harm (harmful / unharmful).
     Prompts that already appear in the annotated-intents train split are
     excluded to prevent within-corpus prompt leakage across the two halves.

Each record keeps its own ``condition`` field (``human_intent`` or
``synthetic_intent``), so the trace-loader in ``causal.py`` selects the right
intent source per row (``ground_truth.intent`` vs ``predicted.prompt_intent``).
The student system prompt is identical for both conditions, so no per-row
template branching is required at train time.

Usage:
    python scripts/hpc/ablations/prepare_broader_intent_mix_samples.py
    python scripts/hpc/ablations/prepare_broader_intent_mix_samples.py --seed 42
"""

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from intention_jailbreak.model_generation.data_utils import map_harm_to_binary


def _load_traces(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def build_mixed_traces(
    rng: random.Random,
    human_traces_path: Path,
    scaling_traces_path: Path,
) -> list[dict]:
    print(f"\n  Loading annotated human_intent traces from {human_traces_path} ...")
    all_human = _load_traces(human_traces_path)
    human = [r for r in all_human if r.get("condition") == "human_intent"]
    # Apply the same keepability filter the trainer uses (non-empty reasoning,
    # mappable harm) so the mix size matches what the trainer will actually
    # see end-to-end.
    human_keep = [
        r for r in human
        if (r.get("reasoning") or "").strip()
        and map_harm_to_binary(r.get("ground_truth", {}).get("prompt_harm_label")) is not None
    ]
    print(f"    Loaded: {len(human)}  keepable: {len(human_keep)}")

    binary_counts = Counter(
        map_harm_to_binary(r["ground_truth"]["prompt_harm_label"]) for r in human_keep
    )
    print(f"    Binary harm distribution: {dict(binary_counts)}")

    print(f"\n  Loading scaling synthetic_intent traces from {scaling_traces_path} ...")
    all_scaling = _load_traces(scaling_traces_path)
    scaling = [r for r in all_scaling if r.get("condition") == "synthetic_intent"]
    print(f"    Loaded: {len(scaling)}")

    excluded_prompts = {(r.get("prompt") or "").strip() for r in human_keep}
    print(f"    Excluding {len(excluded_prompts)} prompts that appear in the human-intent half")

    # Bucket the synthetic pool by binary harm so we can size-match exactly.
    buckets: dict[str, list[dict]] = defaultdict(list)
    skipped_overlap = 0
    skipped_bad = 0
    for rec in scaling:
        prompt = (rec.get("prompt") or "").strip()
        if not prompt:
            skipped_bad += 1
            continue
        if prompt in excluded_prompts:
            skipped_overlap += 1
            continue
        if not (rec.get("reasoning") or "").strip():
            skipped_bad += 1
            continue
        harm_binary = map_harm_to_binary(rec.get("ground_truth", {}).get("prompt_harm_label"))
        if harm_binary is None:
            skipped_bad += 1
            continue
        buckets[harm_binary].append(rec)
    print(f"    Pool after exclusion: {sum(len(v) for v in buckets.values())} "
          f"(overlap with human: {skipped_overlap}, unfiltered bad rows: {skipped_bad})")
    for h in sorted(buckets):
        print(f"      {h}: {len(buckets[h])}")

    sampled: list[dict] = []
    for harm, target in binary_counts.items():
        pool = buckets.get(harm, [])
        if len(pool) < target:
            raise RuntimeError(
                f"Synthetic pool has only {len(pool)} {harm!r} records, "
                f"target {target}. Aborting before producing a skewed mix."
            )
        sampled.extend(rng.sample(pool, target))
    print(f"\n    Synthetic sampled: {len(sampled)}")
    print(f"    Synthetic binary distribution: {Counter(map_harm_to_binary(r['ground_truth']['prompt_harm_label']) for r in sampled)}")

    merged = human_keep + sampled
    rng.shuffle(merged)
    print(f"\n  Merged total: {len(merged)}  (human_intent={len(human_keep)}, synthetic_intent={len(sampled)})")
    print(f"  Per-condition counts: {Counter(r['condition'] for r in merged)}")
    print(f"  Binary harm distribution: {Counter(map_harm_to_binary(r['ground_truth']['prompt_harm_label']) for r in merged)}")
    return merged


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for synthetic subsampling and final shuffle (default: 42).")
    parser.add_argument(
        "--teacher-slug", default="openai-gpt-oss-120b",
        help="Teacher slug (default: openai-gpt-oss-120b).",
    )
    parser.add_argument(
        "--human-traces-path", type=Path, default=None,
        help="Override path to annotated train parsed_results.json.",
    )
    parser.add_argument(
        "--scaling-traces-path", type=Path, default=None,
        help="Override path to scaling synthetic train parsed_results.json.",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=None,
        help="Directory to write the merged parsed_results.json (default derived from teacher slug).",
    )
    args = parser.parse_args()

    teacher_slug = args.teacher_slug
    human_traces_path = args.human_traces_path or (
        PROJECT_ROOT / "data" / "reasoning_traces_v7" / teacher_slug / "train" / "parsed_results.json"
    )
    scaling_traces_path = args.scaling_traces_path or (
        PROJECT_ROOT / "data" / "reasoning_traces_scaling" / teacher_slug / "full_wg" / "train" / "parsed_results.json"
    )
    out_dir = args.output_dir or (
        PROJECT_ROOT / "data" / "reasoning_traces_v7_ablations" / teacher_slug
        / "broader_intent_mix" / "train"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)

    print("=" * 60)
    print("Building broader_intent_mix train traces")
    print(f"  teacher={teacher_slug}  seed={args.seed}")
    print("=" * 60)

    merged = build_mixed_traces(
        rng=rng,
        human_traces_path=human_traces_path,
        scaling_traces_path=scaling_traces_path,
    )

    out_path = out_dir / "parsed_results.json"
    with open(out_path, "w") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print(f"Wrote {out_path}  ({len(merged)} records)")
    print("=" * 60)


if __name__ == "__main__":
    main()
