#!/usr/bin/env python3
"""
Build the merged train traces JSON for the broader-intent-mix ablation.

Concatenates two existing trace pools into a single ``parsed_results.json``
that the distillation training script consumes:

  1. All ``<variant>_intent`` records from the annotated-intents train traces
     at ``data/reasoning_traces_v7/<teacher>/train/parsed_results.json``,
     where ``<variant>`` is ``human`` or ``synthetic`` (selected via
     ``--variant``; default ``human``). Both variants share the same human-
     annotated 4-category harm label on the annotated half; only the intent
     source (``ground_truth.intent`` vs ``predicted.prompt_intent``) differs.

  2. A harm-stratified random sample from the scaling pool's
     ``synthetic_intent`` traces
     (``data/reasoning_traces_scaling/<teacher>/full_wg/train/parsed_results.json``),
     size-matched to the annotated half on binary harm (harmful / unharmful).
     Prompts that already appear in the annotated-intents train split are
     excluded to prevent within-corpus prompt leakage across the two halves.

Each record keeps its own ``condition`` field, so the trace-loader in
``causal.py`` selects the right intent source per row. The student system
prompt is identical across ``human_intent`` / ``synthetic_intent``, so no
per-row template branching is required at train time.

Usage:
    # Mix annotated human_intent with WG synthetic_intent (the default)
    python scripts/hpc/ablations/prepare_broader_intent_mix_samples.py

    # Mix annotated synthetic_intent with WG synthetic_intent
    python scripts/hpc/ablations/prepare_broader_intent_mix_samples.py --variant synthetic
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
    annotated_traces_path: Path,
    scaling_traces_path: Path,
    annotated_condition: str,
) -> list[dict]:
    print(f"\n  Loading annotated {annotated_condition} traces from {annotated_traces_path} ...")
    all_annotated = _load_traces(annotated_traces_path)
    annotated = [r for r in all_annotated if r.get("condition") == annotated_condition]
    # Apply the same keepability filter the trainer uses (non-empty reasoning,
    # mappable harm) so the mix size matches what the trainer will actually
    # see end-to-end.
    annotated_keep = [
        r for r in annotated
        if (r.get("reasoning") or "").strip()
        and map_harm_to_binary(r.get("ground_truth", {}).get("prompt_harm_label")) is not None
    ]
    print(f"    Loaded: {len(annotated)}  keepable: {len(annotated_keep)}")

    binary_counts = Counter(
        map_harm_to_binary(r["ground_truth"]["prompt_harm_label"]) for r in annotated_keep
    )
    print(f"    Binary harm distribution: {dict(binary_counts)}")

    print(f"\n  Loading scaling synthetic_intent traces from {scaling_traces_path} ...")
    all_scaling = _load_traces(scaling_traces_path)
    scaling = [r for r in all_scaling if r.get("condition") == "synthetic_intent"]
    print(f"    Loaded: {len(scaling)}")

    excluded_prompts = {(r.get("prompt") or "").strip() for r in annotated_keep}
    print(f"    Excluding {len(excluded_prompts)} prompts that appear in the annotated half")

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

    merged = annotated_keep + sampled
    rng.shuffle(merged)
    print(f"\n  Merged total: {len(merged)}  "
          f"(annotated {annotated_condition}={len(annotated_keep)}, scaling synthetic_intent={len(sampled)})")
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
        "--variant", choices=["human", "synthetic"], default="human",
        help="Which condition to draw for the annotated half (default: human, "
             "matching the original broader_intent_mix experiment).",
    )
    parser.add_argument(
        "--teacher-slug", default="openai-gpt-oss-120b",
        help="Teacher slug (default: openai-gpt-oss-120b).",
    )
    parser.add_argument(
        "--annotated-traces-path", type=Path, default=None,
        help="Override path to annotated train parsed_results.json.",
    )
    parser.add_argument(
        "--scaling-traces-path", type=Path, default=None,
        help="Override path to scaling synthetic train parsed_results.json.",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=None,
        help="Directory to write the merged parsed_results.json. "
             "Defaults to data/reasoning_traces_v7_ablations/<teacher>/broader_intent_mix[_synthetic]/train/.",
    )
    args = parser.parse_args()

    teacher_slug = args.teacher_slug
    annotated_condition = f"{args.variant}_intent"
    default_subdir = "broader_intent_mix" if args.variant == "human" else "broader_intent_mix_synthetic"

    annotated_traces_path = args.annotated_traces_path or (
        PROJECT_ROOT / "data" / "reasoning_traces_v7" / teacher_slug / "train" / "parsed_results.json"
    )
    scaling_traces_path = args.scaling_traces_path or (
        PROJECT_ROOT / "data" / "reasoning_traces_scaling" / teacher_slug / "full_wg" / "train" / "parsed_results.json"
    )
    out_dir = args.output_dir or (
        PROJECT_ROOT / "data" / "reasoning_traces_v7_ablations" / teacher_slug
        / default_subdir / "train"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)

    print("=" * 60)
    print(f"Building {default_subdir} train traces")
    print(f"  teacher={teacher_slug}  variant={args.variant}  seed={args.seed}")
    print("=" * 60)

    merged = build_mixed_traces(
        rng=rng,
        annotated_traces_path=annotated_traces_path,
        scaling_traces_path=scaling_traces_path,
        annotated_condition=annotated_condition,
    )

    out_path = out_dir / "parsed_results.json"
    with open(out_path, "w") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print(f"Wrote {out_path}  ({len(merged)} records)")
    print("=" * 60)


if __name__ == "__main__":
    main()
