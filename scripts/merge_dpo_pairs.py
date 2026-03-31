"""
Merge DPO pairs from two sources:
  - annotated  : pairs generated from Jazhyc/wildguard-annotated-intents train split
  - wildguard  : pairs generated from high-confidence WildGuardMix examples
                 (via filter_wildguard_easy.py + generate_dpo_pairs.py --from-jsonl)

Also extracts SFT training examples from the WildGuardMix DPO pairs.
The "chosen" intent in WildGuardMix pairs is a synthetic T=0 model output
(no human annotation), but it is correct on the harm label, making it a
reasonable training signal for SFT augmentation.

Usage:
    python scripts/merge_dpo_pairs.py
    python scripts/merge_dpo_pairs.py \\
        --annotated-dir  data/dpo_pairs/train_t0.8 \\
        --wildguard-dir  data/wildguard_easy/dpo_pairs \\
        --output-dir     data/dpo_pairs/augmented \\
        --sft-output     data/wildguard_easy/sft_examples.jsonl
"""

import argparse
import json
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def merge_and_report(path_a: Path, path_b: Path, output: Path, label: str) -> tuple[int, int]:
    records_a = load_jsonl(path_a)
    records_b = load_jsonl(path_b)
    merged = records_a + records_b
    write_jsonl(merged, output)
    print(
        f"{label}: {len(records_a)} annotated + {len(records_b)} wildguard"
        f" = {len(merged)} total  →  {output}"
    )
    return len(records_a), len(records_b)


def main(args):
    annotated_dir = Path(args.annotated_dir)
    wildguard_dir = Path(args.wildguard_dir)
    output_dir    = Path(args.output_dir)

    # ------------------------------------------------------------------
    # Merge dpo_pairs.jsonl
    # ------------------------------------------------------------------
    merge_and_report(
        annotated_dir / "dpo_pairs.jsonl",
        wildguard_dir / "dpo_pairs.jsonl",
        output_dir   / "dpo_pairs.jsonl",
        "dpo_pairs.jsonl",
    )

    # ------------------------------------------------------------------
    # Extract SFT training examples from WildGuardMix DPO pairs.
    # The gold_intent field holds the synthetic T=0 chosen intent.
    # ------------------------------------------------------------------
    wg_pairs = load_jsonl(wildguard_dir / "dpo_pairs.jsonl")
    sft_examples = [
        {
            "prompt":    p["prompt"],
            "intent":    p["gold_intent"],
            "gold_harm": p["gold_harm"],
        }
        for p in wg_pairs
        if p.get("gold_intent")
    ]
    sft_out = Path(args.sft_output)
    write_jsonl(sft_examples, sft_out)
    print(f"sft_examples.jsonl: {len(sft_examples)} examples  →  {sft_out}")

    # ------------------------------------------------------------------
    # Summary: harm label distribution in merged DPO pairs
    # ------------------------------------------------------------------
    merged_pairs = load_jsonl(output_dir / "dpo_pairs.jsonl")
    n_safe    = sum(1 for p in merged_pairs if p["gold_harm"] == "safe")
    n_harmful = sum(1 for p in merged_pairs if p["gold_harm"] == "harmful")
    print(
        f"\nMerged DPO distribution:"
        f"  safe={n_safe} ({100*n_safe/len(merged_pairs):.1f}%)"
        f"  harmful={n_harmful} ({100*n_harmful/len(merged_pairs):.1f}%)"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument(
        "--annotated-dir",
        default="data/dpo_pairs/train_t0.8",
        help="Directory containing dpo_pairs.jsonl from the annotated (Jazhyc) train split.",
    )
    p.add_argument(
        "--wildguard-dir",
        default="data/wildguard_easy/dpo_pairs",
        help="Directory containing dpo_pairs.jsonl generated from WildGuardMix easy examples.",
    )
    p.add_argument(
        "--output-dir",
        default="data/dpo_pairs/augmented",
        help="Output directory for merged pairs files.",
    )
    p.add_argument(
        "--sft-output",
        default="data/wildguard_easy/sft_examples.jsonl",
        help="Output path for SFT training examples extracted from WildGuardMix DPO pairs.",
    )
    main(p.parse_args())
