"""
Balance DPO pairs to 50/50 chosen harm distribution.

The raw dpo_pairs.jsonl is skewed (58.4% safe / 41.6% harmful) because the SFT
model has higher harmful recall — it rarely mislabels harmful prompts, so those
prompts produce fewer rejected samples and get filtered out during pair generation.

This script undersamples the majority class (safe) to match the minority class
(harmful), producing a balanced file suitable for DPO training.

Usage:
    python scripts/balance_dpo_pairs.py \
        --input  data/dpo_pairs/train_t0.8/dpo_pairs.jsonl \
        --output data/dpo_pairs/train_t0.8_balanced/dpo_pairs.jsonl \
        --seed   42
"""

import argparse
import json
import random
from pathlib import Path


def main(args):
    pairs = [json.loads(l) for l in open(args.input, encoding="utf-8") if l.strip()]

    safe    = [p for p in pairs if p["gold_harm"] == "safe"]
    harmful = [p for p in pairs if p["gold_harm"] == "harmful"]

    print(f"Input:   {len(pairs)} pairs total")
    print(f"  safe:    {len(safe)} ({100*len(safe)/len(pairs):.1f}%)")
    print(f"  harmful: {len(harmful)} ({100*len(harmful)/len(pairs):.1f}%)")

    minority = min(len(safe), len(harmful))
    rng = random.Random(args.seed)

    if len(safe) > len(harmful):
        safe = rng.sample(safe, minority)
        print(f"\nUndersampled safe → {len(safe)} (seed={args.seed})")
    elif len(harmful) > len(safe):
        harmful = rng.sample(harmful, minority)
        print(f"\nUndersampled harmful → {len(harmful)} (seed={args.seed})")
    else:
        print("\nAlready balanced, no undersampling needed.")

    balanced = safe + harmful
    rng.shuffle(balanced)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for p in balanced:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"\nOutput:  {len(balanced)} pairs → {out_path}")
    print(f"  safe:    {sum(1 for p in balanced if p['gold_harm'] == 'safe')} (50.0%)")
    print(f"  harmful: {sum(1 for p in balanced if p['gold_harm'] == 'harmful')} (50.0%)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--input",  default="data/dpo_pairs/train_t0.8/dpo_pairs.jsonl")
    p.add_argument("--output", default="data/dpo_pairs/train_t0.8_balanced/dpo_pairs.jsonl")
    p.add_argument("--seed",   type=int, default=42)
    main(p.parse_args())
