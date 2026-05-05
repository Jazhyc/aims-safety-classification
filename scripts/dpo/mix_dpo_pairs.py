"""
Mix hard and judge DPO pairs at a configurable ratio (additive mode).

All hard pairs are always kept. A fraction of the available judge pairs is
sampled on top, controlled by --ratio (0.0 = hard only, 1.0 = full union).
This lets us sweep the contribution of judge signal without discarding any
hard mislabel signal.

Typical sweep:
    for ratio in 0.0 0.25 0.50 0.75 1.00; do
        python scripts/dpo/mix_dpo_pairs.py \\
            --hard-pairs  data/dpo_pairs/hard/pairs_wrong_only.jsonl \\
            --judge-pairs data/dpo_pairs/judge/pairs_judge_bad_only.jsonl \\
            --ratio $ratio \\
            --output      data/dpo_pairs/ratio_${ratio}/dpo_pairs.jsonl
    done
"""

import argparse
import json
import random
from pathlib import Path


def main(args):
    with open(args.hard_pairs, encoding="utf-8") as f:
        hard = [json.loads(l) for l in f if l.strip()]
    with open(args.judge_pairs, encoding="utf-8") as f:
        judge = [json.loads(l) for l in f if l.strip()]

    n_judge_include = int(args.ratio * len(judge))

    print(f"Hard pairs:  {len(hard)} (all kept)")
    print(f"Judge pairs: {len(judge)} available, including {n_judge_include} "
          f"(ratio={args.ratio:.2f}, seed={args.seed})")

    rng = random.Random(args.seed)
    judge_sample = rng.sample(judge, n_judge_include) if n_judge_include > 0 else []

    mixed = hard + judge_sample
    rng.shuffle(mixed)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for p in mixed:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    safe    = sum(1 for p in mixed if p.get("gold_harm") == "safe")
    harmful = sum(1 for p in mixed if p.get("gold_harm") == "harmful")
    print(f"\nOutput: {len(mixed)} pairs → {out}")
    print(f"  safe:    {safe} ({100*safe/len(mixed):.1f}%)")
    print(f"  harmful: {harmful} ({100*harmful/len(mixed):.1f}%)")
    print("(Run balance_dpo_pairs.py on the output to equalise safe/harmful.)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--hard-pairs",  required=True,
                   help="Path to pairs_wrong_only.jsonl (hard mislabels).")
    p.add_argument("--judge-pairs", required=True,
                   help="Path to pairs_judge_bad_only.jsonl (judge bad_match rejecteds).")
    p.add_argument("--ratio", type=float, default=0.5,
                   help="Fraction of judge pairs to include on top of all hard pairs "
                        "(0.0 = hard only, 1.0 = full union).")
    p.add_argument("--output", required=True,
                   help="Output path for the mixed dpo_pairs.jsonl.")
    p.add_argument("--seed", type=int, default=22,
                   help="Random seed for reproducible judge-pair subsampling.")
    main(p.parse_args())
