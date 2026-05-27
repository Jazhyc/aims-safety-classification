#!/usr/bin/env python3
"""Annotate HDBSCAN intent clusters using gpt-oss via vLLM.

Reads the JSONL produced by `export_clusters_jsonl` (one row per unique intent,
including a `cluster` field), samples N intents per cluster, asks the LLM for a
short LABEL + DESCRIPTION, and writes one row per cluster to an output JSONL.

Example:
    python scripts/analysis/annotate_clusters.py \\
        --input  data/cluster_annotations/le_dpo_intent_clusters.jsonl \\
        --output data/cluster_annotations/le_dpo_cluster_labels.jsonl
"""

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# Pick up HF_TOKEN from .env so the gpt-oss weights download with authenticated
# rate limits (anonymous downloads of the ~65 GB MoE checkpoint can take hours
# under HF's throttling).
load_dotenv()


SYSTEM_PROMPT = """You are an expert at categorizing user intents from a safety-classifier dataset.

You will be given a numbered list of user intent strings that an unsupervised clustering algorithm grouped together. Identify the common theme and respond in exactly this format:

LABEL: <short category name, 3-15 words>
DESCRIPTION: <1-2 sentence description of what defines this cluster>

The intents may be benign or harmful; describe the category neutrally without making a value judgment."""


USER_PROMPT_TEMPLATE = """Here are {n} example intents from one cluster:

{examples}

Respond using the LABEL/DESCRIPTION format."""


HARMONY_FINAL_RE = re.compile(
    r"<\|channel\|>final<\|message\|>(.*?)(?:<\|return\|>|<\|end\|>|$)", re.DOTALL
)
# `\*{0,2}` tolerates the model wrapping field names in markdown bold
# (`**LABEL:**`), which gpt-oss does about 2% of the time.
LABEL_RE = re.compile(
    r"^\s*\*{0,2}LABEL\*{0,2}\s*:\s*\*{0,2}\s*(.+?)\s*\*{0,2}\s*$",
    re.IGNORECASE | re.MULTILINE,
)
DESC_RE = re.compile(
    r"^\s*\*{0,2}DESCRIPTION\*{0,2}\s*:\s*\*{0,2}\s*(.+?)\s*\*{0,2}\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def load_clusters(input_path: Path) -> dict[int, list[dict]]:
    """Group input JSONL rows by cluster id, dropping noise (cluster == -1)."""
    rows_by_cluster: dict[int, list[dict]] = defaultdict(list)
    with input_path.open() as f:
        for line in f:
            row = json.loads(line)
            cid = row["cluster"]
            if cid == -1:
                continue
            rows_by_cluster[cid].append(row)
    return dict(rows_by_cluster)


def sample_examples(
    rows: list[dict],
    n: int,
    strategy: str,
    rng: random.Random,
) -> list[dict]:
    if strategy == "top_n":
        return sorted(rows, key=lambda r: -r.get("n_prompts", 1))[:n]
    if strategy == "random":
        return rng.sample(rows, min(len(rows), n))
    raise ValueError(f"unknown strategy {strategy!r}")


def build_user_prompt(examples: list[dict]) -> str:
    lines = []
    for i, r in enumerate(examples, 1):
        n_prompts = r.get("n_prompts", 1)
        # Showing n_prompts gives the LLM a sense of which examples are the
        # most representative of the cluster.
        lines.append(f"{i}. [{n_prompts} prompts] {r['intent']}")
    return USER_PROMPT_TEMPLATE.format(n=len(examples), examples="\n".join(lines))


def parse_response(raw_text: str) -> dict:
    """Pull LABEL/DESCRIPTION out of the harmony final channel."""
    m = HARMONY_FINAL_RE.search(raw_text)
    final_text = m.group(1).strip() if m else raw_text.strip()

    label_m = LABEL_RE.search(final_text)
    desc_m = DESC_RE.search(final_text)
    return {
        "label": label_m.group(1).strip() if label_m else None,
        "description": desc_m.group(1).strip() if desc_m else None,
        "final_channel": final_text,
        "raw_response": raw_text,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", default="openai/gpt-oss-120b")
    parser.add_argument("--n-examples", type=int, default=10)
    parser.add_argument(
        "--sampling-strategy", choices=["random", "top_n"], default="random",
        help="random samples uniformly with --seed (default); top_n picks the "
             "intents with the largest n_prompts (the deduped-prompt counts).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument(
        "--max-tokens", type=int, default=1024,
        help="gpt-oss can spend several hundred tokens in its analysis channel "
             "before emitting the final answer; 1024 leaves comfortable headroom.",
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    clusters = load_clusters(args.input)
    cluster_ids = sorted(clusters.keys())
    print(f"Annotating {len(cluster_ids)} clusters from {args.input}")

    # Build per-cluster prompts; the same system prompt for all.
    sampled = {cid: sample_examples(clusters[cid], args.n_examples, args.sampling_strategy, rng)
               for cid in cluster_ids}

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    formatted_prompts = [
        tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(sampled[cid])},
            ],
            add_generation_prompt=True,
            tokenize=False,
        )
        for cid in cluster_ids
    ]

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype="bfloat16",
        enforce_eager=True,
    )
    # skip_special_tokens=False so harmony <|channel|>...<|message|> markers
    # survive in the output text for parse_response to read.
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        skip_special_tokens=False,
    )

    outputs = llm.generate(formatted_prompts, sampling_params)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    n_parsed = 0
    with args.output.open("w", encoding="utf-8") as f:
        for cid, out in zip(cluster_ids, outputs):
            parsed = parse_response(out.outputs[0].text)
            if parsed["label"] is not None:
                n_parsed += 1
            record = {
                "cluster": cid,
                "n_intents": len(clusters[cid]),
                "n_examples_shown": len(sampled[cid]),
                "examples": [r["intent"] for r in sampled[cid]],
                **parsed,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {len(cluster_ids)} cluster annotations → {args.output}")
    print(f"  parsed LABEL successfully: {n_parsed} / {len(cluster_ids)}")


if __name__ == "__main__":
    main()
