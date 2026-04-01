"""
Filter WildGuardMix training set for high-confidence ("easy") examples.

Uses the trained ModernBERT ensemble classifier checkpoint to score each prompt
and keeps only examples where the model is highly confident — either harmful
(prob > harm_threshold) or safe (prob < safe_threshold). These examples provide
clean, unambiguous signal for augmenting the SFT and DPO training sets.

Examples from the human-annotated set (Jazhyc/wildguard-annotated-intents) are
removed to prevent train/eval leakage.

Output:
    data/wildguard_easy/candidates.jsonl
    Each record: {id, prompt, gold_harm, harmful_probability}

Usage:
    python scripts/dataset_analysis/filter_wildguard_easy.py
    python scripts/dataset_analysis/filter_wildguard_easy.py \\
        --harm-threshold 0.90 --safe-threshold 0.10 \\
        --output-dir data/wildguard_easy_strict
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from intention_jailbreak.dataset import wildguardmix
from intention_jailbreak.ensemble.deepensembleclassifier import DeepEnsembleClassifier
from intention_jailbreak.model_generation.preprocessing import preprocess_data
from intention_jailbreak.training import tokenize_dataset


MODEL_PATH = "models/modernbert-ensemble/final_model"
BATCH_SIZE = 256
MAX_LENGTH = 2048
NUM_WORKERS = 4


def load_annotated_prompts() -> set:
    """Load all prompts from the annotated train split for deduplication."""
    print("Loading annotated prompts for deduplication...")
    ds = preprocess_data(split="train")
    prompts = {ex["prompt"] for ex in ds}
    print(f"  Annotated train prompts: {len(prompts)}")
    return prompts


def load_model_and_tokenizer(model_path: str, device: torch.device):
    """Load ModernBERT ensemble classifier and tokenizer."""
    print(f"\nLoading model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = DeepEnsembleClassifier.from_pretrained(
        model_path,
        model_class=AutoModelForSequenceClassification,
        dtype=torch.bfloat16,
    )

    model = model.to(device)
    model.eval()
    print("  Model loaded.")
    return model, tokenizer


def score_prompts(prompts: list, model, tokenizer, device: torch.device) -> list:
    """Run batch inference and return harmful_probability for each prompt."""
    dataset = Dataset.from_dict({"prompt": prompts})
    dataset = tokenize_dataset(dataset, tokenizer, MAX_LENGTH, "prompt", num_proc=1)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        drop_last=False,
    )

    all_probs = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Scoring prompts"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            if hasattr(outputs, "probs"):
                probs = outputs.probs
            else:
                probs = torch.softmax(outputs.logits, dim=-1)
            harmful_probs = probs[:, 1].float().cpu().numpy()
            all_probs.extend(harmful_probs.tolist())

    return all_probs


def save_jsonl(records: list, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Saved {len(records):>6} records → {path}")


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Load WildGuardMix
    print("\n=== Loading WildGuardMix ===")
    df = wildguardmix.load(subset="wildguardtrain")
    print(f"  Total WildGuardMix rows: {len(df)}")

    # 2. Deduplicate against annotated set
    annotated_prompts = load_annotated_prompts()
    df = df[~df["prompt"].isin(annotated_prompts)].reset_index(drop=True)
    print(f"  After dedup: {len(df)} rows")

    # 3. Score with ModernBERT
    model, tokenizer = load_model_and_tokenizer(args.model_path, device)
    prompts = df["prompt"].tolist()
    probs = score_prompts(prompts, model, tokenizer, device)
    df["harmful_probability"] = probs

    # 4. Filter to high-confidence examples
    mask = (df["harmful_probability"] > args.harm_threshold) | \
           (df["harmful_probability"] < args.safe_threshold)
    easy_df = df[mask].reset_index(drop=True)
    print(f"\n=== Filtering results ===")
    print(f"  Harm threshold  : > {args.harm_threshold}")
    print(f"  Safe threshold  : < {args.safe_threshold}")
    print(f"  Easy examples   : {len(easy_df)} / {len(df)}")
    n_harmful = (easy_df["harmful_probability"] > 0.5).sum()
    n_safe = len(easy_df) - n_harmful
    print(f"  Harmful         : {n_harmful}")
    print(f"  Safe            : {n_safe}")

    # 5. Assign binary harm label and sequential ID
    easy_df["gold_harm"] = easy_df["harmful_probability"].apply(
        lambda p: "harmful" if p > 0.5 else "safe"
    )
    easy_df["id"] = [f"wg_{i}" for i in range(len(easy_df))]

    # 6. Save
    output_dir = Path(args.output_dir)
    records = easy_df[["id", "prompt", "gold_harm", "harmful_probability"]].to_dict(orient="records")
    save_jsonl(records, output_dir / "candidates.jsonl")

    print(f"\nDone. {len(records)} candidates saved to {output_dir}/candidates.jsonl")


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--model-path", type=str, default=MODEL_PATH,
                   help="Path to trained ModernBERT ensemble checkpoint.")
    p.add_argument("--harm-threshold", type=float, default=0.85,
                   help="Minimum harmful_probability to include as 'harmful'.")
    p.add_argument("--safe-threshold", type=float, default=0.15,
                   help="Maximum harmful_probability to include as 'safe'.")
    p.add_argument("--output-dir", type=str, default="data/wildguard_easy",
                   help="Directory to write candidates.jsonl.")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
