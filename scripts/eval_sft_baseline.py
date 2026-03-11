"""
Evaluate the SFT baseline (Llama 3.1-8B + LoRA) on the held-out test set.

Saves predictions in the same format as train_dpo.py and train_contrastive.py
so compare_models.py can load all three side-by-side.

Output: data/predictions/sft_baseline/test_predictions.jsonl

Usage (from project root):
    python scripts/eval_sft_baseline.py \\
        --adapter-path trained_models/causal/hyperparam_sweep/lr_0.0005_e_5_adapter \\
        --base-model   meta-llama/Llama-3.1-8B-Instruct \\
        --output-dir   data/predictions/sft_baseline
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from intention_jailbreak.model_generation.preprocessing import preprocess_data
from intention_jailbreak.model_generation.data_utils import (
    apply_binary_harm_mapping,
    train_val_test_split,
)
from intention_jailbreak.model_generation.causal import extract_intent_and_harm
from sklearn.metrics import classification_report, f1_score, accuracy_score


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--adapter-path", type=str,
                   default="trained_models/causal/hyperparam_sweep/lr_0.0005_e_5_adapter")
    p.add_argument("--base-model",   type=str,
                   default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--output-dir",   type=str,
                   default="data/predictions/sft_baseline")
    p.add_argument("--test-size",    type=float, default=0.1)
    p.add_argument("--val-size",     type=float, default=0.1)
    p.add_argument("--seed",         type=int,   default=22)
    p.add_argument("--max-new-tokens", type=int, default=64)
    return p.parse_args()


def main():
    args = parse_args()

    # ── Dataset ───────────────────────────────────────────────────────────
    print("=== Loading test dataset ===")
    raw = preprocess_data()
    raw = apply_binary_harm_mapping(raw, binary_harm_mapping=True)
    split_cfg = {"data": {"test_size": args.test_size, "val_size": args.val_size,
                           "seed": args.seed}}
    _, _, test_dataset = train_val_test_split(raw, split_cfg)
    print(f"Test set: {len(test_dataset)} examples")

    # ── vLLM ─────────────────────────────────────────────────────────────
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    print(f"\n=== Loading SFT model with vLLM ===")
    print(f"  Base : {args.base_model}")
    print(f"  Adapter: {args.adapter_path}")

    llm = LLM(
        model=args.base_model,
        tokenizer=args.adapter_path,
        enable_lora=True,
        max_lora_rank=64,
        max_loras=1,
        limit_mm_per_prompt={"image": 0},
        gpu_memory_utilization=0.90,
        max_model_len=2048,
        dtype="bfloat16",
        enforce_eager=True,
    )
    lora_request = LoRARequest("sft_lora", 1, args.adapter_path)

    sampling_params = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=0.0,
        top_p=1.0,
        skip_special_tokens=True,
    )

    # ── Generate ──────────────────────────────────────────────────────────
    print("\n=== Generating predictions ===")
    examples = list(test_dataset)
    prompts  = [f"{ex['prompt']}\n" for ex in examples]
    outputs  = llm.generate(prompts, sampling_params, lora_request=lora_request)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_path = output_dir / "test_predictions.jsonl"

    all_preds, all_refs = [], []
    with open(pred_path, "w", encoding="utf-8") as f:
        for ex, output in zip(examples, outputs):
            raw_text  = output.outputs[0].text.strip()
            intent, pred_harm = extract_intent_and_harm(raw_text)
            true_harm = ex.get("Annotator Harm")
            all_preds.append(pred_harm or "")
            all_refs.append(true_harm or "")
            f.write(json.dumps({
                "id":               ex.get("id"),
                "prompt":           ex["prompt"],
                "true_harm":        true_harm,
                "predicted_harm":   pred_harm,
                "generated_intent": intent,
                "gold_intent":      ex.get("intent"),
                "raw_generation":   raw_text,
            }, ensure_ascii=False) + "\n")

    print(f"Saved {len(examples)} predictions → {pred_path}")

    # ── Metrics ───────────────────────────────────────────────────────────
    valid = [(t, p) for t, p in zip(all_refs, all_preds) if t and p]
    y_true = [t for t, _ in valid]
    y_pred = [p for _, p in valid]

    print("\n=== SFT Baseline — Test Results ===")
    print(f"n: {len(y_true)}")
    print(f"Accuracy : {accuracy_score(y_true, y_pred):.4f}")
    print(f"F1 macro : {f1_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, zero_division=0))

    del llm


if __name__ == "__main__":
    main()
