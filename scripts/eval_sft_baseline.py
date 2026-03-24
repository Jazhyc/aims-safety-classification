"""
Evaluate a LoRA adapter on the held-out test set + WildGuardTest + XSTest benchmarks.

Saves predictions so compare_models.py can load all models side-by-side.

Outputs:
  <output-dir>/test_predictions.jsonl            <- annotated intents test split
  <output-dir>/wildguardtest_predictions.jsonl   <- WildGuardTest benchmark
  <output-dir>/xstest_predictions.jsonl          <- XSTest benchmark

Usage (from project root):
    python scripts/eval_sft_baseline.py \\
        --adapter-path trained_models/causal/hyperparam_sweep/lr_5e-05_e_5_adapter \\
        --base-model   meta-llama/Llama-3.1-8B-Instruct \\
        --output-dir   data/predictions/sft_baseline
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from datasets import load_dataset
from intention_jailbreak.model_generation.preprocessing import preprocess_data
from intention_jailbreak.model_generation.data_utils import apply_binary_harm_mapping
from sklearn.metrics import classification_report, f1_score, accuracy_score

import re
from typing import Optional


def _norm_harm(s: str) -> Optional[str]:
    """Normalise any harm label string to 'harmful' or 'safe'."""
    s = s.strip().lower().rstrip(".").rstrip(";")
    if s in {"harmful", "unsafe", "dangerous"}:
        return "harmful"
    if s in {"safe", "harmless", "benign", "unharmful"}:
        return "safe"
    return None


def extract_intent_and_harm(raw_text: str):
    text = raw_text.strip()
    if "<think>" in text and "</think>" in text:
        text = text.split("</think>")[-1].strip()
    m = re.search(r"Intent:\s*(.+?);\s*Harm:\s*(\S+)", text, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip(), _norm_harm(m.group(2))
    m = re.search(r"Intent:\s*(.+?)\n\s*Harm:\s*(\S+)", text, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip(), _norm_harm(m.group(2))
    m = re.search(r"Intent:\s*(.+?)\s+Harm:\s*(\S+)", text, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip(), _norm_harm(m.group(2))
    m = re.search(r"^(.+?);\s*Harm:\s*(\S+)", text, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip(), _norm_harm(m.group(2))
    m = re.search(r"[;\n]\s*Harm:\s*(\S+)", text, re.IGNORECASE)
    if m:
        harm = _norm_harm(m.group(1))
        intent = re.sub(r"^Intent:\s*", "", text[:m.start()].strip(), flags=re.IGNORECASE).strip()
        return intent or text, harm
    m = re.search(r"^Intent:\s*(.+?)(?:;|$)", text, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip(), None
    return text or None, None


def load_benchmark(name: str):
    """Load WildGuardTest or XSTest and return (examples, harm_column)."""
    if name == "wildguardtest":
        ds = load_dataset("allenai/wildguardmix", "wildguardtest", split="test")
        ds = ds.filter(lambda x: x["prompt_harm_label"] is not None)
        # map "unharmful" -> "safe"
        def _map(ex):
            ex["true_harm"] = _norm_harm(ex["prompt_harm_label"])
            return ex
        ds = ds.map(_map)
        return list(ds), "true_harm"
    elif name == "xstest":
        ds = load_dataset("walledai/XSTest", split="test")
        ds = ds.filter(lambda x: x["label"] is not None)
        # map "unsafe" -> "harmful", "safe" -> "safe"
        def _map(ex):
            ex["true_harm"] = _norm_harm(ex["label"])
            return ex
        ds = ds.map(_map)
        return list(ds), "true_harm"
    else:
        raise ValueError(f"Unknown benchmark: {name}")


def run_eval(llm, lora_request, sampling_params, examples, harm_col, pred_path: Path, dataset_name: str):
    """Run inference on examples, save predictions, print metrics."""
    prompts = [f"{ex['prompt']}\n" for ex in examples]
    outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)

    pred_path.parent.mkdir(parents=True, exist_ok=True)
    all_preds, all_refs = [], []
    with open(pred_path, "w", encoding="utf-8") as f:
        for ex, output in zip(examples, outputs):
            raw_text = output.outputs[0].text.strip()
            intent, pred_harm = extract_intent_and_harm(raw_text)
            true_harm = ex.get(harm_col)
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

    valid = [(t, p) for t, p in zip(all_refs, all_preds) if t and p]
    y_true = [t for t, _ in valid]
    y_pred = [p for _, p in valid]
    print(f"\n=== {dataset_name} Results ===")
    print(f"n: {len(y_true)}  unparsed: {len(examples) - len(valid)}")
    print(f"Accuracy : {accuracy_score(y_true, y_pred):.4f}")
    print(f"F1 macro : {f1_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    print(f"F1 harmful: {f1_score(y_true, y_pred, pos_label='harmful', average='binary', zero_division=0):.4f}")
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, zero_division=0))


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--adapter-path", type=str,
                   default="trained_models/causal/hyperparam_sweep/lr_5e-05_e_5_adapter")
    p.add_argument("--base-model",   type=str,
                   default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--output-dir",   type=str,
                   default="data/predictions/sft_baseline")
    p.add_argument("--max-new-tokens", type=int, default=64)
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)

    # ── Dataset ───────────────────────────────────────────────────────────
    print("=== Loading test dataset ===")
    test_dataset = preprocess_data(split="test")
    test_dataset = apply_binary_harm_mapping(test_dataset, binary_harm_mapping=True)
    print(f"Test set: {len(test_dataset)} examples")

    # ── vLLM ─────────────────────────────────────────────────────────────
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    print(f"\n=== Loading model with vLLM ===")
    print(f"  Base   : {args.base_model}")
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

    # ── 1. Annotated intents (HF test split) ─────────────────────────────
    print("\n=== Loading annotated intents test split ===")
    test_dataset = preprocess_data(split="test")
    test_dataset = apply_binary_harm_mapping(test_dataset, binary_harm_mapping=True)
    run_eval(llm, lora_request, sampling_params,
             list(test_dataset), "Annotator Harm",
             output_dir / "test_predictions.jsonl",
             "Annotated Intents")

    # ── 2. WildGuardTest ─────────────────────────────────────────────────
    print("\n=== Loading WildGuardTest ===")
    wgt_examples, wgt_harm_col = load_benchmark("wildguardtest")
    print(f"WildGuardTest: {len(wgt_examples)} examples")
    run_eval(llm, lora_request, sampling_params,
             wgt_examples, wgt_harm_col,
             output_dir / "wildguardtest_predictions.jsonl",
             "WildGuardTest")

    # ── 3. XSTest ────────────────────────────────────────────────────────
    print("\n=== Loading XSTest ===")
    xst_examples, xst_harm_col = load_benchmark("xstest")
    print(f"XSTest: {len(xst_examples)} examples")
    run_eval(llm, lora_request, sampling_params,
             xst_examples, xst_harm_col,
             output_dir / "xstest_predictions.jsonl",
             "XSTest")

    del llm


if __name__ == "__main__":
    main()
