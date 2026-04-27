"""
Evaluate a LoRA adapter on the held-out test set + external benchmarks.

Saves predictions so compare_models.py can load all models side-by-side.

Outputs:
  <output-dir>/test_predictions.jsonl               <- annotated intents test split
  <output-dir>/wildguardtest_predictions.jsonl      <- WildGuardTest
  <output-dir>/xstest_predictions.jsonl             <- XSTest
  <output-dir>/toxic_chat_predictions.jsonl         <- ToxicChat
  <output-dir>/aegis_predictions.jsonl              <- AEGIS 2
  <output-dir>/openai_moderation_predictions.jsonl  <- OpenAI Moderation

Usage (from project root):
    python scripts/eval_sft_baseline.py \\
        --adapter-path Jazhyc/llama-3.1-8b-sft-generation \\
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
from intention_jailbreak.model_generation.prompt_templates import GENERATION_SYSTEM_PROMPT
from intention_jailbreak.model_generation.safety_classifier import map_harm_to_binary
from sklearn.metrics import classification_report, f1_score, accuracy_score

import re
from typing import Optional

DEFAULT_SFT_ADAPTER = "Jazhyc/llama-3.1-8b-sft-generation"


def _resolve_local_adapter_path(adapter_ref: str, revision: Optional[str] = None) -> str:
    """Resolve adapter source (local path or HF repo) to a local directory."""
    p = Path(adapter_ref).expanduser()
    if p.exists():
        return str(p.resolve())

    from huggingface_hub import snapshot_download
    try:
        return snapshot_download(adapter_ref, revision=revision)
    except Exception as e:
        raise RuntimeError(
            f"Failed to resolve adapter '{adapter_ref}'"
            + (f" at revision '{revision}'" if revision else "")
            + f": {e}"
        ) from e


def _norm_harm(s: str) -> Optional[str]:
    """Normalise any harm label string to 'harmful' or 'safe'."""
    s = s.strip().lower().rstrip(".").rstrip(";").strip()
    # Exact matches first
    if s in {"harmful", "unsafe", "dangerous"}:
        return "harmful"
    if s in {"safe", "harmless", "benign", "unharmful"}:
        return "safe"
    # Multi-word phrases: "extremely harmful", "very unsafe", "completely safe", etc.
    # Also bare "harm" which is a truncation of "harmful" in some model outputs.
    if any(w in s for w in ("harmful", "unsafe", "dangerous")) or s == "harm":
        return "harmful"
    if any(w in s for w in ("safe", "harmless", "benign", "unharmful")):
        return "safe"
    return None


def _apply_chat_template(tokenizer, messages: list[dict]) -> str:
    """Apply chat template with optional enable_thinking=False for Qwen3 models."""
    try:
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )


def extract_intent_and_harm(raw_text: str):
    text = raw_text.strip()
    if "<think>" in text and "</think>" in text:
        text = text.split("</think>")[-1].strip()
    m = re.search(r"Intent:\s*(.+?);\s*Harm:\s*(.+?)$", text, re.IGNORECASE | re.MULTILINE)
    if m:
        return m.group(1).strip(), _norm_harm(m.group(2))
    m = re.search(r"Intent:\s*(.+?)\n\s*Harm:\s*(.+?)$", text, re.IGNORECASE | re.MULTILINE)
    if m:
        return m.group(1).strip(), _norm_harm(m.group(2))
    m = re.search(r"Intent:\s*(.+?)\s+Harm:\s*(.+?)$", text, re.IGNORECASE | re.MULTILINE)
    if m:
        return m.group(1).strip(), _norm_harm(m.group(2))
    m = re.search(r"^(.+?);\s*Harm:\s*(.+?)$", text, re.IGNORECASE | re.MULTILINE)
    if m:
        return m.group(1).strip(), _norm_harm(m.group(2))
    m = re.search(r"[;\n]\s*Harm:\s*(\S+)", text, re.IGNORECASE)
    if m:
        harm = _norm_harm(m.group(1))
        intent = re.sub(r"^Intent:\s*", "", text[:m.start()].strip(), flags=re.IGNORECASE).strip()
        return intent or text, harm
    # Fallback: bare harm word after a semicolon/whitespace (e.g. "; Harmful", "; Safe")
    m = re.search(r"[;\s](harmful|unsafe|dangerous|safe|harmless|benign)\s*$", text, re.IGNORECASE)
    if m:
        harm = _norm_harm(m.group(1))
        intent = text[:m.start()].strip()
        intent = re.sub(r"^Intent:\s*", "", intent, flags=re.IGNORECASE).strip("; ")
        return intent or text, harm
    # Fallback: bare harm word anywhere at end of string (e.g. "Intent: ...\nHarmful")
    m = re.search(r"\b(harmful|unsafe|dangerous|safe|harmless|benign)\s*$", text, re.IGNORECASE)
    if m:
        harm = _norm_harm(m.group(1))
        intent = text[:m.start()].strip()
        intent = re.sub(r"^Intent:\s*", "", intent, flags=re.IGNORECASE).strip("; ")
        return intent or text, harm
    m = re.search(r"^Intent:\s*(.+?)(?:;|$)", text, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip(), None
    return text or None, None


def load_benchmark(name: str):
    """Load an external benchmark and return (examples, harm_column)."""
    if name == "wildguardtest":
        ds = load_dataset("allenai/wildguardmix", "wildguardtest", split="test")
        ds = ds.filter(lambda x: x["prompt_harm_label"] is not None)
        def _map(ex):
            ex["true_harm"] = map_harm_to_binary(ex["prompt_harm_label"])
            return ex
        ds = ds.map(_map)
        return list(ds), "true_harm"
    elif name == "xstest":
        ds = load_dataset("walledai/XSTest", split="test")
        ds = ds.filter(lambda x: x["label"] is not None)
        def _map(ex):
            ex["true_harm"] = map_harm_to_binary(ex["label"])
            return ex
        ds = ds.map(_map)
        return list(ds), "true_harm"
    elif name == "toxic-chat":
        ds = load_dataset("lmsys/toxic-chat", "toxicchat0124", split="test")
        ds = ds.filter(lambda x: x["toxicity"] is not None)
        ds = ds.rename_column("user_input", "prompt")
        def _map(ex):
            ex["true_harm"] = "harmful" if int(ex["toxicity"]) == 1 else "safe"
            return ex
        ds = ds.map(_map)
        return list(ds), "true_harm"
    elif name == "aegis":
        ds = load_dataset("nvidia/Aegis-AI-Content-Safety-Dataset-2.0", split="test")
        ds = ds.filter(lambda x: x["prompt_label"] is not None)
        def _map(ex):
            ex["true_harm"] = map_harm_to_binary(ex["prompt_label"])
            return ex
        ds = ds.map(_map)
        ds = ds.filter(lambda x: x["true_harm"] is not None)
        return list(ds), "true_harm"
    elif name == "openai-moderation":
        ds = load_dataset("AllanK24/openai-moderation-binary", split="test")
        ds = ds.filter(lambda x: x["prompt_label"] is not None)
        def _map(ex):
            ex["true_harm"] = map_harm_to_binary(ex["prompt_label"])
            return ex
        ds = ds.map(_map)
        ds = ds.filter(lambda x: x["true_harm"] is not None)
        return list(ds), "true_harm"
    else:
        raise ValueError(f"Unknown benchmark: {name}")


def run_eval(llm, lora_request, sampling_params, tokenizer, examples, harm_col, pred_path: Path, dataset_name: str):
    """Run inference on examples, save predictions, print metrics."""
    prompts = [
        _apply_chat_template(
            tokenizer,
            [
                {"role": "system", "content": GENERATION_SYSTEM_PROMPT},
                {"role": "user", "content": ex["prompt"]},
            ],
        )
        for ex in examples
    ]
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
                   default=DEFAULT_SFT_ADAPTER)
    p.add_argument("--adapter-revision", type=str, default=None,
                   help="Optional HF revision (branch/tag/commit) for --adapter-path.")
    p.add_argument("--base-model",   type=str,
                   default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--output-dir",   type=str,
                   default="data/predictions/sft_baseline")
    p.add_argument("--max-new-tokens", type=int, default=64)
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    local_adapter = _resolve_local_adapter_path(args.adapter_path, args.adapter_revision)

    # ── Dataset ───────────────────────────────────────────────────────────
    print("=== Loading test dataset ===")
    test_dataset = preprocess_data(split="test")
    test_dataset = apply_binary_harm_mapping(test_dataset, binary_harm_mapping=True)
    print(f"Test set: {len(test_dataset)} examples")

    # ── vLLM ─────────────────────────────────────────────────────────────
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from transformers import AutoTokenizer

    print(f"\n=== Loading model with vLLM ===")
    print(f"  Base   : {args.base_model}")
    print(f"  Adapter: {args.adapter_path}")
    print(f"  Local  : {local_adapter}")

    llm = LLM(
        model=args.base_model,
        tokenizer=local_adapter,
        enable_lora=True,
        max_lora_rank=64,
        max_loras=1,
        limit_mm_per_prompt={"image": 0},
        gpu_memory_utilization=0.90,
        max_model_len=2048,
        dtype="bfloat16",
        enforce_eager=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(local_adapter)
    lora_request = LoRARequest("sft_lora", 1, local_adapter)
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
    run_eval(llm, lora_request, sampling_params, tokenizer,
             list(test_dataset), "Annotator Harm",
             output_dir / "test_predictions.jsonl",
             "Annotated Intents")

    # ── 2. WildGuardTest ─────────────────────────────────────────────────
    print("\n=== Loading WildGuardTest ===")
    wgt_examples, wgt_harm_col = load_benchmark("wildguardtest")
    print(f"WildGuardTest: {len(wgt_examples)} examples")
    run_eval(llm, lora_request, sampling_params, tokenizer,
             wgt_examples, wgt_harm_col,
             output_dir / "wildguardtest_predictions.jsonl",
             "WildGuardTest")

    # ── 3. XSTest ────────────────────────────────────────────────────────
    print("\n=== Loading XSTest ===")
    xst_examples, xst_harm_col = load_benchmark("xstest")
    print(f"XSTest: {len(xst_examples)} examples")
    run_eval(llm, lora_request, sampling_params, tokenizer,
             xst_examples, xst_harm_col,
             output_dir / "xstest_predictions.jsonl",
             "XSTest")

    # ── 4. ToxicChat ─────────────────────────────────────────────────────
    print("\n=== Loading ToxicChat ===")
    tc_examples, tc_harm_col = load_benchmark("toxic-chat")
    print(f"ToxicChat: {len(tc_examples)} examples")
    run_eval(llm, lora_request, sampling_params, tokenizer,
             tc_examples, tc_harm_col,
             output_dir / "toxic_chat_predictions.jsonl",
             "ToxicChat")

    # ── 5. AEGIS 2 ───────────────────────────────────────────────────────
    print("\n=== Loading AEGIS 2 ===")
    aegis_examples, aegis_harm_col = load_benchmark("aegis")
    print(f"AEGIS 2: {len(aegis_examples)} examples")
    run_eval(llm, lora_request, sampling_params, tokenizer,
             aegis_examples, aegis_harm_col,
             output_dir / "aegis_predictions.jsonl",
             "AEGIS 2")

    # ── 6. OpenAI Moderation ─────────────────────────────────────────────
    print("\n=== Loading OpenAI Moderation ===")
    oai_examples, oai_harm_col = load_benchmark("openai-moderation")
    print(f"OpenAI Moderation: {len(oai_examples)} examples")
    run_eval(llm, lora_request, sampling_params, tokenizer,
             oai_examples, oai_harm_col,
             output_dir / "openai_moderation_predictions.jsonl",
             "OpenAI Moderation")

    del llm


if __name__ == "__main__":
    main()
