"""
DPO fine-tuning on top of the SFT Llama 3.1-8B LoRA adapter.

Setup:
  - π_θ (policy)   : base model + SFT LoRA adapter (trainable)
  - π_ref (reference): base model + SFT LoRA adapter (frozen)
  Both start from the same SFT checkpoint; DPO nudges the policy toward
  chosen completions and away from rejected ones while the KL term (β)
  keeps it from drifting too far from the reference.

Data format (dpo_pairs.jsonl):
  {"prompt": "...", "chosen": "Intent: ...; Harm: ...",
   "rejected": "Intent: ...; Harm: ..."}

Usage (from project root):
    python scripts/train_dpo.py \\
        --pairs-path     data/dpo_pairs/train_t0.8/dpo_pairs.jsonl \\
        --adapter-path   trained_models/causal/hyperparam_sweep/lr_0.0005_e_5_adapter \\
        --base-model     meta-llama/Llama-3.1-8B-Instruct \\
        --output-dir     trained_models/causal/llama-dpo \\
        --beta           0.1
"""

import argparse
import gc
import json
import os
import sys
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import wandb
from datasets import Dataset
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DPOConfig, DPOTrainer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from intention_jailbreak.training import set_all_seeds, print_gpu_info


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dpo_dataset(pairs_path: str, prompt_suffix: str = "\n") -> Dataset:
    """
    Load dpo_pairs.jsonl and return a HuggingFace Dataset with columns:
      prompt, chosen, rejected
    The prompt is formatted with the same trailing newline used during SFT.
    """
    records = []
    with open(pairs_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            records.append({
                "prompt":   d["prompt"] + prompt_suffix,
                "chosen":   d["chosen"],
                "rejected": d["rejected"],
            })
    print(f"Loaded {len(records)} DPO pairs from {pairs_path}")
    if records:
        print("  Example prompt  :", records[0]["prompt"][:80])
        print("  Example chosen  :", records[0]["chosen"])
        print("  Example rejected:", records[0]["rejected"])
    return Dataset.from_list(records)


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def _bnb_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def load_policy_model(base_model: str, adapter_path: str, tokenizer):
    """Load base model in 4-bit and attach the SFT LoRA adapter (trainable)."""
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=_bnb_config(),
        device_map="auto",
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
    print(f"Policy model loaded. Trainable params: "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    return model


def load_ref_model(base_model: str, adapter_path: str):
    """Load base model in 4-bit and attach the SFT LoRA adapter (frozen)."""
    ref = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=_bnb_config(),
        device_map="auto",
        attn_implementation="sdpa",
    )
    ref = PeftModel.from_pretrained(ref, adapter_path, is_trainable=False)
    ref.eval()
    print("Reference model loaded (frozen).")
    return ref


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    set_all_seeds(args.seed)
    print_gpu_info()

    # ── Weights & Biases ──────────────────────────────────────────────────
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run,
        config={
            "base_model":    args.base_model,
            "adapter_path":  args.adapter_path,
            "pairs_path":    args.pairs_path,
            "beta":          args.beta,
            "loss_type":     args.loss_type,
            "label_smoothing": args.label_smoothing,
            "epochs":        args.epochs,
            "learning_rate": args.learning_rate,
            "batch_size":    args.batch_size,
            "gradient_accumulation": args.gradient_accumulation,
            "max_length":    args.max_length,
            "val_split":     args.val_split,
            "seed":          args.seed,
        },
    )

    # ── Data ──────────────────────────────────────────────────────────────
    dataset = load_dpo_dataset(args.pairs_path)

    split = dataset.train_test_split(test_size=args.val_split, seed=args.seed)
    train_dataset = split["train"]
    val_dataset   = split["test"]
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    # ── Tokenizer ─────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Models ────────────────────────────────────────────────────────────
    policy_model = load_policy_model(args.base_model, args.adapter_path, tokenizer)
    ref_model    = load_ref_model(args.base_model, args.adapter_path)

    # ── DPO config ────────────────────────────────────────────────────────
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dpo_config = DPOConfig(
        output_dir=str(output_dir / "checkpoints"),
        beta=args.beta,
        loss_type=args.loss_type,
        label_smoothing=args.label_smoothing,

        # Optimisation
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=0.01,
        optim="paged_adamw_8bit",

        # Precision
        bf16=True,
        fp16=False,

        # Sequence lengths
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,

        # Saving / evaluation
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",

        # Misc
        gradient_checkpointing=True,
        remove_unused_columns=False,
        report_to="wandb",
        seed=args.seed,
    )

    trainer = DPOTrainer(
        model=policy_model,
        ref_model=ref_model,
        args=dpo_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
    )

    print("\n=== Starting DPO training ===")
    trainer.train()

    eval_results = trainer.evaluate()
    print(f"Final eval loss: {eval_results['eval_loss']:.4f}")
    wandb.run.summary["final_eval_loss"] = eval_results["eval_loss"]

    # ── Save adapter ───────────────────────────────────────────────────────
    adapter_save_dir = str(output_dir) + "_adapter"
    os.makedirs(adapter_save_dir, exist_ok=True)
    trainer.model.save_pretrained(adapter_save_dir)
    tokenizer.save_pretrained(adapter_save_dir)
    print(f"DPO adapter saved to {adapter_save_dir}")

    # Save config summary
    summary = {
        "base_model":    args.base_model,
        "sft_adapter":   args.adapter_path,
        "dpo_adapter":   adapter_save_dir,
        "pairs_path":    args.pairs_path,
        "beta":          args.beta,
        "epochs":        args.epochs,
        "learning_rate": args.learning_rate,
        "n_train":       len(train_dataset),
        "n_val":         len(val_dataset),
        "eval_loss":     eval_results["eval_loss"],
    }
    with open(output_dir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    wandb.finish()

    # ── Clean up before vLLM eval ──────────────────────────────────────────
    del trainer, policy_model, ref_model
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()

    # ── vLLM evaluation on test set ────────────────────────────────────────
    if not args.skip_eval:
        _run_vllm_eval(args, adapter_save_dir, tokenizer)


# ---------------------------------------------------------------------------
# vLLM evaluation (mirrors save_preds_causal in causal.py)
# ---------------------------------------------------------------------------

def _run_vllm_eval(args, adapter_save_dir: str, tokenizer):
    """Evaluate the DPO adapter on the held-out test split using vLLM."""
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from intention_jailbreak.model_generation.preprocessing import preprocess_data
    from intention_jailbreak.model_generation.data_utils import (
        apply_binary_harm_mapping,
        train_val_test_split,
    )
    from intention_jailbreak.model_generation.evaluate_generations import compute_and_log_metrics

    # Inline parser (extract_intent_and_harm is in intent_diversity_analysis.py,
    # not in causal.py — avoid cross-script imports by defining it here)
    import re as _re
    def _norm(s):
        s = s.strip().lower().rstrip(".;")
        return "safe" if s in {"safe","harmless","benign"} else ("harmful" if s in {"harmful","unsafe","dangerous"} else None)
    def extract_intent_and_harm(raw):
        t = raw.strip()
        if "<think>" in t and "</think>" in t:
            t = t.split("</think>")[-1].strip()
        for pat in [
            r"Intent:\s*(.+?);\s*Harm:\s*(\S+)",
            r"Intent:\s*(.+?)\n\s*Harm:\s*(\S+)",
            r"Intent:\s*(.+?)\s+Harm:\s*(\S+)",
            r"^(.+?);\s*Harm:\s*(\S+)",
        ]:
            m = _re.search(pat, t, _re.IGNORECASE | _re.DOTALL)
            if m:
                return m.group(1).strip(), _norm(m.group(2))
        m = _re.search(r"[;\n]\s*Harm:\s*(\S+)", t, _re.IGNORECASE)
        if m:
            return _re.sub(r"^Intent:\s*", "", t[:m.start()].strip(), flags=_re.IGNORECASE).strip() or t, _norm(m.group(1))
        return t or None, None

    print("\n=== Loading test dataset for evaluation ===")
    raw = preprocess_data()
    raw = apply_binary_harm_mapping(raw, binary_harm_mapping=True)
    split_cfg = {"data": {"test_size": 0.1, "val_size": 0.1, "seed": 22}}
    _, _, test_dataset = train_val_test_split(raw, split_cfg)
    print(f"Test set: {len(test_dataset)} examples")

    print(f"\n=== Loading DPO model with vLLM ===")
    llm = LLM(
        model=args.base_model,
        tokenizer=adapter_save_dir,
        enable_lora=True,
        max_lora_rank=64,
        max_loras=1,
        limit_mm_per_prompt={"image": 0},
        gpu_memory_utilization=0.90,
        max_model_len=2048,
        dtype="bfloat16",
        enforce_eager=True,
    )
    lora_request = LoRARequest("dpo_lora", 1, adapter_save_dir)

    sampling_params = SamplingParams(
        max_tokens=64,
        temperature=0.0,
        top_p=1.0,
        skip_special_tokens=True,
    )

    examples = list(test_dataset)
    prompts  = [f"{ex['prompt']}\n" for ex in examples]
    outputs  = llm.generate(prompts, sampling_params, lora_request=lora_request)

    preds_dir = Path(args.output_dir) / "predictions"
    preds_dir.mkdir(parents=True, exist_ok=True)
    pred_path = preds_dir / "test_predictions.jsonl"

    all_preds, all_refs = [], []
    with open(pred_path, "w", encoding="utf-8") as f:
        for ex, output in zip(examples, outputs):
            raw_text = output.outputs[0].text.strip()
            intent, pred_harm = extract_intent_and_harm(raw_text)
            true_harm = ex.get("Annotator Harm")
            all_preds.append(pred_harm or "")
            all_refs.append(true_harm or "")
            f.write(json.dumps({
                "id":              ex.get("id"),
                "prompt":          ex["prompt"],
                "true_harm":       true_harm,
                "predicted_harm":  pred_harm,
                "generated_intent": intent,
                "raw_generation":  raw_text,
            }, ensure_ascii=False) + "\n")

    print(f"Predictions saved to {pred_path}")
    compute_and_log_metrics(all_refs, all_preds, split_name="test_dpo")

    del llm


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Data
    p.add_argument("--pairs-path",   type=str,
                   default="data/dpo_pairs/train_t0.8/dpo_pairs.jsonl")
    p.add_argument("--val-split",    type=float, default=0.1,
                   help="Fraction of DPO pairs held out for validation.")

    # Model
    p.add_argument("--adapter-path", type=str,
                   default="trained_models/causal/hyperparam_sweep/lr_0.0005_e_5_adapter")
    p.add_argument("--base-model",   type=str,
                   default="meta-llama/Llama-3.1-8B-Instruct")

    # Output
    p.add_argument("--output-dir",   type=str,
                   default="trained_models/causal/llama-dpo")

    # DPO hyperparameters
    p.add_argument("--beta",                type=float, default=0.1)
    p.add_argument("--loss-type",           type=str,   default="sigmoid",
                   choices=["sigmoid", "ipo", "hinge", "robust"],
                   help="DPO loss formulation.")
    p.add_argument("--label-smoothing",     type=float, default=0.0,
                   help="Label smoothing for DPO loss (0.0 = standard).")
    p.add_argument("--epochs",              type=int,   default=3)
    p.add_argument("--learning-rate",       type=float, default=5e-5)
    p.add_argument("--batch-size",          type=int,   default=2)
    p.add_argument("--gradient-accumulation", type=int, default=8)
    p.add_argument("--max-length",          type=int,   default=512,
                   help="Max total length (prompt + completion) for DPO.")
    p.add_argument("--max-prompt-length",   type=int,   default=448,
                   help="Max prompt length; completion gets the remaining tokens.")

    # Misc
    p.add_argument("--seed",          type=int,  default=22)
    p.add_argument("--skip-eval",     action="store_true",
                   help="Skip vLLM evaluation after training.")
    p.add_argument("--wandb-project", type=str,  default="intention-jailbreak",
                   help="Weights & Biases project name.")
    p.add_argument("--wandb-run",     type=str,  default=None,
                   help="W&B run name (auto-generated if not set).")

    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
