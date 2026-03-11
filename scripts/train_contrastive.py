"""
Contrastive (InfoNCE) fine-tuning on top of the SFT Llama 3.1-8B LoRA adapter.

For each training prompt the loss is:
    L = -log( exp(s_+) / (exp(s_+) + Σ_j exp(s_j^-)) )

where s(x) = Σ log p_θ(x_t | x_<t, prompt) is the mean per-token log-prob
of a completion under the current policy.

This explicitly increases the margin between the chosen (ground-truth) and
every rejected (model-sampled, wrong-label) completion.

Data format (contrastive_pairs.jsonl):
  {
    "prompt":   "...",
    "chosen":   "Intent: ...; Harm: ...",
    "rejecteds": [
      {"text": "Intent: ...; Harm: ...", ...},
      ...
    ]
  }

Usage (from project root):
    python scripts/train_contrastive.py \\
        --pairs-path     data/dpo_pairs/train_t0.8/contrastive_pairs.jsonl \\
        --adapter-path   trained_models/causal/hyperparam_sweep/lr_0.0005_e_5_adapter \\
        --base-model     meta-llama/Llama-3.1-8B-Instruct \\
        --output-dir     trained_models/causal/llama-contrastive
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
import torch.nn.functional as F
from peft import PeftModel, prepare_model_for_kbit_training
from torch.utils.data import DataLoader, Dataset as TorchDataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, get_cosine_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from intention_jailbreak.training import set_all_seeds, print_gpu_info


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ContrastiveDataset(TorchDataset):
    """
    Each item: one prompt with its chosen completion and ≥1 rejected completions.
    Returns raw strings; collation and tokenisation happen in the training loop.
    """

    def __init__(self, records: list):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.records[idx]


def load_contrastive_records(pairs_path: str, prompt_suffix: str = "\n") -> list:
    records = []
    with open(pairs_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            rejecteds = [r["text"] for r in d["rejecteds"]]
            if not rejecteds:
                continue
            records.append({
                "prompt":    d["prompt"] + prompt_suffix,
                "chosen":    d["chosen"],
                "rejecteds": rejecteds,
            })
    print(f"Loaded {len(records)} contrastive examples from {pairs_path}")
    if records:
        ex = records[0]
        print(f"  Example prompt     : {ex['prompt'][:80]}")
        print(f"  Example chosen     : {ex['chosen']}")
        print(f"  Example rejecteds  : {ex['rejecteds']}")
        neg_counts = [len(r["rejecteds"]) for r in records]
        import numpy as np
        print(f"  Negatives/prompt   : mean={np.mean(neg_counts):.2f} "
              f"max={max(neg_counts)}")
    return records


# ---------------------------------------------------------------------------
# Log-prob computation
# ---------------------------------------------------------------------------

def compute_completion_logprob(
    model,
    tokenizer,
    prompt: str,
    completion: str,
    max_length: int,
    device,
) -> torch.Tensor:
    """
    Return the mean per-token log-prob of `completion` given `prompt`.

    We concatenate [prompt + completion], run a forward pass, then
    sum the log-probs at the completion token positions only.
    Using mean (not sum) normalises for completion length differences.
    """
    prompt_ids     = tokenizer(prompt,     add_special_tokens=True,  return_tensors="pt").input_ids
    completion_ids = tokenizer(completion, add_special_tokens=False, return_tensors="pt").input_ids

    # Truncate so the full sequence fits in max_length
    max_completion = max_length - prompt_ids.shape[1]
    if max_completion <= 0:
        raise ValueError("Prompt exceeds max_length; shorten it.")
    completion_ids = completion_ids[:, :max_completion]

    input_ids = torch.cat([prompt_ids, completion_ids], dim=1).to(device)

    with torch.no_grad() if not model.training else torch.enable_grad():
        logits = model(input_ids=input_ids).logits  # (1, seq_len, vocab)

    # Shift: logits at position t predict token t+1
    shift_logits = logits[:, :-1, :]           # (1, seq_len-1, vocab)
    shift_labels = input_ids[:, 1:]            # (1, seq_len-1)

    log_probs = F.log_softmax(shift_logits, dim=-1)  # (1, seq_len-1, vocab)

    # Gather the log-prob of each actual next token
    token_logprobs = log_probs.gather(
        2, shift_labels.unsqueeze(-1)
    ).squeeze(-1)  # (1, seq_len-1)

    # Only count the completion positions
    prompt_len     = prompt_ids.shape[1]
    completion_len = completion_ids.shape[1]

    # completion tokens start at position prompt_len in input_ids,
    # which corresponds to positions prompt_len-1 in the shifted sequence
    comp_start = prompt_len - 1
    comp_end   = comp_start + completion_len

    completion_logprobs = token_logprobs[:, comp_start:comp_end]  # (1, comp_len)
    return completion_logprobs.mean()  # scalar


def infonce_loss(chosen_score: torch.Tensor, rejected_scores: list) -> torch.Tensor:
    """
    InfoNCE loss given one chosen score and a list of rejected scores (all scalars).
    L = -log( exp(s_+) / (exp(s_+) + Σ exp(s_j^-)) )
    """
    all_scores = torch.stack([chosen_score] + rejected_scores)  # (1 + N,)
    return -chosen_score + torch.logsumexp(all_scores, dim=0)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    set_all_seeds(args.seed)
    print_gpu_info()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Data ──────────────────────────────────────────────────────────────
    records = load_contrastive_records(args.pairs_path)

    split_idx  = int(len(records) * (1 - args.val_split))
    train_recs = records[:split_idx]
    val_recs   = records[split_idx:]
    print(f"Train: {len(train_recs)} | Val: {len(val_recs)}")

    train_ds = ContrastiveDataset(train_recs)
    val_ds   = ContrastiveDataset(val_recs)

    # ── Tokenizer ─────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(args.adapter_path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Model (policy only — no frozen reference needed for InfoNCE) ───────
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation="sdpa",
    )
    base.config.use_cache = False
    base.gradient_checkpointing_enable()
    base = prepare_model_for_kbit_training(base)
    model = PeftModel.from_pretrained(base, args.adapter_path, is_trainable=True)
    model.train()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable:,}")

    # ── Optimiser & scheduler ─────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=0.01,
        betas=(0.9, 0.98),
    )

    steps_per_epoch = len(train_ds) // args.grad_accum
    total_steps     = steps_per_epoch * args.epochs
    warmup_steps    = int(0.1 * total_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # ── Training ──────────────────────────────────────────────────────────
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss  = float("inf")
    best_ckpt_dir  = str(output_dir) + "_adapter_best"

    print(f"\n=== Starting contrastive (InfoNCE) training ===")
    print(f"  Epochs            : {args.epochs}")
    print(f"  LR                : {args.learning_rate}")
    print(f"  Grad accumulation : {args.grad_accum}")
    print(f"  Max length        : {args.max_length}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss   = 0.0
        n_steps      = 0
        accum_loss   = torch.tensor(0.0, device=device)

        # Shuffle manually (no DataLoader batching — each example is variable-size)
        import random
        indices = list(range(len(train_ds)))
        random.shuffle(indices)

        optimizer.zero_grad()

        for step_idx, idx in enumerate(indices):
            rec = train_ds[idx]
            try:
                chosen_score = compute_completion_logprob(
                    model, tokenizer,
                    rec["prompt"], rec["chosen"],
                    args.max_length, device,
                )
                rejected_scores = [
                    compute_completion_logprob(
                        model, tokenizer,
                        rec["prompt"], rej,
                        args.max_length, device,
                    )
                    for rej in rec["rejecteds"]
                ]
            except ValueError:
                continue  # prompt too long — skip

            loss = infonce_loss(chosen_score, rejected_scores)
            loss = loss / args.grad_accum
            loss.backward()
            accum_loss = accum_loss + loss.detach()

            if (step_idx + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                epoch_loss += accum_loss.item()
                n_steps    += 1
                accum_loss  = torch.tensor(0.0, device=device)

                if n_steps % 50 == 0:
                    print(f"  Epoch {epoch} step {n_steps}/{steps_per_epoch} "
                          f"loss={epoch_loss/n_steps:.4f}")

        avg_train_loss = epoch_loss / max(n_steps, 1)

        # ── Validation ────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        n_val    = 0
        with torch.no_grad():
            for rec in val_ds:
                try:
                    cs = compute_completion_logprob(
                        model, tokenizer,
                        rec["prompt"], rec["chosen"],
                        args.max_length, device,
                    )
                    rs = [
                        compute_completion_logprob(
                            model, tokenizer,
                            rec["prompt"], rej,
                            args.max_length, device,
                        )
                        for rej in rec["rejecteds"]
                    ]
                except ValueError:
                    continue
                val_loss += infonce_loss(cs, rs).item()
                n_val    += 1

        avg_val_loss = val_loss / max(n_val, 1)
        print(f"Epoch {epoch}/{args.epochs}  "
              f"train_loss={avg_train_loss:.4f}  val_loss={avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs(best_ckpt_dir, exist_ok=True)
            model.save_pretrained(best_ckpt_dir)
            tokenizer.save_pretrained(best_ckpt_dir)
            print(f"  → Best checkpoint saved (val_loss={best_val_loss:.4f})")

    # ── Final adapter save ─────────────────────────────────────────────────
    adapter_save_dir = str(output_dir) + "_adapter"
    os.makedirs(adapter_save_dir, exist_ok=True)
    model.save_pretrained(adapter_save_dir)
    tokenizer.save_pretrained(adapter_save_dir)
    print(f"Final adapter saved to {adapter_save_dir}")

    summary = {
        "base_model":      args.base_model,
        "sft_adapter":     args.adapter_path,
        "contrastive_adapter": adapter_save_dir,
        "best_adapter":    best_ckpt_dir,
        "pairs_path":      args.pairs_path,
        "epochs":          args.epochs,
        "learning_rate":   args.learning_rate,
        "n_train":         len(train_recs),
        "n_val":           len(val_recs),
        "best_val_loss":   best_val_loss,
    }
    with open(output_dir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ── vLLM evaluation ────────────────────────────────────────────────────
    del model
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()

    if not args.skip_eval:
        _run_vllm_eval(args, best_ckpt_dir)


# ---------------------------------------------------------------------------
# vLLM evaluation (mirrors DPO script)
# ---------------------------------------------------------------------------

def _run_vllm_eval(args, adapter_dir: str):
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from intention_jailbreak.model_generation.preprocessing import preprocess_data
    from intention_jailbreak.model_generation.data_utils import (
        apply_binary_harm_mapping,
        train_val_test_split,
    )
    from intention_jailbreak.model_generation.evaluate_generations import compute_and_log_metrics

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

    print(f"\n=== Loading contrastive model with vLLM ===")
    llm = LLM(
        model=args.base_model,
        tokenizer=adapter_dir,
        enable_lora=True,
        max_lora_rank=64,
        max_loras=1,
        limit_mm_per_prompt={"image": 0},
        gpu_memory_utilization=0.90,
        max_model_len=2048,
        dtype="bfloat16",
        enforce_eager=True,
    )
    lora_request = LoRARequest("contrastive_lora", 1, adapter_dir)

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
            raw_text   = output.outputs[0].text.strip()
            intent, pred_harm = extract_intent_and_harm(raw_text)
            true_harm  = ex.get("Annotator Harm")
            all_preds.append(pred_harm or "")
            all_refs.append(true_harm or "")
            f.write(json.dumps({
                "id":               ex.get("id"),
                "prompt":           ex["prompt"],
                "true_harm":        true_harm,
                "predicted_harm":   pred_harm,
                "generated_intent": intent,
                "raw_generation":   raw_text,
            }, ensure_ascii=False) + "\n")

    print(f"Predictions saved to {pred_path}")
    compute_and_log_metrics(all_refs, all_preds, split_name="test_contrastive")
    del llm


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Data
    p.add_argument("--pairs-path",   type=str,
                   default="data/dpo_pairs/train_t0.8/contrastive_pairs.jsonl")
    p.add_argument("--val-split",    type=float, default=0.1)

    # Model
    p.add_argument("--adapter-path", type=str,
                   default="trained_models/causal/hyperparam_sweep/lr_0.0005_e_5_adapter")
    p.add_argument("--base-model",   type=str,
                   default="meta-llama/Llama-3.1-8B-Instruct")

    # Output
    p.add_argument("--output-dir",   type=str,
                   default="trained_models/causal/llama-contrastive")

    # Hyperparameters
    p.add_argument("--epochs",          type=int,   default=3)
    p.add_argument("--learning-rate",   type=float, default=5e-5)
    p.add_argument("--grad-accum",      type=int,   default=8,
                   help="Gradient accumulation steps (effective batch size = grad_accum).")
    p.add_argument("--max-length",      type=int,   default=512)

    # Misc
    p.add_argument("--seed",       type=int,  default=22)
    p.add_argument("--skip-eval",  action="store_true")

    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
