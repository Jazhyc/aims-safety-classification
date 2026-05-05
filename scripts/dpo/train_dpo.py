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
    python scripts/dpo/train_dpo.py \\
        --pairs-path     data/dpo_pairs/train_t0.8/dpo_pairs.jsonl \\
        --adapter-path   Jazhyc/llama-3.1-8b-sft-generation \\
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
from peft import PeftModel, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch.nn as nn
from trl import DPOConfig, DPOTrainer

DEFAULT_SFT_ADAPTER = "Jazhyc/llama-3.1-8b-sft-generation"


# ---------------------------------------------------------------------------
# Weighted DPO trainer
# ---------------------------------------------------------------------------

class WeightedDPOTrainer(DPOTrainer):
    """DPOTrainer that up-weights harmful pairs to correct class imbalance.

    The standard DPO loss averages uniformly over the batch. When chosen labels
    are skewed (e.g. 58% safe), the model learns a safe-bias. Setting
    harmful_weight > 1 rescales the per-sample loss so harmful pairs contribute
    proportionally more, without discarding any data.

    harmful_weight = n_safe / n_harmful  (≈ 1.4 for the unbalanced pairs file)
    """

    def __init__(self, *args, harmful_weight: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self._harmful_weight = harmful_weight
        self._batch_weights: torch.Tensor | None = None

    def get_batch_loss_metrics(self, model, batch, train_eval="train"):
        # Stash per-sample weights so dpo_loss can apply them
        if self._harmful_weight != 1.0 and "gold_harm" in batch:
            w = [self._harmful_weight if h == "harmful" else 1.0
                 for h in batch["gold_harm"]]
            self._batch_weights = torch.tensor(w, dtype=torch.float32)
        else:
            self._batch_weights = None
        return super().get_batch_loss_metrics(model, batch, train_eval)

    def dpo_loss(self, chosen_logps, rejected_logps,
                 ref_chosen_logps, ref_rejected_logps,
                 loss_type="sigmoid", model_output=None):
        losses, chosen_rewards, rejected_rewards = super().dpo_loss(
            chosen_logps, rejected_logps,
            ref_chosen_logps, ref_rejected_logps,
            loss_type=loss_type, model_output=model_output,
        )
        if self._batch_weights is not None:
            w = self._batch_weights.to(losses.device)
            # Normalise so weighted mean ≈ unweighted mean in scale
            w = w / w.mean()
            losses = losses * w
        return losses, chosen_rewards, rejected_rewards

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from intention_jailbreak.training import set_all_seeds, print_gpu_info
from intention_jailbreak.model_generation.parsing import extract_intent_and_harm
from intention_jailbreak.model_generation.prompt_templates import GENERATION_SYSTEM_PROMPT
from intention_jailbreak.model_generation.causal import _load_causal_lm, _fix_lora_keys_for_vllm


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

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


def load_dpo_dataset(pairs_path: str, tokenizer,
                     include_gold_harm: bool = False) -> Dataset:
    """
    Load dpo_pairs.jsonl and return a HuggingFace Dataset with columns:
      prompt, chosen, rejected  (and optionally gold_harm for weighted loss)
    The prompt is formatted with the model chat template used for SFT inference.
    """
    records = []
    with open(pairs_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            rec = {
                "prompt":   _apply_chat_template(
                    tokenizer,
                    [
                        {"role": "system", "content": GENERATION_SYSTEM_PROMPT},
                        {"role": "user", "content": d["prompt"]},
                    ],
                ),
                "chosen":   d["chosen"],
                "rejected": d["rejected"],
            }
            if include_gold_harm:
                rec["gold_harm"] = d.get("gold_harm", "safe")
            records.append(rec)
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


def _resolve_local_adapter_path(adapter_ref: str, revision: str | None = None) -> str:
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


def _load_base_tokenizer(base_model: str):
    """Load tokenizer from local HF cache snapshot for base_model when possible."""
    import os as _os

    # Resolve local snapshot path to avoid network calls (is_base_mistral check).
    # HF_HUB_CACHE takes priority (set in the SLURM template); fall back to
    # HF_HOME/hub, then the standard ~/.cache/huggingface/hub default.
    hf_hub_cache = _os.environ.get(
        "HF_HUB_CACHE",
        _os.path.join(
            _os.environ.get("HF_HOME", _os.path.expanduser("~/.cache/huggingface")),
            "hub",
        ),
    )
    model_id = base_model.replace("/", "--")
    snapshots = _os.path.join(hf_hub_cache, f"models--{model_id}", "snapshots")
    tok_path = base_model
    if _os.path.isdir(snapshots):
        # Pick the newest snapshot that actually has tokenizer assets.
        # Some cache snapshots can be incomplete (e.g., interrupted downloads).
        for snap in sorted(_os.listdir(snapshots), reverse=True):
            candidate = _os.path.join(snapshots, snap)
            if not _os.path.isdir(candidate):
                continue
            if _os.path.exists(_os.path.join(candidate, "tokenizer.json")) or _os.path.exists(
                _os.path.join(candidate, "tokenizer.model")
            ):
                tok_path = candidate
                break
    # use_fast=True avoids instantiating LlamaTokenizer (slow/sentencepiece)
    # which fails for Llama 3.x models that have no tokenizer.model file.
    try:
        tokenizer = AutoTokenizer.from_pretrained(tok_path, local_files_only=True, use_fast=True)
    except (TypeError, OSError, ValueError):
        # Fallback: let HF resolve from model id within the explicit cache dir.
        tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            cache_dir=hf_hub_cache,
            local_files_only=True,
            use_fast=True,
        )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _adapter_uses_conditional_gen_keys(adapter_path: str | None) -> bool:
    """Return True if the adapter was trained on a ForConditionalGeneration model.

    ForConditionalGeneration adapters have 'language_model.' in their key names
    (e.g. 'base_model.model.language_model.model.layers.0...').
    Language-tower adapters (from _load_causal_lm) have 'base_model.model.model.layers.0...'.
    When ForConditionalGeneration keys are detected, we must NOT use _load_causal_lm
    (tower extraction would cause a key mismatch) and must NOT apply _fix_lora_keys_for_vllm
    (keys are already in the correct format for vLLM).
    """
    if not adapter_path:
        return False
    import os
    adapter_file = os.path.join(adapter_path, "adapter_model.safetensors")
    if not os.path.exists(adapter_file):
        return False
    try:
        from safetensors import safe_open
        with safe_open(adapter_file, framework="pt") as f:
            keys = list(f.keys())
        return bool(keys) and "language_model." in keys[0]
    except Exception:
        return False


def _load_base_model(base_model: str, attn_implementation: str,
                     use_lm_wrapper: bool = True):
    """Load base model in 4-bit.

    use_lm_wrapper=True  : use _load_causal_lm (tower extraction for new LoRAs on
                           ForConditionalGeneration models like Gemma3).
    use_lm_wrapper=False : load with AutoModelForCausalLM directly — required when
                           the adapter was already trained on ForConditionalGeneration
                           keys (language_model. prefix) so PEFT key matching works.
    """
    kwargs = dict(
        quantization_config=_bnb_config(),
        device_map="auto",
        attn_implementation=attn_implementation,
    )
    if use_lm_wrapper:
        return _load_causal_lm(base_model, **kwargs)
    return AutoModelForCausalLM.from_pretrained(base_model, **kwargs)


def load_policy_model(base_model: str, adapter_path: str | None, attn_implementation: str,
                      init_new_lora: bool = False, lora_r: int = 16,
                      lora_alpha: int = 32, lora_dropout: float = 0.05,
                      use_lm_wrapper: bool = True):
    """Load base model in 4-bit and attach a LoRA adapter for training.

    When adapter_path is provided, loads the existing LoRA checkpoint (standard
    DPO or curriculum phase-2 policy init). When init_new_lora is True, creates
    a fresh LoRA adapter instead — used when the base model is already a merged
    SFT checkpoint with no separate LoRA to load.
    """
    from peft import LoraConfig, get_peft_model
    model = _load_base_model(base_model, attn_implementation, use_lm_wrapper=use_lm_wrapper)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    if init_new_lora:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"]
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        print(f"Policy model loaded with fresh LoRA (r={lora_r}, alpha={lora_alpha}).")
    else:
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
        print(f"Policy model loaded from {adapter_path}.")
    print(f"  Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    return model


def load_ref_model(base_model: str, adapter_path: str | None, attn_implementation: str,
                   init_new_lora: bool = False, use_lm_wrapper: bool = True):
    """Load base model in 4-bit as the frozen DPO reference.

    When adapter_path is provided, attaches the LoRA adapter frozen. When
    init_new_lora is True (merged-SFT mode), the base model itself is the
    reference — no adapter is needed.
    """
    ref = _load_base_model(base_model, attn_implementation, use_lm_wrapper=use_lm_wrapper)
    if not init_new_lora and adapter_path:
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

    # Resolve adapter paths (reference always from --adapter-path; policy from
    # --policy-adapter if set, otherwise same as reference).
    local_ref_adapter = (
        _resolve_local_adapter_path(args.adapter_path, args.adapter_revision)
        if args.adapter_path and not args.init_new_lora
        else None
    )
    if args.policy_adapter:
        local_policy_adapter = _resolve_local_adapter_path(args.policy_adapter)
    elif args.init_new_lora:
        local_policy_adapter = None
    else:
        local_policy_adapter = local_ref_adapter

    # ── Weights & Biases ──────────────────────────────────────────────────
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run,
        config={
            "base_model":      args.base_model,
            "adapter_path":    args.adapter_path,
            "adapter_revision": args.adapter_revision,
            "policy_adapter":  args.policy_adapter,
            "init_new_lora":   args.init_new_lora,
            "pairs_path":      args.pairs_path,
            "beta":            args.beta,
            "loss_type":       args.loss_type,
            "label_smoothing": args.label_smoothing,
            "epochs":          args.epochs,
            "learning_rate":   args.learning_rate,
            "batch_size":      args.batch_size,
            "gradient_accumulation": args.gradient_accumulation,
            "max_length":      args.max_length,
            "seed":            args.seed,
            "harmful_weight":  args.harmful_weight,
        },
    )

    # ── Tokenizer ─────────────────────────────────────────────────────────
    tokenizer = _load_base_tokenizer(args.base_model)

    # ── Data ──────────────────────────────────────────────────────────────
    use_weighted = args.harmful_weight > 1.0
    train_dataset = load_dpo_dataset(args.pairs_path, tokenizer, include_gold_harm=use_weighted)
    print(f"Train: {len(train_dataset)} pairs")

    # ── Models ────────────────────────────────────────────────────────────
    # Detect key format: if the adapter uses ForConditionalGeneration keys
    # (language_model. prefix), bypass _load_causal_lm — tower extraction would
    # cause a key mismatch and silently initialize random weights instead of
    # loading the adapter's trained weights.
    adapter_for_detection = local_policy_adapter or local_ref_adapter
    use_lm_wrapper = not _adapter_uses_conditional_gen_keys(adapter_for_detection)
    if not use_lm_wrapper:
        print("[model loading] Detected ForConditionalGeneration adapter keys — "
              "bypassing _load_causal_lm tower extraction.")

    policy_model = load_policy_model(
        args.base_model, local_policy_adapter, args.attn_implementation,
        init_new_lora=args.init_new_lora,
        lora_r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        use_lm_wrapper=use_lm_wrapper,
    )
    ref_model = load_ref_model(
        args.base_model, local_ref_adapter, args.attn_implementation,
        init_new_lora=args.init_new_lora,
        use_lm_wrapper=use_lm_wrapper,
    )

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

        # No eval during training — fixed epochs, final model saved explicitly
        eval_strategy="no",
        save_strategy="no",

        # Misc
        gradient_checkpointing=True,
        remove_unused_columns=False,
        report_to="wandb",
        seed=args.seed,
    )

    trainer_cls = WeightedDPOTrainer if use_weighted else DPOTrainer
    trainer_kwargs = dict(
        model=policy_model,
        ref_model=ref_model,
        args=dpo_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )
    if use_weighted:
        trainer_kwargs["harmful_weight"] = args.harmful_weight
        print(f"Using WeightedDPOTrainer (harmful_weight={args.harmful_weight:.2f})")
    trainer = trainer_cls(**trainer_kwargs)

    print("\n=== Starting DPO training ===")
    trainer.train()

    # ── Save adapter ───────────────────────────────────────────────────────
    adapter_save_dir = str(output_dir) + "_adapter"
    os.makedirs(adapter_save_dir, exist_ok=True)
    trainer.model.save_pretrained(adapter_save_dir)
    tokenizer.save_pretrained(adapter_save_dir)
    # Fix LoRA keys for vLLM only when using the language-tower path (_load_causal_lm).
    # When the adapter already uses ForConditionalGeneration keys (language_model. prefix),
    # the saved keys are already correct for vLLM — applying the fix would double-prefix them.
    if use_lm_wrapper:
        _fix_lora_keys_for_vllm(adapter_save_dir, trainer.model)
    print(f"DPO adapter saved to {adapter_save_dir}")

    # Save config summary
    summary = {
        "base_model":      args.base_model,
        "sft_adapter":     args.adapter_path,
        "sft_adapter_revision": args.adapter_revision,
        "policy_adapter":  args.policy_adapter,
        "init_new_lora":   args.init_new_lora,
        "dpo_adapter":     adapter_save_dir,
        "pairs_path":      args.pairs_path,
        "beta":            args.beta,
        "epochs":          args.epochs,
        "learning_rate":   args.learning_rate,
        "n_train":         len(train_dataset),
        "harmful_weight":  args.harmful_weight,
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

    from intention_jailbreak.model_generation.preprocessing import preprocess_data
    from intention_jailbreak.model_generation.data_utils import apply_binary_harm_mapping
    from intention_jailbreak.model_generation.evaluate_generations import compute_and_log_metrics

    print("\n=== Loading test dataset for evaluation (HF test split) ===")
    test_dataset = preprocess_data(split="test")
    test_dataset = apply_binary_harm_mapping(test_dataset, binary_harm_mapping=True)
    print(f"Test set: {len(test_dataset)} examples")

    print(f"\n=== Loading DPO model with vLLM ===")
    llm = LLM(
        model=args.base_model,
        tokenizer=adapter_save_dir,
        enable_lora=True,
        max_lora_rank=64,
        max_loras=1,
        limit_mm_per_prompt={"image": 0},
        gpu_memory_utilization=0.75,
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

    # Model
    p.add_argument("--adapter-path", type=str, default=DEFAULT_SFT_ADAPTER,
                   help="SFT LoRA adapter used as the *reference* model (frozen). "
                        "Also used as the policy starting point unless --policy-adapter "
                        "or --init-new-lora is set. Set to None with --init-new-lora "
                        "when the base model is already a merged SFT checkpoint.")
    p.add_argument("--adapter-revision", type=str, default=None,
                   help="Optional HF revision (branch/tag/commit) for --adapter-path.")
    p.add_argument("--policy-adapter", type=str, default=None,
                   help="Override the *policy* starting adapter independently of the "
                        "reference (--adapter-path). Useful for curriculum training: "
                        "phase 2 sets --adapter-path=SFT and --policy-adapter=phase1_DPO "
                        "so the reference stays at SFT while the policy starts warm.")
    p.add_argument("--init-new-lora", action="store_true", default=False,
                   help="Initialize a fresh LoRA adapter instead of loading an existing "
                        "one. Use when --base-model is already a merged SFT checkpoint "
                        "(e.g. Gemma-based DPO where the SFT is a full HF model). "
                        "--adapter-path is ignored for model loading when this is set.")
    p.add_argument("--lora-r",       type=int,   default=16,
                   help="LoRA rank when --init-new-lora is used.")
    p.add_argument("--lora-alpha",   type=int,   default=32,
                   help="LoRA alpha when --init-new-lora is used.")
    p.add_argument("--lora-dropout", type=float, default=0.05,
                   help="LoRA dropout when --init-new-lora is used.")
    p.add_argument("--base-model",   type=str,
                   default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument(
        "--attn-implementation",
        type=str,
        default="flash_attention_2",
        choices=["flash_attention_2", "sdpa", "eager"],
        help="Attention backend for loading policy/reference models. "
             "Use 'sdpa' if flash_attention_2 is unavailable in your environment.",
    )

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
    p.add_argument("--harmful-weight",      type=float, default=1.0,
                   help="Loss weight for harmful pairs (1.0 = uniform). "
                        "Set to n_safe/n_harmful (e.g. 1.4) to correct class imbalance "
                        "without discarding any pairs.")
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
