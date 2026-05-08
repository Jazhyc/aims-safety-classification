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
import platform
import shutil
import subprocess
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _safe_version(pkg_name: str) -> str | None:
    try:
        from importlib.metadata import version
        return version(pkg_name)
    except Exception:
        return None


def _git_commit() -> str | None:
    try:
        import subprocess
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def _f1_from_jsonl(path: Path) -> float | None:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    y_true, y_pred = [], []
    for r in rows:
        t = r.get("true_harm_binary")
        p = r.get("predicted_harm")
        if t is None or p is None:
            continue
        y_true.append(str(t).lower())
        y_pred.append(str(p).lower())
    if not y_true:
        return None
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == "harmful" and p == "harmful")
    fp = sum(1 for t, p in zip(y_true, y_pred) if t != "harmful" and p == "harmful")
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == "harmful" and p != "harmful")
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0


def _pick_best_checkpoint_by_ood(args, tokenizer, trainer, output_dir: Path) -> dict | None:
    """Evaluate each saved checkpoint on OOD (toxic_chat + aegis), return best."""
    ckpt_root = output_dir / "checkpoints"
    if not ckpt_root.exists():
        return None
    ckpts = sorted([p for p in ckpt_root.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")])
    if not ckpts:
        return None

    model_slug = args.base_model.replace("/", "_")
    best = None
    results = []

    for ckpt in ckpts:
        ood_out = output_dir / "ood_checkpoint_eval" / ckpt.name
        cmd = [
            sys.executable,
            "scripts/eval_safety_classifier.py",
            "--config-name=eval_ood_validation",
            f"finetuned.generation_adapter={ckpt}",
            "experiment.conditions=[finetuned_generation]",
            f"paths.output_dir={ood_out}",
            f"model.name={args.base_model}",
            f"lora.rank={args.lora_r}",
        ]
        print(f"\n=== OOD checkpoint eval: {ckpt.name} ===")
        res = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
        if res.returncode != 0:
            print(f"[WARN] OOD eval failed for {ckpt.name}; skipping checkpoint.")
            continue

        toxic_p = ood_out / "toxic_chat" / f"{model_slug}_finetuned_generation.jsonl"
        aegis_p = ood_out / "aegis" / f"{model_slug}_finetuned_generation.jsonl"
        if not toxic_p.exists() or not aegis_p.exists():
            print(f"[WARN] Missing OOD outputs for {ckpt.name}; skipping checkpoint.")
            continue
        toxic_f1 = _f1_from_jsonl(toxic_p)
        aegis_f1 = _f1_from_jsonl(aegis_p)
        if toxic_f1 is None or aegis_f1 is None:
            print(f"[WARN] Could not parse OOD F1 for {ckpt.name}; skipping checkpoint.")
            continue
        avg = (toxic_f1 + aegis_f1) / 2
        row = {
            "checkpoint": ckpt.name,
            "path": str(ckpt),
            "toxic_f1": toxic_f1,
            "aegis_f1": aegis_f1,
            "ood_avg": avg,
        }
        results.append(row)
        print(f"  toxic={toxic_f1:.4f}  aegis={aegis_f1:.4f}  avg={avg:.4f}")
        if best is None or avg > best["ood_avg"]:
            best = row

    if not results or best is None:
        return None

    adapter_save_dir = str(output_dir) + "_adapter"
    if os.path.exists(adapter_save_dir):
        shutil.rmtree(adapter_save_dir)
    shutil.copytree(best["path"], adapter_save_dir)
    tokenizer.save_pretrained(adapter_save_dir)
    _fix_lora_keys_for_vllm(adapter_save_dir, trainer.model)
    print(f"[best-checkpoint] Selected {best['checkpoint']} by OOD avg={best['ood_avg']:.4f}")

    return {"best": best, "all": sorted(results, key=lambda r: r["ood_avg"], reverse=True)}


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

    Such adapters have 'language_model.' in their key names, e.g.
    'base_model.model.language_model.model.layers.0...'.
    Language-tower adapters (from _load_causal_lm) use 'base_model.model.model.layers.0...'.
    """
    if not adapter_path:
        return False
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


def _translate_adapter_keys_to_language_tower(src_path: str) -> str:
    """Create a temp copy of a ForConditionalGeneration adapter with keys translated
    to the language-tower format expected by _load_causal_lm.

    Renames 'base_model.model.language_model.' → 'base_model.model.' so the adapter
    matches the CausalLM wrapper that _load_causal_lm produces from the language tower.
    The original adapter is not modified. The caller is responsible for cleanup.
    """
    import shutil
    import tempfile
    from safetensors.torch import load_file, save_file

    tmp = tempfile.mkdtemp(prefix="dpo_adapter_translated_")
    src = Path(src_path)
    for f in src.iterdir():
        if f.name != "adapter_model.safetensors" and f.is_file():
            shutil.copy2(f, Path(tmp) / f.name)

    prefix_old = "base_model.model.language_model."
    prefix_new = "base_model.model."
    tensors = load_file(str(src / "adapter_model.safetensors"))
    translated = {
        (prefix_new + k[len(prefix_old):] if k.startswith(prefix_old) else k): v
        for k, v in tensors.items()
    }
    save_file(translated, str(Path(tmp) / "adapter_model.safetensors"))
    n_translated = sum(1 for k in tensors if k.startswith(prefix_old))
    print(f"[key translation] Translated {n_translated}/{len(tensors)} keys "
          f"from ForConditionalGeneration → language-tower format in {tmp}")
    return tmp


def _load_base_model(base_model: str, attn_implementation: str):
    """Load base model in 4-bit via _load_causal_lm (handles tower extraction)."""
    return _load_causal_lm(
        base_model,
        quantization_config=_bnb_config(),
        device_map="auto",
        attn_implementation=attn_implementation,
    )


def load_policy_model(base_model: str, adapter_path: str | None, attn_implementation: str,
                      init_new_lora: bool = False, lora_r: int = 16,
                      lora_alpha: int = 32, lora_dropout: float = 0.05):
    """Load base model in 4-bit and attach a LoRA adapter for training.

    When adapter_path is provided, loads the existing LoRA checkpoint (standard
    DPO or curriculum phase-2 policy init). When init_new_lora is True, creates
    a fresh LoRA adapter instead — used when the base model is already a merged
    SFT checkpoint with no separate LoRA to load.
    """
    from peft import LoraConfig, get_peft_model
    model = _load_base_model(base_model, attn_implementation)
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
                   init_new_lora: bool = False):
    """Load base model in 4-bit as the frozen DPO reference.

    When adapter_path is provided, attaches the LoRA adapter frozen. When
    init_new_lora is True (merged-SFT mode), the base model itself is the
    reference — no adapter is needed.
    """
    ref = _load_base_model(base_model, attn_implementation)
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
    # If either adapter was trained on a ForConditionalGeneration model (keys have
    # 'language_model.' prefix), translate keys to language-tower format so PEFT
    # can load them into the CausalLM wrapper produced by _load_causal_lm.
    # _fix_lora_keys_for_vllm restores the language_model. prefix after training.
    _tmp_dirs: list[str] = []
    adapter_for_detection = local_policy_adapter or local_ref_adapter
    if _adapter_uses_conditional_gen_keys(adapter_for_detection):
        print("[model loading] ForConditionalGeneration adapter keys detected — "
              "translating to language-tower format for _load_causal_lm.")
        orig_ref    = local_ref_adapter
        orig_policy = local_policy_adapter
        translated: dict[str, str] = {}  # original path → translated tmp path
        for orig in {orig_ref, orig_policy} - {None}:
            translated[orig] = _translate_adapter_keys_to_language_tower(orig)
            _tmp_dirs.append(translated[orig])
        if orig_ref:
            local_ref_adapter = translated[orig_ref]
        if orig_policy:
            local_policy_adapter = translated[orig_policy]

    policy_model = load_policy_model(
        args.base_model, local_policy_adapter, args.attn_implementation,
        init_new_lora=args.init_new_lora,
        lora_r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
    )
    ref_model = load_ref_model(
        args.base_model, local_ref_adapter, args.attn_implementation,
        init_new_lora=args.init_new_lora,
    )

    # ── DPO config ────────────────────────────────────────────────────────
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dpo_kwargs = dict(
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
        save_strategy=args.save_strategy,
        save_total_limit=args.save_total_limit,

        # Misc
        gradient_checkpointing=True,
        remove_unused_columns=False,
        report_to="wandb",
        seed=args.seed,
    )
    if args.save_strategy == "steps":
        dpo_kwargs["save_steps"] = args.save_steps
    dpo_config = DPOConfig(**dpo_kwargs)

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
    # Rename language-tower keys back to language_model. prefix for vLLM.
    # _fix_lora_keys_for_vllm is a no-op for models without _vllm_lora_prefix (e.g. Llama).
    _fix_lora_keys_for_vllm(adapter_save_dir, trainer.model)
    print(f"DPO adapter saved to {adapter_save_dir}")

    best_ckpt_selection = None
    if args.select_best_ood_checkpoint:
        best_ckpt_selection = _pick_best_checkpoint_by_ood(args, tokenizer, trainer, output_dir)
        if best_ckpt_selection is None:
            print("[best-checkpoint] No valid checkpoint selection found; keeping final adapter.")

    # Clean up translated adapter temp dirs
    if _tmp_dirs:
        import shutil
        for tmp in _tmp_dirs:
            shutil.rmtree(tmp, ignore_errors=True)
        print(f"Cleaned up {len(_tmp_dirs)} temporary translated adapter dir(s).")

    # Save config summary
    reproducibility = {
        "seed": args.seed,
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "tokenizers_parallelism": os.environ.get("TOKENIZERS_PARALLELISM"),
        "cuda_launch_blocking": os.environ.get("CUDA_LAUNCH_BLOCKING"),
        "git_commit": _git_commit(),
        "hostname": platform.node(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "transformers_version": _safe_version("transformers"),
        "trl_version": _safe_version("trl"),
        "peft_version": _safe_version("peft"),
        "datasets_version": _safe_version("datasets"),
        "wandb_version": _safe_version("wandb"),
        "vllm_version": _safe_version("vllm"),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
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
        "save_strategy":   args.save_strategy,
        "save_steps":      args.save_steps if args.save_strategy == "steps" else None,
        "save_total_limit": args.save_total_limit,
        "select_best_ood_checkpoint": args.select_best_ood_checkpoint,
        "best_checkpoint_selection": best_ckpt_selection,
        "reproducibility": reproducibility,
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
    p.add_argument("--save-strategy", type=str, default="no",
                   choices=["no", "epoch", "steps"],
                   help="Checkpoint saving strategy during DPO training.")
    p.add_argument("--save-steps",    type=int, default=200,
                   help="Save interval when --save-strategy=steps.")
    p.add_argument("--save-total-limit", type=int, default=2,
                   help="Maximum number of checkpoints to keep.")
    p.add_argument("--select-best-ood-checkpoint", action="store_true",
                   help="After training, evaluate saved checkpoints on OOD "
                        "(toxic_chat + aegis) and replace final adapter with "
                        "the best checkpoint by average F1.")
    p.add_argument("--skip-eval",     action="store_true",
                   help="Skip vLLM evaluation after training.")
    p.add_argument("--wandb-project", type=str,  default="intention-jailbreak",
                   help="Weights & Biases project name.")
    p.add_argument("--wandb-run",     type=str,  default=None,
                   help="W&B run name (auto-generated if not set).")

    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
