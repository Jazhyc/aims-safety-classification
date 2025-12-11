import os
import json
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
)

from preprocessing import preprocess_data
from data_utils import train_val_test_split

try:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
except ImportError:
    LoraConfig = None
    get_peft_model = None
    prepare_model_for_kbit_training = None


def format_input_output_causal(examples, tokenizer, max_length=512):
    """
    Formatting for causal LMs (LLaMA, Qwen, etc.):

    We build:
      prompt_text  = "Prompt: <prompt>\\nIntent:"
      target_text  = " <intent>"
      full_text    = prompt_text + target_text

    We then:
      - tokenize full_text as input_ids,
      - mask the prompt part in labels with -100,
      - keep the intent part as labels, so loss is only on intent tokens.
    """
    prompts = examples["prompt"]
    intents = examples["intent"]

    prompt_texts = [f"Prompt: {p}\nIntent:" for p in prompts]
    target_texts = [f" {t}" for t in intents]
    full_texts = [pt + tt for pt, tt in zip(prompt_texts, target_texts)]

    prompt_enc = tokenizer(
        prompt_texts,
        max_length=max_length,
        truncation=True,
        add_special_tokens=True,
    )
    full_enc = tokenizer(
        full_texts,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        add_special_tokens=True,
    )

    input_ids = full_enc["input_ids"]
    labels = []

    for full_ids, prompt_ids in zip(input_ids, prompt_enc["input_ids"]):
        prompt_len = len(prompt_ids)
        label_ids = [
            (-100 if i < prompt_len else tok_id)
            for i, tok_id in enumerate(full_ids)
        ]
        labels.append(label_ids)

    full_enc["labels"] = labels
    full_enc["id"] = examples["id"]
    return full_enc


def save_preds_causal_old_one(model_name, model, tokenizer, eval_dataset, split_name, config, max_length=256):
    paths_cfg = config.get("paths", {})
    gen_cfg = config.get("generation", {})

    base_pred_dir = paths_cfg.get("predictions_dir", "predictions_causal")
    os.makedirs(base_pred_dir, exist_ok=True)

    filename = f"{model_name.replace('/', '_')}_{split_name}.jsonl"
    full_path = os.path.join(base_pred_dir, filename)

    # Generation settings
    gen_max_new_tokens = int(gen_cfg.get("max_new_tokens", 32))
    num_beams = int(gen_cfg.get("num_beams", 1))
    do_sample = bool(gen_cfg.get("do_sample", False))

    device = next(model.parameters()).device
    model.eval()

    with open(full_path, "w", encoding="utf-8") as f:
        for ex in eval_dataset:
            # Build the same prompt format as during training
            prompt_text = f"Prompt: {ex['prompt']}\nIntent:"
            inputs = tokenizer(
                prompt_text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            ).to(device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=gen_max_new_tokens,
                    do_sample=do_sample,
                    num_beams=num_beams,
                )

            decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

            generated_intent = decoded

            if generated_intent.startswith(prompt_text):
                generated_intent = generated_intent[len(prompt_text):]
            generated_intent = generated_intent.lstrip()
            if generated_intent.lower().startswith("intent:"):
                generated_intent = generated_intent[len("intent:"):].lstrip()

            json_line = {
                "id": ex["id"],
                "prompt": ex["prompt"],
                "true_intent": ex["intent"],
                "generated_intent": generated_intent,
            }
            f.write(json.dumps(json_line, ensure_ascii=False) + "\n")

    print(f"Saved {split_name} predictions to {full_path}")

def save_preds_causal(model_name, model, tokenizer, eval_dataset, split_name, config, max_length=256):
    paths_cfg = config.get("paths", {})
    gen_cfg = config.get("generation", {})

    base_pred_dir = paths_cfg.get("predictions_dir", "predictions_causal")
    os.makedirs(base_pred_dir, exist_ok=True)

    filename = f"{model_name.replace('/', '_')}_{split_name}.jsonl"
    full_path = os.path.join(base_pred_dir, filename)

    # Generation settings
    gen_max_new_tokens = int(gen_cfg.get("max_new_tokens", 32))
    num_beams = int(gen_cfg.get("num_beams", 1))
    do_sample = bool(gen_cfg.get("do_sample", False))

    device = next(model.parameters()).device
    model.eval()

    with open(full_path, "w", encoding="utf-8") as f:
        for ex in eval_dataset:
            # Build the same prompt format as during training
            prompt_text = f"Prompt: {ex['prompt']}\nIntent:"
            inputs = tokenizer(
                prompt_text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            ).to(device)

            input_ids = inputs["input_ids"]
            prompt_len = input_ids.shape[1]

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=gen_max_new_tokens,
                    do_sample=do_sample,
                    num_beams=num_beams,
                )

            # outputs[0] contains [prompt tokens + new tokens]
            full_output_ids = outputs[0]

            # Keep only the newly generated tokens (after the prompt)
            gen_token_ids = full_output_ids[prompt_len:]

            # Decode just the generation part
            generated_intent = tokenizer.decode(
                gen_token_ids,
                skip_special_tokens=True,
            ).lstrip()

            # If the model echoes "Intent:" at the start, strip it
            if generated_intent.lower().startswith("intent:"):
                generated_intent = generated_intent[len("intent:"):].lstrip()

            json_line = {
                "id": ex["id"],
                "prompt": ex["prompt"],
                "true_intent": ex["intent"],
                "generated_intent": generated_intent,
            }
            f.write(json.dumps(json_line, ensure_ascii=False) + "\n")

    print(f"Saved {split_name} predictions to {full_path}")


def setup_causal_model_and_tokenizer(config):
    """
    Set up tokenizer and model for causal LM training.

    - If config['peft']['use_lora'] is False or missing:
        → standard full fine-tuning (what you already did before).
    - If True:
        → load model in 4-bit and wrap with LoRA (QLoRA-style).

    Returns:
        tokenizer, model, is_peft
    """
    model_cfg = config.get("model", {})
    peft_cfg = config.get("peft", {})
    quant_cfg = config.get("quantization", {})
    model_name = model_cfg["name"]

    use_lora = bool(peft_cfg.get("use_lora", False))

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if not use_lora:
        # === Original full fine-tune path ===
        model = AutoModelForCausalLM.from_pretrained(model_name)
        return tokenizer, model, False

    # === QLoRA path ===
    if BitsAndBytesConfig is None or LoraConfig is None or get_peft_model is None:
        raise ImportError(
            "QLoRA mode requested but `bitsandbytes` and/or `peft` are not installed.\n"
            "Install them with: pip install bitsandbytes peft"
        )

    # 4-bit quantization config
    load_in_4bit = bool(quant_cfg.get("load_in_4bit", True))
    bnb_4bit_use_double_quant = bool(quant_cfg.get("bnb_4bit_use_double_quant", True))
    bnb_4bit_quant_type = quant_cfg.get("bnb_4bit_quant_type", "nf4")
    compute_dtype_str = quant_cfg.get("bnb_4bit_compute_dtype", "bfloat16")
    compute_dtype = getattr(torch, compute_dtype_str, torch.bfloat16)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    # Load model in 4-bit, let HF place it on devices automatically
    trust_remote_code = bool(model_cfg.get("trust_remote_code", False))
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=trust_remote_code,
    )

    # Recommended for QLoRA
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)

    # LoRA config (defaults can be overridden in config)
    lora_r = int(peft_cfg.get("r", 16))
    lora_alpha = int(peft_cfg.get("lora_alpha", 32))
    lora_dropout = float(peft_cfg.get("lora_dropout", 0.05))
    bias = peft_cfg.get("bias", "none")

    # Target modules: typical names for LLaMA/Qwen-style models
    default_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    target_modules = peft_cfg.get("target_modules", default_target_modules)

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias=bias,
        target_modules=target_modules,
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)

    # Optional: print how many parameters are trainable
    model.print_trainable_parameters()

    return tokenizer, model, True


def run_causal_flow(config):
    model_cfg = config.get("model", {})
    train_cfg = config.get("training", {})
    paths_cfg = config.get("paths", {})
    data_cfg = config.get("data", {})
    peft_cfg = config.get("peft", {})

    model_name = model_cfg["name"]
    max_length = model_cfg.get("max_length_causal", 256)

    print("Loading dataset with preprocess_data()...")
    raw_dataset = preprocess_data()
    print("Dataset size:", len(raw_dataset))

    # NEW: centralised model/tokenizer setup (full FT or QLoRA)
    tokenizer, model, is_peft = setup_causal_model_and_tokenizer(config)

    # Tokenize dataset
    tokenized = raw_dataset.map(
        format_input_output_causal,
        fn_kwargs={"tokenizer": tokenizer, "max_length": max_length},
        batched=True,
    )

    train_dataset, val_dataset, test_dataset = train_val_test_split(
        tokenized, config
    )

    epochs = int(train_cfg.get("epochs", 8))
    lr = float(train_cfg.get("learning_rate", 5e-5))
    batch_size = int(train_cfg.get("batch_size", 8))
    weight_decay = float(train_cfg.get("weight_decay", 0.01))
    grad_accum = int(train_cfg.get("gradient_accumulation", 1))

    # Mixed precision flags
    use_fp16 = bool(train_cfg.get("fp16", True)) and torch.cuda.is_available()
    use_bf16 = bool(train_cfg.get("bf16", False)) and torch.cuda.is_available()

    # Optimizer: use paged_adamw_8bit by default when doing QLoRA
    default_optim = "paged_adamw_8bit" if is_peft else "adamw_torch"
    optim_name = train_cfg.get("optim", default_optim)

    out_name = model_name.replace("/", "_")
    output_dir = paths_cfg.get("output_dir", f"./train_results/causal/{out_name}")
    logs_dir = paths_cfg.get("logs_dir", f"./logs/causal/{out_name}")
    model_save_dir = paths_cfg.get(
        "model_save_dir",
        f"./trained_models/causal/{model_name}-model",
    )

    # Put non-PEFT model on a single device explicitly.
    # For PEFT/QLoRA we already used device_map="auto".
    if not is_peft:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=epochs,
        weight_decay=weight_decay,
        save_total_limit=2,
        logging_dir=logs_dir,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        fp16=use_fp16,
        bf16=use_bf16,
        optim=optim_name,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()

    os.makedirs(model_save_dir, exist_ok=True)
    trainer.save_model(model_save_dir)
    tokenizer.save_pretrained(model_save_dir)
    print(f"Causal LM model and tokenizer saved to {model_save_dir}")

    eval_results = trainer.evaluate(val_dataset)
    print(f"[Causal LM] Validation loss: {eval_results['eval_loss']}")

    # NOTE: save_preds_causal works unchanged for both full FT and QLoRA
    save_preds_causal(
        model_name, model, tokenizer, val_dataset, "val", config,
        max_length=max_length,
    )
    save_preds_causal(
        model_name, model, tokenizer, test_dataset, "test", config,
        max_length=max_length,
    )
