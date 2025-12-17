import os
import json
import torch
from tqdm import tqdm

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from trl import SFTTrainer, SFTConfig

from .preprocessing import preprocess_data
from .data_utils import train_val_test_split, align_tokenizer_with_model
from .evaluate_generations import compute_and_log_metrics

try:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
except ImportError:
    LoraConfig = None
    get_peft_model = None
    prepare_model_for_kbit_training = None

def format_prompt(prompt_text, predict_harm=False):
    """
    Format the input prompt for the model.
    
    Args:
        prompt_text: The prompt text (raw text from dataset)
        predict_harm: Whether to include harm prediction task
    
    Returns:
        Formatted prompt string
    """
    return f"{prompt_text}\n"

def format_completion(intent, harm=None, predict_harm=False):
    """
    Format the completion/target for the model.
    
    Args:
        intent: The intent text
        harm: The harm category (optional)
        predict_harm: Whether to include harm in completion
    
    Returns:
        Formatted completion string
    """
    if predict_harm and harm:
        return f"Intent: {intent}; Harm: {harm}"
    else:
        return intent

def save_preds_causal(model_name, model, tokenizer, eval_dataset, split_name, config, max_length=256):
    paths_cfg = config.get("paths", {})
    gen_cfg = config.get("generation", {})
    data_cfg = config.get("data", {})
    predict_harm = data_cfg.get("predict_harm", False)

    # Use different directory if predict_harm is enabled
    if predict_harm:
        base_pred_dir = paths_cfg.get("predictions_dir", "predictions_causal")
        base_pred_dir = base_pred_dir.replace("predictions", "predictions_with_harm")
    else:
        base_pred_dir = paths_cfg.get("predictions_dir", "predictions_causal")
    
    os.makedirs(base_pred_dir, exist_ok=True)

    filename = f"{model_name.replace('/', '_')}_{split_name}.jsonl"
    full_path = os.path.join(base_pred_dir, filename)

    # Generation settings
    gen_max_new_tokens = int(gen_cfg.get("max_new_tokens", 32))
    num_beams = int(gen_cfg.get("num_beams", 1))
    do_sample = bool(gen_cfg.get("do_sample", False))
    batch_size = int(gen_cfg.get("batch_size", 4))
    
    # Sampling parameters (only used when do_sample=True)
    temperature = float(gen_cfg.get("temperature", 1.0))
    top_p = float(gen_cfg.get("top_p", 1.0))
    top_k = int(gen_cfg.get("top_k", 50))

    device = next(model.parameters()).device
    model.eval()

    # Prepare batches
    examples = list(eval_dataset)
    batches = [examples[i:i + batch_size] for i in range(0, len(examples), batch_size)]
    
    # Collect all predictions and references for metric computation
    all_preds = []
    all_refs = []

    with open(full_path, "w", encoding="utf-8") as f:
        for batch in tqdm(batches, desc=f"Generating {split_name} predictions"):
            # Build prompts using format_prompt function
            prompt_texts = [ex['prompt'] for ex in batch]
            
            inputs = tokenizer(
                prompt_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(device)

            # Build generation kwargs
            gen_kwargs = {
                "max_new_tokens": gen_max_new_tokens,
                "do_sample": do_sample,
                "num_beams": num_beams,
                "pad_token_id": tokenizer.pad_token_id,
                "return_dict_in_generate": True,
                "output_scores": False,
            }
            
            # Only add sampling parameters when do_sample=True
            if do_sample:
                gen_kwargs.update({
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                })

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    **gen_kwargs,
                )

            # outputs.sequences contains the full output (input + generated)
            # Slice off the input portion to get only generated tokens
            input_length = inputs["input_ids"].shape[1]
            generated_ids = outputs.sequences[:, input_length:]
            
            # Batch decode all generated sequences at once
            generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            # Process each sample in the batch
            for ex, generated_intent in zip(batch, generated_texts):
                # Format ground truth with harm if predict_harm is enabled
                harm = ex.get("Annotator Harm") if predict_harm else None
                true_intent_formatted = format_completion(ex["intent"], harm, predict_harm)
                
                all_preds.append(generated_intent.strip())
                all_refs.append(true_intent_formatted)
                
                json_line = {
                    "id": ex["id"],
                    "prompt": ex["prompt"],
                    "true_intent": true_intent_formatted,
                    "generated_intent": generated_intent.strip(),
                }
                f.write(json.dumps(json_line, ensure_ascii=False) + "\n")

    print(f"Saved {split_name} predictions to {full_path}")
    return all_refs, all_preds


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
    tokenizer.padding_side = 'left'  # Required for decoder-only models during generation

    if not use_lora:
        # Get attention implementation from config
        attn_implementation = model_cfg.get("attn_implementation", "flash_attention_2")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            attn_implementation=attn_implementation,
            dtype=torch.bfloat16
        )
        model.config.use_cache = False  # Required for gradient checkpointing
        align_tokenizer_with_model(tokenizer, model)
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
    attn_implementation = model_cfg.get("attn_implementation", "flash_attention_2")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=trust_remote_code,
        attn_implementation=attn_implementation,
        dtype=torch.bfloat16,
    )

    # Required for gradient checkpointing
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

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
    align_tokenizer_with_model(tokenizer, model)

    # Optional: print how many parameters are trainable
    model.print_trainable_parameters()

    return tokenizer, model, True


def prepare_training_arguments(config, is_peft=False, num_train_samples=None):
    """
    Extract and prepare training arguments from config.
    
    Args:
        config: Full config dictionary
        is_peft: Whether using PEFT/LoRA (affects default optimizer)
        num_train_samples: Number of training samples (for eval step calculation)
    
    Returns:
        Dictionary of SFTConfig parameters
    """
    train_cfg = config.get("training", {})
    paths_cfg = config.get("paths", {})
    model_cfg = config.get("model", {})
    
    model_name = model_cfg["name"]
    out_name = model_name.replace("/", "_")
    
    # Default paths
    output_dir = paths_cfg.get("output_dir", f"./train_results/causal/{out_name}")
    logs_dir = paths_cfg.get("logs_dir", f"./logs/causal/{out_name}")
    
    # Training hyperparameters
    epochs = train_cfg.get("epochs", 8)
    lr = train_cfg.get("learning_rate", 5e-5)
    batch_size = train_cfg.get("batch_size", 8)
    weight_decay = train_cfg.get("weight_decay", 0.01)
    grad_accum = train_cfg.get("gradient_accumulation", 1)
    
    # Mixed precision - only one can be enabled
    use_fp16 = train_cfg.get("fp16", False) and torch.cuda.is_available()
    use_bf16 = train_cfg.get("bf16", False) and torch.cuda.is_available()
    
    # Prefer bf16
    if use_fp16 and use_bf16:
        use_fp16 = False
    
    # Optimizer
    default_optim = "paged_adamw_8bit" if is_peft else "adamw_torch"
    optim_name = train_cfg.get("optim", default_optim)
    
    # Torch compile for speedup
    torch_compile = train_cfg.get("torch_compile", True)
    
    # Gradient checkpointing for memory efficiency
    gradient_checkpointing = train_cfg.get("gradient_checkpointing", True)
    
    # SFT-specific parameters
    padding_free = train_cfg.get("padding_free", False)
    
    # Max sequence length for use with padding_free
    max_length = model_cfg.get("max_length_causal", 512)
    
    # WandB configuration
    wandb_cfg = config.get("wandb", {})
    report_to = "wandb" if wandb_cfg else "none"
    run_name = wandb_cfg.get("run_name", None)
    
    return {
        "output_dir": output_dir,
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
        "learning_rate": lr,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "gradient_accumulation_steps": grad_accum,
        "num_train_epochs": epochs,
        "weight_decay": weight_decay,
        "save_total_limit": 2,
        "logging_dir": logs_dir,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "fp16": use_fp16,
        "bf16": use_bf16,
        "optim": optim_name,
        "torch_compile": torch_compile,
        "gradient_checkpointing": gradient_checkpointing,
        "report_to": report_to,
        "run_name": run_name,
        "padding_free": padding_free,
        "completion_only_loss": True,
        "remove_unused_columns": False,
        "max_length": max_length,
    }


def run_causal_flow(config):
    model_cfg = config.get("model", {})
    paths_cfg = config.get("paths", {})
    model_name = model_cfg["name"]
    max_length = model_cfg.get("max_length_causal", 256)

    print("Loading dataset with preprocess_data()...")
    raw_dataset = preprocess_data()
    print("Dataset size:", len(raw_dataset))

    # Setup model and tokenizer (full FT or QLoRA)
    tokenizer, model, is_peft = setup_causal_model_and_tokenizer(config)
    
    # Get predict_harm flag from config
    data_cfg = config.get("data", {})
    predict_harm = data_cfg.get("predict_harm", False)

    # Create prompt and completion columns for SFTTrainer with completion_only_loss
    def create_prompt_completion(examples):
        # For completion_only_loss, we need separate prompt and completion columns
        # The model will only calculate loss on the completion tokens
        prompts = [format_prompt(p, predict_harm) for p in examples["prompt"]]
        
        # Get harm values if predict_harm is enabled
        if predict_harm:
            harms = examples.get("Annotator Harm", [None] * len(examples["prompt"]))
            completions = [format_completion(i, h, predict_harm) 
                          for i, h in zip(examples["intent"], harms)]
        else:
            completions = [format_completion(i, None, predict_harm) for i in examples["intent"]]
        
        return {
            "prompt": prompts,
            "completion": completions,
        }
    
    dataset_with_cols = raw_dataset.map(
        create_prompt_completion,
        batched=True,
    )
    
    train_dataset, val_dataset, test_dataset = train_val_test_split(
        dataset_with_cols, config
    )

    # Put non-PEFT model on a single device
    if not is_peft:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

    # Prepare training arguments from config
    training_args = SFTConfig(**prepare_training_arguments(config, is_peft, len(train_dataset)))

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
    )

    trainer.train()

    model_save_dir = paths_cfg.get(
        "model_save_dir",
        f"./trained_models/causal/{model_name}-model",
    )
    os.makedirs(model_save_dir, exist_ok=True)
    trainer.save_model(model_save_dir)
    tokenizer.save_pretrained(model_save_dir)
    print(f"Causal LM model and tokenizer saved to {model_save_dir}")

    eval_results = trainer.evaluate()
    print(f"[Causal LM] Validation loss: {eval_results['eval_loss']}")

    # Generate predictions
    val_refs, val_preds = save_preds_causal(
        model_name, model, tokenizer, val_dataset, "val", config,
        max_length=max_length,
    )
    test_refs, test_preds = save_preds_causal(
        model_name, model, tokenizer, test_dataset, "test", config,
        max_length=max_length,
    )
    
    # Compute and log metrics
    compute_and_log_metrics(val_refs, val_preds, split_name="val")
