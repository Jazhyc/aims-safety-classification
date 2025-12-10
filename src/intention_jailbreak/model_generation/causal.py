import os
import json
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)

from preprocessing import preprocess_data
from data_utils import train_val_test_split


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




def run_causal_flow(config):
    model_name = config["model"]["name"]
    model_cfg = config.get("model", {})
    train_cfg = config.get("training", {})
    paths_cfg = config.get("paths", {})
    data_cfg = config.get("data", {})
    max_length = model_cfg.get("max_length_causal", 256)

    print("Loading dataset with preprocess_data()...")
    raw_dataset = preprocess_data()
    print("Dataset size:", len(raw_dataset))

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

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
    use_fp16 = bool(train_cfg.get("fp16", True)) and torch.cuda.is_available()

    out_name = model_name.replace("/", "_")
    output_dir = paths_cfg.get("output_dir", f"./train_results/causal/{out_name}")
    logs_dir = paths_cfg.get("logs_dir", f"./logs/causal/{out_name}")
    model_save_dir = paths_cfg.get("model_save_dir", f"./trained_models/causal/{model_name}-model")

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

    save_preds_causal(model_name, model, tokenizer, val_dataset, "val", config, max_length=max_length)
    save_preds_causal(model_name, model, tokenizer, test_dataset, "test", config, max_length=max_length)
