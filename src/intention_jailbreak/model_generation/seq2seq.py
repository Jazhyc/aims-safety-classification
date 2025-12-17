import os
import json
import numpy as np
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from .preprocessing import preprocess_data
from .data_utils import get_lengths, train_val_test_split


def format_input_output_seq2seq(examples, tokenizer, prompt_max=512, intent_max=32):
    """
    Formatting used for T5 / encoder-decoder models:
    - Input: prompt tokens, padded to prompt_max
    - Labels: intent tokens, padded to intent_max
    """
    inputs = examples["prompt"]
    targets = examples["intent"]

    model_inputs = tokenizer(
        inputs,
        max_length=prompt_max,
        truncation=True,
        padding="max_length",
    )
    labels = tokenizer(
        targets,
        max_length=intent_max,
        truncation=True,
        padding="max_length",
    )

    label_ids = labels["input_ids"]

    # This block masks padding tokens (-100) so they are ignored in the loss, leading to
    # higher loss. Leaving it disabled includes pads in the loss, lowering loss artificially. Since the moel just needs to predict the pad token
    # In their version they did it without masking pads, so I just did it like they did but we might want to discuss later.

    #label_ids = [
    #    [(lid if lid != tokenizer.pad_token_id else -100) for lid in seq]
    #    for seq in label_ids
    #]
    #model_inputs["labels"] = label_ids

    model_inputs["labels"] = label_ids
    model_inputs["id"] = examples["id"]

    return model_inputs


def get_seq2seq_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer


def get_seq2seq_model(model_name):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return model


def t5_trainer(
    model,
    tokenizer,
    train_dataset,
    val_dataset,
    intent_max,
    config,
):
    model_name = config["model"]["name"]
    train_cfg = config.get("training", {})
    paths_cfg = config.get("paths", {})

    epochs = int(train_cfg.get("epochs", 8))
    lr = float(train_cfg.get("learning_rate", 5e-5))
    batch_size = int(train_cfg.get("batch_size", 8))
    weight_decay = float(train_cfg.get("weight_decay", 0.01))
    grad_accum = int(train_cfg.get("gradient_accumulation", 1))
    use_fp16 = bool(train_cfg.get("fp16", True)) and torch.cuda.is_available()

    output_dir = paths_cfg.get("output_dir", f"./train_results/seq2seq/{model_name.replace('/', '_')}")
    logs_dir = paths_cfg.get("logs_dir", f"./logs/seq2seq/{model_name.replace('/', '_')}")

    training_args = Seq2SeqTrainingArguments(
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
        predict_with_generate=True,
        logging_dir=logs_dir,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        generation_max_length=intent_max,
        report_to="none",
        fp16=use_fp16,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
    )

    return trainer


def save_preds_seq2seq(model_name, predictions, eval_dataset, tokenizer, split_name, config):
    paths_cfg = config.get("paths", {})
    base_pred_dir = paths_cfg.get("predictions_dir", "predictions")
    os.makedirs(base_pred_dir, exist_ok=True)

    filename = f"{model_name.replace('/', '_')}_{split_name}.jsonl"
    full_path = os.path.join(base_pred_dir, filename)

    with open(full_path, "w", encoding="utf-8") as f:
        for i, pred in enumerate(predictions.predictions):
            original_id = eval_dataset[i]["id"]
            pred = np.where(pred != -100, pred, tokenizer.pad_token_id)
            decoded_pred = tokenizer.decode(pred, skip_special_tokens=True)
            true_intent = eval_dataset[i]["intent"]
            json_line = {
                "id": original_id,
                "prediction": decoded_pred,
                "true_intent": true_intent,
            }
            f.write(json.dumps(json_line, ensure_ascii=False) + "\n")

    print(f"Saved {split_name} predictions to {full_path}")


def run_seq2seq_flow(config):
    model_name = config["model"]["name"]
    max_prompt_cfg = config["model"].get("max_length_prompt", 256)
    max_intent_cfg = config["model"].get("max_length_intent", 64)
    data_cfg = config.get("data", {})
    paths_cfg = config.get("paths", {})

    final_dataset = preprocess_data()

    prompt_max_raw, intent_max_raw = get_lengths(
        final_dataset,
        plot=data_cfg.get("plot_lengths", False),
    )

    prompt_max = min(prompt_max_raw, max_prompt_cfg)
    intent_max = min(intent_max_raw, max_intent_cfg)

    print(f"Using prompt_max={prompt_max} (raw={prompt_max_raw}), "
          f"intent_max={intent_max} (raw={intent_max_raw})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_seq2seq_model(model_name=model_name).to(device)
    tokenizer = get_seq2seq_tokenizer(model_name=model_name)

    tokenized_dataset = final_dataset.map(
        format_input_output_seq2seq,
        fn_kwargs={
            "tokenizer": tokenizer,
            "prompt_max": prompt_max,
            "intent_max": intent_max,
        },
        batched=True,
    )

    train_dataset, val_dataset, test_dataset = train_val_test_split(
        tokenized_dataset, config
    )

    trainer = t5_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        intent_max=intent_max,
        config=config,
    )

    trainer.train()

    model_save_dir = paths_cfg.get(
        "model_save_dir",
        os.path.join("trained_models/seq2seq/", f"{model_name}-model"),
    )
    os.makedirs(model_save_dir, exist_ok=True)
    trainer.save_model(model_save_dir)
    tokenizer.save_pretrained(model_save_dir)
    print(f"Seq2Seq model and tokenizer saved to {model_save_dir}")

    eval_results = trainer.evaluate(val_dataset)
    print(f"[Seq2Seq] Validation Loss: {eval_results['eval_loss']}")

    preds_val = trainer.predict(val_dataset)
    save_preds_seq2seq(model_name, preds_val, val_dataset, tokenizer, "val", config)
    preds_test = trainer.predict(test_dataset)
    save_preds_seq2seq(model_name, preds_test, test_dataset, tokenizer, "test", config)
