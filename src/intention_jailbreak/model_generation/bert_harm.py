import os
import json
import numpy as np
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support

from .preprocessing import preprocess_data
from .data_utils import train_val_test_split, apply_binary_harm_mapping


def build_label_mapping(dataset, label_key: str):
    """
    Build stable label<->id mapping from the dataset column.

    Returns:
        label2id (dict[str,int]), id2label (dict[int,str])
    """
    labels = []
    for ex in dataset:
        lbl = ex.get(label_key)
        if lbl is None:
            continue
        lbl = str(lbl).strip()
        if lbl == "":
            continue
        labels.append(lbl)

    unique = sorted(set(labels))
    if not unique:
        raise ValueError(
            f"No non-empty labels found in column '{label_key}'. "
            f"Available columns: {dataset.column_names}"
        )

    label2id = {l: i for i, l in enumerate(unique)}
    id2label = {i: l for l, i in label2id.items()}
    return label2id, id2label


def format_for_classification(
    examples,
    tokenizer,
    label_key,
    label2id,
    input_key="prompt",   # ← NEW: "prompt" or "intent"
    max_length=512,
):
    """
    Tokenize input text and add integer labels.

    input_key controls what text is classified:
      "prompt"  → original bert_harm behaviour, classifies prompt text
      "intent"  → intent-classifier mode, classifies generated/annotated intent text

    The label column (label_key) is always "Annotator Harm" mapped to harmful/safe.
    """
    texts = examples[input_key]
    toks = tokenizer(
        texts,
        max_length=max_length,
        truncation=True,
        padding="max_length",
    )

    raw_labels = examples.get(label_key, [None] * len(texts))
    int_labels = []
    for lbl in raw_labels:
        if lbl is None:
            int_labels.append(-100)  # ignored by loss
        else:
            lbl = str(lbl).strip()
            int_labels.append(label2id.get(lbl, -100))
    toks["labels"] = int_labels
    toks["id"] = examples["id"]
    return toks


def compute_metrics_fn(id2label):
    """
    Create a HF Trainer compute_metrics callable.
    Logs macro F1, accuracy, and per-class precision/recall/F1.
    """
    def _compute(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)

        mask = labels != -100
        labels_f = labels[mask]
        preds_f = preds[mask]

        if len(labels_f) == 0:
            return {"f1_macro": 0.0, "accuracy": 0.0}

        f1 = f1_score(labels_f, preds_f, average="macro")
        acc = accuracy_score(labels_f, preds_f)
        p, r, f1_per_class, _ = precision_recall_fscore_support(
            labels_f, preds_f, labels=list(id2label.keys()), zero_division=0
        )

        metrics = {"f1_macro": float(f1), "accuracy": float(acc)}
        for class_id, class_name in id2label.items():
            metrics[f"precision/{class_name}"] = float(p[class_id])
            metrics[f"recall/{class_name}"] = float(r[class_id])
            metrics[f"f1/{class_name}"] = float(f1_per_class[class_id])
        return metrics

    return _compute


def save_preds(model_name, trainer, eval_dataset, id2label, split_name, config, input_key="prompt"):
    """
    Save harm predictions to JSONL.

    Fields saved:
      { id, input_text, input_key, true_harm, pred_harm }

    input_key is stored in each row so the file is self-documenting —
    you can tell immediately whether it was trained on prompts or intents.
    """
    paths_cfg = config.get("paths", {})
    base_pred_dir = paths_cfg.get("predictions_dir", "predictions_harm")
    os.makedirs(base_pred_dir, exist_ok=True)

    filename = f"{model_name.replace('/', '_')}_{split_name}.jsonl"
    full_path = os.path.join(base_pred_dir, filename)

    preds = trainer.predict(eval_dataset)
    logits = preds.predictions
    pred_ids = np.argmax(logits, axis=-1)

    with open(full_path, "w", encoding="utf-8") as f:
        for i in range(len(eval_dataset)):
            ex = eval_dataset[i]
            true_id = int(ex["labels"])
            pred_id = int(pred_ids[i])

            obj = {
                "id": ex["id"],
                "input_key": input_key,
                "input_text": ex.get(input_key, ex.get("prompt", None)),
                "true_harm": None if true_id == -100 else id2label[true_id],
                "pred_harm": id2label[pred_id],
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"Saved {split_name} predictions ({input_key} -> harm) to {full_path}")
    return full_path


# Keep old name as alias so any existing call sites don't break
save_preds_harm = save_preds


def run_bert_flow(config):
    """
    Train an encoder model (BERT / ModernBERT) to predict harm.

    Controlled entirely by config — set data.input_key to switch modes:
      input_key: "prompt"   → original harm classifier (prompt -> harmful/safe)
      input_key: "intent"   → intent classifier       (intent -> harmful/safe)

    When input_key is "intent", apply_binary_harm_mapping is always applied
    since the intent classifier is always binary (harmful/safe).
    """
    model_cfg  = config.get("model", {})
    train_cfg  = config.get("training", {})
    data_cfg   = config.get("data", {})
    paths_cfg  = config.get("paths", {})
    wandb_cfg  = config.get("wandb", {})

    model_name  = model_cfg["name"]
    label_key   = data_cfg.get("harm_label_key", "Annotator Harm")
    max_length  = int(model_cfg.get("max_length_prompt", 512))

    # ── NEW: input_key controls whether we classify prompts or intents ──
    input_key   = data_cfg.get("input_key", "prompt")
    binary      = bool(data_cfg.get("binary_harm_mapping", False))

    mode_label  = "intent classifier" if input_key == "intent" else "prompt harm classifier"
    print(f"[BERT] Mode         : {mode_label}")
    print(f"[BERT] Input key    : {input_key}")
    print(f"[BERT] Binary labels: {binary}")

    # Load all splits from Hub
    def _load_split(split):
        ds = preprocess_data(split=split)
        if binary:
            ds = apply_binary_harm_mapping(ds, binary_harm_mapping=True)
        return ds

    train_raw = _load_split("train")
    val_raw   = _load_split("validation")
    test_raw  = _load_split("test")

    # Build label mapping from all splits combined so IDs are consistent
    from datasets import concatenate_datasets
    all_raw = concatenate_datasets([train_raw, val_raw, test_raw])
    label2id, id2label = build_label_mapping(all_raw, label_key)
    num_labels = len(label2id)
    print(f"[BERT] Label column : '{label_key}' — {num_labels} classes: {list(label2id.keys())}")

    # Tokenizer and model
    trust_remote_code = bool(model_cfg.get("trust_remote_code", False))
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        label2id=label2id,
        id2label={str(k): v for k, v in id2label.items()},
        trust_remote_code=trust_remote_code,
    )

    def _tokenize(ds):
        return ds.map(
            format_for_classification,
            fn_kwargs={
                "tokenizer": tokenizer,
                "label_key":  label_key,
                "label2id":   label2id,
                "input_key":  input_key,
                "max_length": max_length,
            },
            batched=True,
        )

    train_dataset = _tokenize(train_raw)
    val_dataset   = _tokenize(val_raw)
    test_dataset  = _tokenize(test_raw)

    # Hyperparameters
    epochs     = int(train_cfg.get("epochs", 5))
    lr         = float(train_cfg.get("learning_rate", 2e-5))
    batch_size = int(train_cfg.get("batch_size", 16))
    weight_decay = float(train_cfg.get("weight_decay", 0.01))
    grad_accum = int(train_cfg.get("gradient_accumulation", 1))

    use_fp16 = bool(train_cfg.get("fp16", False)) and torch.cuda.is_available()
    use_bf16 = bool(train_cfg.get("bf16", True))  and torch.cuda.is_available()
    if use_fp16 and use_bf16:
        use_fp16 = False

    out_name      = model_name.replace("/", "_")
    output_dir    = paths_cfg.get("output_dir",    f"data/train_results/bert_harm/{out_name}")
    logs_dir      = paths_cfg.get("logs_dir",      f"./logs/bert_harm/{out_name}")
    model_save_dir = paths_cfg.get("model_save_dir", f"models/bert_harm/{out_name}")

    report_to = "wandb" if wandb_cfg else "none"
    run_name  = wandb_cfg.get("run_name", None)

    if wandb_cfg.get("project"):
        os.environ.setdefault("WANDB_PROJECT", str(wandb_cfg["project"]))
    if wandb_cfg.get("entity"):
        os.environ.setdefault("WANDB_ENTITY", str(wandb_cfg["entity"]))

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
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        report_to=report_to,
        run_name=run_name,
        fp16=use_fp16,
        bf16=use_bf16,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        compute_metrics=compute_metrics_fn(id2label),
    )

    # Baseline eval before any training (sanity check)
    trainer.evaluate()
    trainer.train()

    os.makedirs(model_save_dir, exist_ok=True)
    trainer.save_model(model_save_dir)
    tokenizer.save_pretrained(model_save_dir)
    print(f"[BERT] Model saved to {model_save_dir}")

    # Save predictions — input_key is recorded in every row
    save_preds(model_name, trainer, val_dataset,  id2label, "val",  config, input_key)
    save_preds(model_name, trainer, test_dataset, id2label, "test", config, input_key)