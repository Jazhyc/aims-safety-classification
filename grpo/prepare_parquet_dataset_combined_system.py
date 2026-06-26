import argparse
import json
import os
import datasets


def map_harm_label(annotator_harm):
    if annotator_harm in ["Completely Harmful", "Uncertain Harmful"]:
        return "harmful"
    elif annotator_harm in ["Completely Safe", "Uncertain Safe"]:
        return "safe"
    elif annotator_harm is None:
        return None  # will be filtered out
    else:
        raise ValueError(f"Unknown label: {annotator_harm}")


def make_map_fn(split, system_prompt):
    def process_fn(example):
        prompt = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": example["Prompt"]}
        ]
        return {
            "data_source": "aims_safety_intents",
            "prompt": prompt,
            "reward_model": {
                "ground_truth": map_harm_label(example["Annotator Harm"])
            },
            "extra_info": {
                "id": example["ID"],
                "golden_intent": example["Intent"],
                "original_prompt": example["Prompt"],
                "adversarial": example["Adversarial"],
                "annotator_harm": example["Annotator Harm"],
                "dataset_harm": example["Dataset Harm"],
                "split": split,
            }
        }
    return process_fn


def process_split(dataset, split, system_prompt):
    none_ids = [ex["ID"] for ex in dataset[split] if ex["Annotator Harm"] is None]
    if none_ids:
        print(f"{split}: {len(none_ids)} samples with None labels, IDs: {none_ids}")

    processed = dataset[split].map(
        make_map_fn(split, system_prompt),
        remove_columns=dataset[split].column_names
    )

    before = len(processed)
    processed = processed.filter(lambda x: x["reward_model"]["ground_truth"] is not None)
    after = len(processed)
    if before != after:
        print(f"{split}: filtered out {before - after} samples with missing labels")

    return processed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--system_prompt", required=True, help="Path to system prompt .txt file")
    parser.add_argument("--output_dir", required=True, help="Directory to write train.parquet and validation.parquet")
    args = parser.parse_args()

    with open(args.system_prompt, "r", encoding="utf-8") as f:
        system_prompt = f.read().strip()

    os.makedirs(args.output_dir, exist_ok=True)

    dataset = datasets.load_dataset("Jazhyc/aims-safety-intents")

    # Train
    processed_train = process_split(dataset, "train", system_prompt)
    train_out = os.path.join(args.output_dir, "train.parquet")
    processed_train.to_parquet(train_out)
    print(f"train: {len(processed_train)} samples → {train_out}")

    # Validation + test combined
    print("\n=== Combining validation and test sets ===")
    processed_val = process_split(dataset, "validation", system_prompt)
    processed_test = process_split(dataset, "test", system_prompt)
    combined_validation = datasets.concatenate_datasets([processed_val, processed_test])

    before = len(combined_validation)
    combined_validation = combined_validation.filter(lambda x: x["reward_model"]["ground_truth"] is not None)
    after = len(combined_validation)
    if before != after:
        print(f"combined validation+test: filtered out {before - after} samples with missing labels")

    val_out = os.path.join(args.output_dir, "validation.parquet")
    combined_validation.to_parquet(val_out)
    print(f"combined validation+test: {len(combined_validation)} samples → {val_out}")

    val_count = sum(1 for x in combined_validation if x["extra_info"]["split"] == "validation")
    test_count = sum(1 for x in combined_validation if x["extra_info"]["split"] == "test")
    print(f"  - from validation: {val_count} samples")
    print(f"  - from test: {test_count} samples")

    # Sanity check
    train = datasets.Dataset.from_parquet(train_out)
    validation = datasets.Dataset.from_parquet(val_out)

    print("\n=== Sample (train) ===")
    print(json.dumps(train[0], indent=2))
    print("\n=== Sample (combined validation) ===")
    print(json.dumps(validation[0], indent=2))

    print("\n=== Label distribution ===")
    train_labels = [ex["reward_model"]["ground_truth"] for ex in train]
    print(f"Train - harmful: {train_labels.count('harmful')}, safe: {train_labels.count('safe')}")

    val_labels = [ex["reward_model"]["ground_truth"] for ex in validation]
    print(f"Combined validation - harmful: {val_labels.count('harmful')}, safe: {val_labels.count('safe')}")

    val_from_val = [ex for ex in validation if ex["extra_info"]["split"] == "validation"]
    val_from_test = [ex for ex in validation if ex["extra_info"]["split"] == "test"]
    val_val_labels = [ex["reward_model"]["ground_truth"] for ex in val_from_val]
    val_test_labels = [ex["reward_model"]["ground_truth"] for ex in val_from_test]
    print(f"  - From original validation: harmful: {val_val_labels.count('harmful')}, safe: {val_val_labels.count('safe')}")
    print(f"  - From original test: harmful: {val_test_labels.count('harmful')}, safe: {val_test_labels.count('safe')}")
