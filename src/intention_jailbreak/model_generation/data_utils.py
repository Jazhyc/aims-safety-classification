from typing import Tuple
import matplotlib.pyplot as plt


def get_lengths(dataset, plot: bool = False) -> Tuple[int, int]:
    prompt_lengths = [len(item["prompt"].split()) for item in dataset]
    intent_lengths = [len(item["intent"].split()) for item in dataset]
    prompt_max = max(prompt_lengths)
    intent_max = max(intent_lengths)
    print("Maximum prompt length:", prompt_max)
    print("Maximum intent length:", intent_max)

    if plot:
        plt.hist(prompt_lengths, bins=30, alpha=0.5, label="Prompt Lengths")
        plt.hist(intent_lengths, bins=30, alpha=0.5, label="Intent Lengths")
        plt.legend(loc="upper right")
        plt.show()

    return prompt_max, intent_max


def train_val_test_split(dataset, config):
    data_cfg = config.get("data", {})
    test_size = data_cfg.get("test_size", 0.1)
    val_size = data_cfg.get("val_size", 0.1)
    seed = data_cfg.get("seed", 22)

    train_test_split = dataset.train_test_split(test_size=test_size, seed=seed)
    test_dataset = train_test_split["test"]
    true_val_size = val_size / (1 - test_size)
    val_dataset_split = train_test_split["train"].train_test_split(
        test_size=true_val_size, seed=seed
    )
    train_dataset = val_dataset_split["train"]
    val_dataset = val_dataset_split["test"]
    return train_dataset, val_dataset, test_dataset
