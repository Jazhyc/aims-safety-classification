import argparse
from transformers import AutoConfig
from config_loader import load_config

from seq2seq import run_seq2seq_flow
from causal import run_causal_flow


def main(config_path: str):
    config = load_config(config_path)
    model_name = config["model"]["name"]

    print(f"Loaded config file: {config_path}")
    print(f"Model: {model_name}")

    hf_config = AutoConfig.from_pretrained(model_name)
    is_seq2seq = getattr(hf_config, "is_encoder_decoder", False)

    print(f"is_encoder_decoder = {is_seq2seq}")

    if is_seq2seq:
        print("Running T5-style Seq2Seq training.")
        run_seq2seq_flow(config)
    else:
        print("Running causal LM training (LLaMA/Qwen).")
        run_causal_flow(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified training script.")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML config file (e.g. config/t5-small.yaml)")
    args = parser.parse_args()

    main(config_path=args.config)
