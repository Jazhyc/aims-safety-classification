"""
Training script for model generation (Seq2Seq or Causal LM).

Usage:
    python scripts/train_generator.py
    python scripts/train_generator.py --config-name=t5-large
    python scripts/train_generator.py training.batch_size=16
"""

import logging
import os
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["USE_TF"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

for _name in ("httpx", "httpcore", "huggingface_hub", "filelock", "urllib3"):
    logging.getLogger(_name).setLevel(logging.WARNING)

import hydra
from omegaconf import DictConfig, OmegaConf
from transformers import AutoConfig
from intention_jailbreak.model_generation.seq2seq import run_seq2seq_flow
from intention_jailbreak.model_generation.causal import run_causal_flow
from intention_jailbreak.model_generation.bert_harm import run_bert_flow
from intention_jailbreak.training import set_all_seeds, print_gpu_info
from dotenv import load_dotenv

load_dotenv()

try:
    import wandb
except ImportError:
    wandb = None

@hydra.main(version_base=None, config_path="../../configs/experiments", config_name="llm_sweep")
def main(cfg: DictConfig):
    # Convert Hydra config to dict for compatibility with existing flows
    config = OmegaConf.to_container(cfg, resolve=True)
    
    # Set seeds if provided in config, otherwise default to 42 or similar
    seed = config.get("data", {}).get("seed", 42)
    set_all_seeds(seed)
    print_gpu_info()
    
    # Initialize wandb if configured
    wandb_cfg = config.get("wandb", {})
    if wandb_cfg.get("enabled", False) and wandb is not None:
        wandb.init(
            entity=wandb_cfg.get("entity"),
            project=wandb_cfg.get("project"),
            dir=str(Path(hydra.utils.get_original_cwd()) / wandb_cfg.get("dir", "logs/wandb")),
            name=wandb_cfg.get("run_name"),
            tags=wandb_cfg.get("tags", []),
            mode=wandb_cfg.get("mode", "online"),
            config=config,
        )

    model_name = config["model"]["name"]
    print(f"Model: {model_name}")

    # Determine model type
    task_type = config.get("task", {}).get("type")

    if task_type == "bert_harm":
        print("Running BERT-style harm classifier training.")
        run_bert_flow(config)
    else:
        hf_config = AutoConfig.from_pretrained(
            model_name,
            trust_remote_code=config.get("model", {}).get("trust_remote_code", False),
        )
        is_seq2seq = bool(getattr(hf_config, "is_encoder_decoder", False))

        print(f"is_encoder_decoder = {is_seq2seq}")

        if is_seq2seq:
            print("Running T5-style Seq2Seq training.")
            run_seq2seq_flow(config)
        else:
            print("Running causal LM training (LLaMA/Qwen).")
            run_causal_flow(config)


if __name__ == "__main__":
    main()
