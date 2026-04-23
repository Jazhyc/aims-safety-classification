"""
CLI entry point for merging a LoRA adapter into a base model.

See src/intention_jailbreak/model_generation/merge_adapter.py for the library function.

Usage:
    python scripts/dpo/merge_adapter.py \\
        --base-model meta-llama/Llama-3.1-8B-Instruct \\
        --adapter models/sft/lr_5e-05_e_5_adapter \\
        --output models/sft/lr_5e-05_e_5_merged

    python scripts/dpo/merge_adapter.py \\
        --base-model meta-llama/Llama-3.1-8B-Instruct \\
        --adapter models/distillation/reasoning-distillation-gemma-3-27b-without-intent_adapter \\
        --output models/distillation/reasoning-distillation-gemma-3-27b-without-intent_merged \\
        --dtype bfloat16

After merging, update your eval config:
    model:
      name: <output>
    finetuned:
      use_merged: true
"""

import argparse

from intention_jailbreak.model_generation.merge_adapter import merge_adapter

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge a LoRA adapter into a base model.")
    parser.add_argument("--base-model", required=True,
                        help="HF model ID or local path of the base model")
    parser.add_argument("--adapter", required=True,
                        help="Path to the LoRA adapter directory")
    parser.add_argument("--output", required=True,
                        help="Output directory for the merged model")
    parser.add_argument("--dtype", default="bfloat16",
                        choices=["bfloat16", "float16", "float32"],
                        help="Dtype to use when loading and saving (default: bfloat16)")
    args = parser.parse_args()

    merge_adapter(args.base_model, args.adapter, args.output, args.dtype)
