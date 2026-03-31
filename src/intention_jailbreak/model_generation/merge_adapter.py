"""
Merge a LoRA adapter into its base model, producing a standard dense model.

The merged model can be used directly in eval_safety_classifier.py by setting:
  model.name: <output_dir>
  finetuned.use_merged: true

This avoids all per-forward-pass LoRA overhead during evaluation.
"""

from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def merge_adapter(
    base_model: str,
    adapter_path: str,
    output_dir: str,
    dtype: str = "bfloat16",
) -> Path:
    """
    Merge a LoRA adapter into a base model and save the result.

    Args:
        base_model: HF model ID or local path of the base model.
        adapter_path: Path to the LoRA adapter directory.
        output_dir: Where to save the merged model and tokenizer.
        dtype: Torch dtype for loading ("bfloat16", "float16", "float32").

    Returns:
        Path to the saved merged model directory.
    """
    torch_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype]

    print(f"Base model : {base_model}")
    print(f"Adapter    : {adapter_path}")
    print(f"Output     : {output_dir}")
    print(f"Dtype      : {dtype}")

    print("\nLoading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch_dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, adapter_path, torch_dtype=torch_dtype)

    print("Merging weights...")
    model = model.merge_and_unload()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Saving merged model to {output_path} ...")
    model.save_pretrained(output_path)

    print("Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    tokenizer.save_pretrained(output_path)

    print("Done.")
    return output_path
