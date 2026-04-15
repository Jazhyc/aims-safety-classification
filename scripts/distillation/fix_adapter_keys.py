"""
Fix LoRA adapter safetensors keys for vLLM compatibility.

When training on VLMs (Gemma3, Mistral3, Qwen3.5) via the _load_causal_lm
extraction path, PEFT saves keys relative to the CausalLM wrapper:
    base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight

But vLLM serves *ForConditionalGeneration where the same module is at:
    language_model.model.layers.0.self_attn.q_proj

This script renames all adapter safetensors files in
models/distillation-sweep/ (excluding llama adapters which are fine)
to add the `language_model.` prefix where needed.

Usage:
    python scripts/distillation/fix_adapter_keys.py --dry-run
    python scripts/distillation/fix_adapter_keys.py
"""

import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SWEEP_DIR = PROJECT_ROOT / "models" / "distillation-sweep"

# These students were trained via the CausalLM extraction path and need fixing.
# Llama 3.1 8B is a pure CausalLM and is already correct.
NEEDS_FIX = {"gemma-3-12b", "ministral-3-14b-reasoning", "qwen3.5-9b"}


def needs_renaming(sf_path: Path) -> bool:
    """Return True if the first key still has the old base_model.model.model. prefix."""
    from safetensors import safe_open
    with safe_open(str(sf_path), framework="pt") as f:
        keys = list(f.keys())
    return any(k.startswith("base_model.model.model.") for k in keys)


def fix_adapter(sf_path: Path, dry_run: bool) -> bool:
    """Rename keys and overwrite the file. Returns True if changed."""
    from safetensors import safe_open
    from safetensors.torch import save_file

    tensors = {}
    changed = False
    with safe_open(str(sf_path), framework="pt") as f:
        for key in f.keys():
            if key.startswith("base_model.model.model."):
                new_key = "base_model.model.language_model.model." + key[len("base_model.model.model."):]
                changed = True
            else:
                new_key = key
            tensors[new_key] = f.get_tensor(key)

    if changed and not dry_run:
        save_file(tensors, str(sf_path))
    return changed


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", "-n", action="store_true", help="Print actions without modifying files.")
    args = parser.parse_args()

    if not SWEEP_DIR.exists():
        print(f"Directory not found: {SWEEP_DIR}")
        return

    fixed = skipped = already_ok = 0
    for adapter_dir in sorted(SWEEP_DIR.iterdir()):
        if not adapter_dir.is_dir():
            continue

        # Check if any known-affected student is in the directory name
        if not any(s in adapter_dir.name for s in NEEDS_FIX):
            skipped += 1
            continue

        sf_path = adapter_dir / "adapter_model.safetensors"
        if not sf_path.exists():
            print(f"  [missing] {adapter_dir.name}")
            continue

        if not needs_renaming(sf_path):
            already_ok += 1
            print(f"  [ok]      {adapter_dir.name}")
            continue

        if args.dry_run:
            print(f"  [fix]     {adapter_dir.name}")
        else:
            fix_adapter(sf_path, dry_run=False)
            print(f"  [fixed]   {adapter_dir.name}")
        fixed += 1

    print(f"\nDone. Fixed: {fixed}  Already correct: {already_ok}  Skipped (llama): {skipped}")
    if args.dry_run and fixed > 0:
        print("Run without --dry-run to apply changes.")


if __name__ == "__main__":
    main()
