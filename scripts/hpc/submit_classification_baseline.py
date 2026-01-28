#!/usr/bin/env python3
"""
Submit a single job to train the classification-only model with baseline LoRA.

This trains: prompt -> "harmful" or "safe"
Using: rank=16, alpha=32 (same as your generation model)

This is the minimal experiment to get the classification model working.
For the full LoRA sweep, use submit_lora_sweep_classification.py instead.

Usage:
    python scripts/hpc/submit_classification_baseline.py
    python scripts/hpc/submit_classification_baseline.py --dry-run
"""

import argparse
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Submit classification baseline training job")
    parser.add_argument("--dry-run", action="store_true", help="Print without submitting")
    args = parser.parse_args()
    
    # Create logs directory
    Path("logs/slurm").mkdir(parents=True, exist_ok=True)
    
    script = """#!/bin/bash
#SBATCH --job-name=class-r8-a16
#SBATCH --time=01:00:00
#SBATCH --mem=32GB
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=a100:1
#SBATCH --output=logs/slurm/%x-%j.out

module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.8.0

source .venv/bin/activate

# Train classification-only model with baseline LoRA (rank=8, alpha=16)
python scripts/train_generator.py \\
    --config-name=llama3.1-8b-classification-base \\
    peft.lora_rank=16 \\
    peft.lora_alpha=32 \\
    paths.output_dir="train_results/causal/classification-r16-a32" \\
    paths.logs_dir="logs/causal/classification-r16-a32" \\
    paths.model_save_dir="trained_models/causal/classification-r16-a32" \\
    paths.predictions_dir="data/predictions/causal/classification-r16-a32" \\
    wandb.name="classification-r16-a32"
"""
    
    print("=" * 60)
    print("Classification-Only Baseline Training")
    print("=" * 60)
    print("\nModel: meta-llama/Llama-3.1-8B-Instruct")
    print("Format: prompt -> 'harmful' or 'safe'")
    print("LoRA: rank=16, alpha=32")
    print("\nHyperparameters:")
    print("  - learning_rate: 5e-5")
    print("  - epochs: 5")
    print("  - adam_beta1: 0.9, adam_beta2: 0.98")
    
    if args.dry_run:
        print("\n" + "=" * 60)
        print("DRY RUN - Would submit:")
        print("=" * 60)
        print(script)
        return
    
    # Write temp script and submit
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
        f.write(script)
        temp_path = f.name
    
    try:
        result = subprocess.run(['sbatch', temp_path], capture_output=True, text=True, check=True)
        job_id = result.stdout.strip().split()[-1]
        print(f"\n✓ Submitted: Job ID {job_id}")
        print(f"\nOutput will be saved to:")
        print(f"  trained_models/causal/classification-r8-a16_adapter/")
        print(f"\nMonitor with: squeue -j {job_id}")
        print(f"View logs: tail -f logs/slurm/class-baseline-{job_id}.out")
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Failed to submit: {e.stderr}")
    finally:
        Path(temp_path).unlink()


if __name__ == "__main__":
    main()