#!/usr/bin/env python3
"""
SLURM Orchestrator for LoRA Rank/Alpha Sweep on Classification-Only Model

This script submits multiple SLURM jobs to sweep over different LoRA configurations
for the classification-only model (prompt -> harm label).

Uses the best hyperparameters from your previous sweep:
- learning_rate: 5e-5
- epochs: 5
- adam_beta1: 0.9
- adam_beta2: 0.98

Sweeps over:
- LoRA rank: 8, 16, 32, 64
- LoRA alpha: 2*rank (standard ratio)

Usage:
    python scripts/hpc/submit_lora_sweep_classification.py
    python scripts/hpc/submit_lora_sweep_classification.py --dry-run
"""

import argparse
import subprocess
from pathlib import Path


# LoRA configurations to sweep
# Format: (rank, alpha) - alpha is typically 2*rank
LORA_CONFIGS = [
    (8, 16),    # Smallest
    (16, 32),   # Baseline (what you already used for generation)
    (32, 64),   # 2x (what your teammate specifically asked about)
    (64, 128),  # 4x (to see the trend)
]

# Base config for classification-only model
CONFIG_NAME = "llama3.1-8b-classification-base"


def create_logs_dir():
    """Ensure the logs directory exists for SLURM output."""
    logs_dir = Path("logs/slurm")
    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir


def submit_sbatch(template_path: str, export_vars: dict, dry_run: bool = False) -> str:
    """
    Submit a SLURM job using sbatch.
    
    Args:
        template_path: Path to the SLURM script template
        export_vars: Dictionary of environment variables to export
        dry_run: If True, just print what would be submitted
    
    Returns:
        Job ID as string (or "DRY_RUN" if dry_run=True)
    """
    export_str = ','.join(f'{k}={v}' for k, v in export_vars.items())
    
    if dry_run:
        print(f"  Would submit: sbatch --export={export_str} {template_path}")
        return "DRY_RUN"
    
    result = subprocess.run(
        ['sbatch', f'--export={export_str}', template_path],
        capture_output=True,
        text=True,
        check=True
    )
    
    job_id = result.stdout.strip().split()[-1]
    return job_id


def main():
    parser = argparse.ArgumentParser(
        description="Submit LoRA sweep jobs for classification-only model"
    )
    parser.add_argument("--dry-run", action="store_true", 
                        help="Print what would be submitted without actually submitting")
    parser.add_argument("--config", type=int, nargs=2, metavar=('RANK', 'ALPHA'),
                        help="Run only a specific (rank, alpha) configuration")
    args = parser.parse_args()
    
    print("=" * 70)
    print("LoRA Rank/Alpha Sweep - Classification-Only Model")
    print("=" * 70)
    print(f"\nBase config: {CONFIG_NAME}")
    print(f"Model: meta-llama/Llama-3.1-8B-Instruct")
    print(f"Training: prompt -> harm label ('harmful' or 'safe')")
    print(f"\nHyperparameters (from best previous run):")
    print(f"  - learning_rate: 5e-5")
    print(f"  - epochs: 5")
    print(f"  - adam_beta1: 0.9, adam_beta2: 0.98")
    
    # Filter configs if specific one requested
    configs = LORA_CONFIGS
    if args.config:
        configs = [tuple(args.config)]
        print(f"\nRunning only: rank={args.config[0]}, alpha={args.config[1]}")
    
    print(f"\nLoRA configurations to sweep:")
    for rank, alpha in configs:
        print(f"  - rank={rank}, alpha={alpha}")
    
    print(f"\nSubmitting {len(configs)} job(s)...\n")
    
    # Create logs directory
    logs_dir = create_logs_dir()
    print(f"✓ Logs directory: {logs_dir}")
    
    # Submit jobs
    template_path = "scripts/hpc/lora_sweep_template.sh"
    job_ids = []
    
    for rank, alpha in configs:
        experiment_name = f"classification-r{rank}-a{alpha}"
        
        export_vars = {
            'LORA_RANK': rank,
            'LORA_ALPHA': alpha,
            'CONFIG_NAME': CONFIG_NAME,
            'EXPERIMENT_NAME': experiment_name,
        }
        
        try:
            job_id = submit_sbatch(template_path, export_vars, args.dry_run)
            job_ids.append((rank, alpha, experiment_name, job_id))
            
            if not args.dry_run:
                print(f"✓ Submitted rank={rank}, alpha={alpha}: Job ID {job_id}")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed rank={rank}, alpha={alpha}: {e.stderr}")
        except FileNotFoundError:
            print(f"✗ Template not found: {template_path}")
            print(f"  Make sure to copy lora_sweep_template.sh to scripts/hpc/")
            break
        except Exception as e:
            print(f"✗ Error: {str(e)}")
    
    # Summary
    print("\n" + "=" * 70)
    if args.dry_run:
        print("DRY RUN COMPLETE - No jobs were submitted")
    else:
        print(f"Successfully submitted {len(job_ids)}/{len(configs)} jobs")
    print("=" * 70)
    
    if job_ids and not args.dry_run:
        print("\nJob Summary:")
        print(f"{'Rank':<8} {'Alpha':<8} {'Experiment Name':<30} {'Job ID':<10}")
        print("-" * 60)
        for rank, alpha, name, job_id in job_ids:
            print(f"{rank:<8} {alpha:<8} {name:<30} {job_id:<10}")
        
        print("\nExpected outputs:")
        for rank, alpha, name, _ in job_ids:
            print(f"  models/sft/{name}_adapter/")
        
        print("\nMonitor jobs:")
        print("  squeue -u $USER")
        
        print("\nCancel all jobs:")
        all_ids = ' '.join(str(item[3]) for item in job_ids)
        print(f"  scancel {all_ids}")


if __name__ == "__main__":
    main()