#!/usr/bin/env python3
"""
SLURM Orchestrator for Hyperparameter Sweep

This script submits multiple SLURM jobs in parallel to sweep over learning rates
for Llama 3.1 8B Instruct fine-tuning. Each job will train with a different
learning rate specified in LEARNING_RATES.

Usage:
    python scripts/hpc/submit_hyperparam_sweep.py
"""

import subprocess
from slurm_utils import create_logs_dir, submit_sbatch, print_header, print_job_summary


# Model configuration
MODEL_HF_PATH = "meta-llama/Llama-3.1-8B-Instruct"
MODEL_NAME = "llama3.1-8b"

# Learning rates to sweep over
LEARNING_RATES = [
    1e-5,
    2e-5,
    5e-5,
    1e-4,
    2e-4,
    5e-4,
]

# Adam optimizer hyperparameters
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.98


def submit_job(lr: float) -> str:
    """
    Submit a SLURM job for a specific learning rate.
    
    Args:
        lr: Learning rate value
    
    Returns:
        Job ID as string
    """
    template_path = "scripts/hpc/hyperparam_sweep_template.sh"
    export_vars = {
        'LEARNING_RATE': lr,
        'ADAM_BETA1': ADAM_BETA1,
        'ADAM_BETA2': ADAM_BETA2,
    }
    return submit_sbatch(template_path, export_vars)


def main():
    """Main orchestration function."""
    print_header(
        "Hyperparameter Sweep - SLURM Job Submission",
        f"Model: {MODEL_HF_PATH}"
    )
    print(f"\nSubmitting {len(LEARNING_RATES)} jobs...\n")
    
    # Create necessary directories
    logs_dir = create_logs_dir()
    print(f"✓ Created logs directory: {logs_dir}")
    
    # Submit all jobs
    job_ids = []
    for lr in LEARNING_RATES:
        try:
            job_id = submit_job(lr)
            job_ids.append((lr, job_id))
            print(f"✓ Submitted lr={lr:.0e}: Job ID {job_id}")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to submit lr={lr:.0e}: {e.stderr}")
        except Exception as e:
            print(f"✗ Error submitting lr={lr:.0e}: {str(e)}")
    
    # Summary
    print("\n" + "=" * 60)
    print(f"Successfully submitted {len(job_ids)}/{len(LEARNING_RATES)} jobs")
    print("=" * 60)
    
    if job_ids:
        print("\nJob Summary:")
        for lr, job_id in job_ids:
            print(f"  lr={lr:.0e} → Job {job_id}")
        
        print(f"\nAdam hyperparameters:")
        print(f"  beta1: {ADAM_BETA1}")
        print(f"  beta2: {ADAM_BETA2}")
        
        print_job_summary(job_ids)


if __name__ == "__main__":
    main()
