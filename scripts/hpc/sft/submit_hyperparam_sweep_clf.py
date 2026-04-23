#!/usr/bin/env python3
"""
SLURM Orchestrator for Hyperparameter Sweep (Classification-Only)

Submits multiple SLURM jobs to sweep over learning rates for Llama 3.1 8B
in classification-only mode: the model outputs only "safe" or "harmful"
with no intent generation.

Usage:
    python scripts/hpc/submit_hyperparam_sweep_clf.py
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

# Epochs to sweep over
EPOCHS = [5]

# Adam optimizer hyperparameters
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.98


def submit_job(lr: float, epochs: int) -> str:
    template_path = "scripts/hpc/hyperparam_sweep_clf_template.sh"
    export_vars = {
        'LEARNING_RATE': lr,
        'EPOCHS': epochs,
        'ADAM_BETA1': ADAM_BETA1,
        'ADAM_BETA2': ADAM_BETA2,
    }
    return submit_sbatch(template_path, export_vars)


def main():
    total_jobs = len(LEARNING_RATES) * len(EPOCHS)
    print_header(
        "Hyperparameter Sweep (Classification-Only) - SLURM Job Submission",
        f"Model: {MODEL_HF_PATH}"
    )
    print(f"\nSubmitting {total_jobs} jobs ({len(LEARNING_RATES)} LRs × {len(EPOCHS)} epochs)...\n")

    create_logs_dir()

    job_ids = []
    for lr in LEARNING_RATES:
        for epochs in EPOCHS:
            try:
                job_id = submit_job(lr, epochs)
                job_ids.append((lr, epochs, job_id))
                print(f"✓ Submitted lr={lr:.0e}, epochs={epochs}: Job ID {job_id}")
            except subprocess.CalledProcessError as e:
                print(f"✗ Failed to submit lr={lr:.0e}, epochs={epochs}: {e.stderr}")
            except Exception as e:
                print(f"✗ Error submitting lr={lr:.0e}, epochs={epochs}: {str(e)}")

    print("\n" + "=" * 60)
    print(f"Successfully submitted {len(job_ids)}/{total_jobs} jobs")
    print("=" * 60)

    if job_ids:
        print("\nJob Summary:")
        for lr, epochs, job_id in job_ids:
            print(f"  lr={lr:.0e}, epochs={epochs} → Job {job_id}")

        print(f"\nAdam hyperparameters:")
        print(f"  beta1: {ADAM_BETA1}")
        print(f"  beta2: {ADAM_BETA2}")

        print_job_summary([(f"lr={lr:.0e}_e={epochs}", job_id) for lr, epochs, job_id in job_ids])


if __name__ == "__main__":
    main()
