#!/usr/bin/env python3
"""
SLURM Orchestrator for LLM Fine-tuning Sweep

This script submits multiple SLURM jobs in parallel to fine-tune different LLMs
using the same hyperparameters. Each job will train a different model specified
in the MODELS list.

Usage:
    python scripts/hpc/submit_llm_sweep.py
"""

import subprocess
import os
from pathlib import Path


# List of models to test - add or modify as needed
MODELS = [
    # ("Qwen/Qwen3-0.6B", "qwen3-0.6b"),
    # ("Qwen/Qwen3-4B", "qwen3-4b"),
    # ("Qwen/Qwen3-8B", "qwen3-8b"),
    ("Qwen/Qwen3-14B", "qwen3-14b"),
    ("google/gemma-3-4b-it", "gemma3-4b"),
    ("google/gemma-3-12b-it", "gemma3-12b"),
]


def create_logs_dir():
    """Ensure the logs directory exists for SLURM output."""
    logs_dir = Path("logs/slurm")
    logs_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ Created logs directory: {logs_dir}")


def submit_job(model_hf_path: str, model_name: str) -> str:
    """
    Submit a SLURM job for a specific model.
    
    Args:
        model_hf_path: HuggingFace model path (e.g., "Qwen/Qwen3-0.6B")
        model_name: Short name for the model (e.g., "qwen3-0.6b")
    
    Returns:
        Job ID as string
    """
    # Set environment variables for the SLURM script
    env = os.environ.copy()
    env['MODEL_HF_PATH'] = model_hf_path
    env['MODEL_NAME'] = model_name
    
    # Submit the job
    template_path = "scripts/hpc/generation_template.sh"
    result = subprocess.run(
        ['sbatch', f'--export=MODEL_HF_PATH={model_hf_path},MODEL_NAME={model_name}', template_path],
        capture_output=True,
        text=True,
        check=True
    )
    
    # Extract job ID from output (typically "Submitted batch job 12345")
    job_id = result.stdout.strip().split()[-1]
    return job_id


def main():
    """Main orchestration function."""
    print("=" * 60)
    print("LLM Fine-tuning Sweep - SLURM Job Submission")
    print("=" * 60)
    print(f"\nSubmitting {len(MODELS)} jobs...\n")
    
    # Create necessary directories
    create_logs_dir()
    
    # Submit all jobs
    job_ids = []
    for model_hf_path, model_name in MODELS:
        try:
            job_id = submit_job(model_hf_path, model_name)
            job_ids.append((model_name, job_id))
            print(f"✓ Submitted {model_name}: Job ID {job_id}")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to submit {model_name}: {e.stderr}")
        except Exception as e:
            print(f"✗ Error submitting {model_name}: {str(e)}")
    
    # Summary
    print("\n" + "=" * 60)
    print(f"Successfully submitted {len(job_ids)}/{len(MODELS)} jobs")
    print("=" * 60)
    
    if job_ids:
        print("\nJob Summary:")
        for model_name, job_id in job_ids:
            print(f"  {model_name:20s} → Job {job_id}")
        
        print("\nMonitor jobs with:")
        print(f"  squeue -u $USER")
        print("\nCheck specific job:")
        print(f"  squeue -j {job_ids[0][1]}")
        print("\nCancel all jobs:")
        print(f"  scancel {' '.join(job_id for _, job_id in job_ids)}")


if __name__ == "__main__":
    main()
