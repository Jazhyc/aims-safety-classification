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
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from slurm_utils import create_logs_dir, submit_sbatch, print_header, print_job_summary


# List of models to test - add or modify as needed
MODELS = [
    # ("Qwen/Qwen3-0.6B", "qwen3-0.6b"),
    # ("Qwen/Qwen3-4B", "qwen3-4b"),
    # ("Qwen/Qwen3-8B", "qwen3-8b"),
    # ("Qwen/Qwen3-14B", "qwen3-14b"),
    # ("google/gemma-3-4b-it", "gemma3-4b"),
    # ("google/gemma-3-12b-it", "gemma3-12b"),
    # ("meta-llama/Llama-3.2-1B-Instruct", "llama3.2-1b"),
    # ("meta-llama/Llama-3.1-8B-Instruct", "llama3.1-8b"),
]


def submit_job(model_hf_path: str, model_name: str) -> str:
    """
    Submit a SLURM job for a specific model.
    
    Args:
        model_hf_path: HuggingFace model path (e.g., "Qwen/Qwen3-0.6B")
        model_name: Short name for the model (e.g., "qwen3-0.6b")
    
    Returns:
        Job ID as string
    """
    template_path = "scripts/hpc/baselines/generation_template.sh"
    export_vars = {
        'MODEL_HF_PATH': model_hf_path,
        'MODEL_NAME': model_name,
    }
    return submit_sbatch(template_path, export_vars)


def main():
    """Main orchestration function."""
    print_header("LLM Fine-tuning Sweep - SLURM Job Submission")
    print(f"\nSubmitting {len(MODELS)} jobs...\n")
    
    # Create necessary directories
    logs_dir = create_logs_dir()
    print(f"✓ Created logs directory: {logs_dir}")
    
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
        
        print_job_summary(job_ids)


if __name__ == "__main__":
    main()
