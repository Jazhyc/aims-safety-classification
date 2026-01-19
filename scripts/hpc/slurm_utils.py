"""
Common utilities for SLURM job submission.
"""

import subprocess
from pathlib import Path


def create_logs_dir():
    """Ensure the logs directory exists for SLURM output."""
    logs_dir = Path("logs/slurm")
    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir


def submit_sbatch(template_path: str, export_vars: dict) -> str:
    """
    Submit a SLURM job using sbatch.
    
    Args:
        template_path: Path to the SLURM script template
        export_vars: Dictionary of environment variables to export
    
    Returns:
        Job ID as string
    
    Raises:
        subprocess.CalledProcessError: If sbatch submission fails
    """
    # Build export string
    export_str = ','.join(f'{k}={v}' for k, v in export_vars.items())
    
    # Submit the job
    result = subprocess.run(
        ['sbatch', f'--export={export_str}', template_path],
        capture_output=True,
        text=True,
        check=True
    )
    
    # Extract job ID from output (typically "Submitted batch job 12345")
    job_id = result.stdout.strip().split()[-1]
    return job_id


def print_header(title: str, subtitle: str = None):
    """Print a formatted header."""
    print("=" * 60)
    print(title)
    if subtitle:
        print(subtitle)
    print("=" * 60)


def print_job_summary(job_ids: list):
    """
    Print a summary of submitted jobs with monitoring commands.
    
    Args:
        job_ids: List of tuples containing job information
    """
    if not job_ids:
        return
    
    print("\nMonitor jobs with:")
    print(f"  squeue -u $USER")
    print("\nCheck specific job:")
    print(f"  squeue -j {job_ids[0][-1]}")  # First job ID
    print("\nCancel all jobs:")
    all_ids = ' '.join(str(item[-1]) for item in job_ids)
    print(f"  scancel {all_ids}")
