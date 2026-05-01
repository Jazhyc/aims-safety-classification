#!/usr/bin/env python3
"""
SLURM Orchestrator for v7 Reasoning Trace Generation

Submits one SLURM job per teacher model listed in TEACHER_MODELS.
Each job runs generate_reasoning_traces.py with the vLLM backend and
saves outputs to data/reasoning_traces_v7/<model-slug>/.

Usage:
    cd /scratch/s4626451/intention-jailbreak

    # Preview without submitting:
    python scripts/hpc/submit_trace_generation.py --dry-run

    # Submit all jobs:
    python scripts/hpc/submit_trace_generation.py
"""

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from slurm_utils import create_logs_dir, submit_sbatch, print_header, print_job_summary


# ── Teacher models to generate v7 traces for ─────────────────────────────────
# Each tuple: (HuggingFace model ID, thinking_mode)
# thinking_mode=false for non-native-thinking models.
TEACHER_MODELS = [
    # ("cyankiwi/gemma-4-31B-it-AWQ-4bit",             "false"),
    #("cyankiwi/Mistral-Small-4-119B-2603-AWQ-4bit",  "false"),
    ("Qwen/Qwen3-32B",                               "false"),
    #("openai/gpt-oss-120b",                         "false"),
    #("RedHatAI/gemma-3-27b-it-quantized.w4a16",     "false"),
]

TRACES_VERSION = "v7"
OUTPUT_DIR = f"data/reasoning_traces_{TRACES_VERSION}"
TEMPLATE_PATH = "scripts/hpc/distillation/trace_generation_template.sh"


def submit_job(model_name: str, thinking_mode: str, dry_run: bool = False) -> str:
    model_slug = model_name.replace("/", "-")
    export_vars = {
        "MODEL_NAME":    model_name,
        "OUTPUT_DIR":    OUTPUT_DIR,
        "THINKING_MODE": thinking_mode,
    }
    if dry_run:
        print(f"    MODEL_NAME:    {model_name}")
        print(f"    OUTPUT_DIR:    {OUTPUT_DIR}/{model_slug}")
        print(f"    THINKING_MODE: {thinking_mode}")
        return model_slug
    return submit_sbatch(TEMPLATE_PATH, export_vars)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--dry-run", "-n", action="store_true",
        help="Print all job configs without submitting.",
    )
    args = parser.parse_args()

    print_header(
        f"Reasoning Trace Generation — {TRACES_VERSION} — SLURM Job Submission",
        f"{len(TEACHER_MODELS)} teacher model(s)"
        + (" [DRY RUN]" if args.dry_run else ""),
    )

    print(f"\nOutput directory: {OUTPUT_DIR}/")
    print(f"Note: Kimi K2.5 v7 traces are generated separately via")
    print(f"      scripts/distillation/generate_kimi_v7_topup.py\n")

    if not args.dry_run:
        create_logs_dir()

    job_ids = []
    for model_name, thinking_mode in TEACHER_MODELS:
        label = model_name
        try:
            job_id = submit_job(model_name, thinking_mode, dry_run=args.dry_run)
            job_ids.append((label, job_id))
            if not args.dry_run:
                print(f"  ✓ {label} → Job {job_id}")
            else:
                print(f"  · {label}")
        except subprocess.CalledProcessError as e:
            print(f"  ✗ FAILED {label}: {e.stderr}")
        except Exception as e:
            print(f"  ✗ ERROR  {label}: {e}")

    print("\n" + "=" * 60)
    if args.dry_run:
        print(f"Dry run complete — {len(job_ids)}/{len(TEACHER_MODELS)} jobs would be submitted.")
        print("Run without --dry-run to submit.")
    else:
        print(f"Submitted {len(job_ids)}/{len(TEACHER_MODELS)} jobs")
        if job_ids:
            print_job_summary(job_ids)
    print("=" * 60)


if __name__ == "__main__":
    main()
