#!/usr/bin/env python3
"""
SLURM Orchestrator for Distillation Hyperparameter Sweep

Sweeps over (teacher, condition, student model, learning rate) combinations,
submitting one SLURM job per configuration.

W&B project: one per (student x teacher) combo, so runs within each project
only vary by condition and learning rate.

Usage:
    cd /scratch/s4626451/intention-jailbreak

    # Validate trace files and preview all jobs without submitting:
    python scripts/hpc/submit_distillation_sweep.py --dry-run

    # Submit all jobs:
    python scripts/hpc/submit_distillation_sweep.py
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from slurm_utils import create_logs_dir, submit_sbatch, print_header, print_job_summary


# ── Sweep dimensions ──────────────────────────────────────────────────────────

# Teacher model slug → base dir for pre-generated reasoning traces.
# Slug must match model_name.replace("/", "-") used by generate_reasoning_traces.py.
TEACHER_MODELS = [
    ("cyankiwi/Mistral-Small-4-119B-2603-AWQ-4bit",  "cyankiwi-Mistral-Small-4-119B-2603-AWQ-4bit"),
    ("cyankiwi/Qwen3.5-122B-A10B-AWQ-4bit",          "cyankiwi-Qwen3.5-122B-A10B-AWQ-4bit"),
    ("openai/gpt-oss-120b",                           "openai-gpt-oss-120b"),
    ("RedHatAI/gemma-3-27b-it-quantized.w4a16",       "RedHatAI-gemma-3-27b-it-quantized.w4a16"),
]

# (HuggingFace model ID, short slug used in run names and W&B project names)
STUDENT_MODELS = [
    # ("meta-llama/Llama-3.1-8B-Instruct",              "llama-3.1-8b"),
    # ("mistralai/Ministral-3-14B-Reasoning-2512",        "ministral-3-14b-reasoning"),
    ("Qwen/Qwen3-8B",                                  "qwen3-8b"),
    # ("google/gemma-3-12b-it",                          "gemma-3-12b"),
]

LEARNING_RATES = [
    1e-5,
    2e-5,
    5e-5,
    1e-4,
    2e-4,
]

CONDITIONS = ["no_intent", "synthetic_intent", "human_intent"]

# Version tag appended to all run names and W&B run names for traceability.
TRACES_VERSION = "v7"

# Base directory where v7 reasoning traces are stored.
TRACES_BASE_DIR = f"data/reasoning_traces_{TRACES_VERSION}"

TEMPLATE_PATH = "scripts/hpc/distillation_sweep_template.sh"


# ── Validation ────────────────────────────────────────────────────────────────

def validate_traces() -> bool:
    """Check that train and validation parsed_results.json exist and are non-empty
    for every teacher model. Returns True if all checks pass."""
    all_ok = True
    for _, teacher_slug in TEACHER_MODELS:
        for split in ("train", "validation"):
            path = Path(TRACES_BASE_DIR) / teacher_slug / split / "parsed_results.json"
            if not path.exists():
                print(f"  [MISSING] {path}")
                all_ok = False
            else:
                try:
                    data = json.loads(path.read_text())
                    if not data:
                        print(f"  [EMPTY]   {path}")
                        all_ok = False
                    else:
                        print(f"  [OK]      {path}  ({len(data)} examples)")
                except json.JSONDecodeError as e:
                    print(f"  [CORRUPT] {path}: {e}")
                    all_ok = False
    return all_ok


# ── Helpers ───────────────────────────────────────────────────────────────────

def condition_slug(condition: str) -> str:
    return condition.replace("_", "-")


def build_export_vars(
    teacher_slug: str,
    student_hf: str,
    student_slug: str,
    lr: float,
    condition: str,
) -> dict:
    cond_slug = condition_slug(condition)
    lr_str = f"{lr:.0e}"

    # Filesystem-unique run name: encodes all sweep dimensions including trace version.
    run_name = f"{teacher_slug}--{student_slug}--{cond_slug}--lr{lr_str}--{TRACES_VERSION}"

    # W&B: one project per (student x teacher); run name captures condition, lr, and version.
    wandb_project = f"distillation-sweep-{student_slug}--{teacher_slug}"
    wandb_run_name = f"{cond_slug}--lr{lr_str}--{TRACES_VERSION}"

    return {
        "STUDENT_MODEL":      student_hf,
        "LEARNING_RATE":      lr,
        "CONDITION":          condition,
        "TRACES_TRAIN_PATH":  f"{TRACES_BASE_DIR}/{teacher_slug}/train/parsed_results.json",
        "TRACES_VAL_PATH":    f"{TRACES_BASE_DIR}/{teacher_slug}/validation/parsed_results.json",
        "TRACES_TEST_PATH":   f"{TRACES_BASE_DIR}/{teacher_slug}/test/parsed_results.json",
        "RUN_NAME":           run_name,
        "WANDB_PROJECT":      wandb_project,
        "WANDB_RUN_NAME":     wandb_run_name,
    }


def submit_job(
    teacher_hf: str,
    teacher_slug: str,
    student_hf: str,
    student_slug: str,
    lr: float,
    condition: str,
    dry_run: bool = False,
) -> str:
    """Build export vars and submit (or print in dry-run mode). Returns job ID or run name."""
    export_vars = build_export_vars(teacher_slug, student_hf, student_slug, lr, condition)
    if dry_run:
        print(f"    RUN_NAME:      {export_vars['RUN_NAME']}")
        print(f"    WANDB_PROJECT: {export_vars['WANDB_PROJECT']}")
        print(f"    WANDB_RUN:     {export_vars['WANDB_RUN_NAME']}")
        print(f"    TRACES_TRAIN:  {export_vars['TRACES_TRAIN_PATH']}")
        return export_vars["RUN_NAME"]
    return submit_sbatch(TEMPLATE_PATH, export_vars)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--dry-run", "-n", action="store_true",
        help="Validate traces and print all job configs without submitting to SLURM.",
    )
    args = parser.parse_args()

    total_jobs = (
        len(TEACHER_MODELS) * len(STUDENT_MODELS) * len(LEARNING_RATES) * len(CONDITIONS)
    )

    print_header(
        "Distillation Hyperparameter Sweep — SLURM Job Submission",
        f"{len(TEACHER_MODELS)} teacher(s) × {len(STUDENT_MODELS)} student(s) "
        f"× {len(LEARNING_RATES)} LRs × {len(CONDITIONS)} conditions = {total_jobs} jobs"
        + (" [DRY RUN]" if args.dry_run else ""),
    )

    # Always validate traces before doing anything.
    print("\nValidating reasoning trace files...")
    if not validate_traces():
        print("\nAborting: one or more trace files are missing or empty.")
        sys.exit(1)
    print("All trace files OK.\n")

    if not args.dry_run:
        create_logs_dir()

    job_ids = []
    for teacher_hf, teacher_slug in TEACHER_MODELS:
        if args.dry_run:
            print(f"\n── Teacher: {teacher_slug} ──")
        for student_hf, student_slug in STUDENT_MODELS:
            for condition in CONDITIONS:
                for lr in LEARNING_RATES:
                    label = (
                        f"teacher={teacher_slug}, student={student_slug}, "
                        f"condition={condition}, lr={lr:.0e}"
                    )
                    try:
                        job_id = submit_job(
                            teacher_hf, teacher_slug,
                            student_hf, student_slug,
                            lr, condition,
                            dry_run=args.dry_run,
                        )
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
        print(f"Dry run complete — {len(job_ids)}/{total_jobs} jobs would be submitted.")
        print("Run without --dry-run to submit.")
    else:
        print(f"Submitted {len(job_ids)}/{total_jobs} jobs")
        if job_ids:
            print_job_summary(job_ids)
    print("=" * 60)


if __name__ == "__main__":
    main()
