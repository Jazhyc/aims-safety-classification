#!/usr/bin/env python3
"""
Orchestrator for the data-scaling experiment.

Currently supports only the first two pipeline stages — sample preparation and
teacher trace generation. Subsampling, training, and eval will be added once
the full trace corpus is on disk.

Modes (selected via --mode):

    samples   Build the train sample JSON for the full WG pool (no SLURM).
    traces    Submit the SLURM trace-gen job.

Reads configs/experiments/scaling/data_scaling.yaml.

Usage:
    cd /scratch/s4626451/intention-jailbreak

    python scripts/hpc/scaling/submit_scaling.py --mode samples
    python scripts/hpc/scaling/submit_scaling.py --mode traces

Both modes accept --dry-run and --force.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "hpc"))
from slurm_utils import create_logs_dir, submit_sbatch, print_header, print_job_summary


CONFIG_PATH = PROJECT_ROOT / "configs" / "experiments" / "scaling" / "data_scaling.yaml"

SCALING_SCRIPT_DIR = PROJECT_ROOT / "scripts" / "hpc" / "scaling"
TRACE_GEN_TEMPLATE = str(SCALING_SCRIPT_DIR / "scaling_trace_gen_template.sh")
PREPARE_SAMPLES_SCRIPT = str(SCALING_SCRIPT_DIR / "prepare_scaling_samples.py")


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def trace_train_path(cfg: dict, data_condition_name: str) -> Path:
    teacher_slug = cfg["teacher"]["slug"]
    return (
        PROJECT_ROOT / cfg["paths"]["traces_base_dir"] / teacher_slug
        / data_condition_name / "train" / "parsed_results.json"
    )


# ── Mode: samples ────────────────────────────────────────────────────────────

def mode_samples(cfg: dict, dry_run: bool):
    seed = cfg["seed"]
    out_dir = PROJECT_ROOT / cfg["paths"]["samples_dir"]
    cmd = [
        sys.executable,
        PREPARE_SAMPLES_SCRIPT,
        "--seed", str(seed),
        "--output-dir", str(out_dir),
    ]
    print(f"\nCommand: {' '.join(cmd)}")
    if dry_run:
        print("[dry-run] would run sample preparation.")
        return
    subprocess.run(cmd, check=True)


# ── Mode: traces ─────────────────────────────────────────────────────────────

def mode_traces(cfg: dict, dry_run: bool, force: bool):
    teacher_hf  = cfg["teacher"]["hf_id"]
    teacher_slug = cfg["teacher"]["slug"]
    cond = cfg["condition"]
    traces_base = cfg["paths"]["traces_base_dir"]

    if not dry_run:
        create_logs_dir()

    submitted, skipped = [], []
    for entry in cfg["data_conditions"]:
        name = entry["name"]
        samples_json = entry["samples_json"]
        out_path = trace_train_path(cfg, name)

        if not force and out_path.exists() and out_path.stat().st_size > 0:
            print(f"  [skip] {name} — traces already present at {out_path}")
            skipped.append(name)
            continue

        if not (PROJECT_ROOT / samples_json).exists():
            print(f"  [error] samples JSON missing: {samples_json}  "
                  f"(run --mode samples first)")
            continue

        export_vars = {
            "MODEL_NAME":          teacher_hf,
            "OUTPUT_DIR":          traces_base,
            "THINKING_MODE":       "false",
            "SAMPLES_JSON":        samples_json,
            "CONDITION":           cond,
            "DATA_CONDITION_DIR":  name,
        }
        label = f"trace-gen / {teacher_slug} / {name}"
        if dry_run:
            print(f"  [dry-run] {label}")
            for k, v in export_vars.items():
                print(f"      {k}={v}")
            continue
        try:
            job_id = submit_sbatch(TRACE_GEN_TEMPLATE, export_vars)
            print(f"  ✓ {label} → Job {job_id}")
            submitted.append((label, job_id))
        except subprocess.CalledProcessError as e:
            print(f"  ✗ FAILED {label}: {e.stderr}")

    print(f"\nSubmitted: {len(submitted)}  |  skipped: {len(skipped)}")
    if submitted:
        print_job_summary(submitted)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", required=True, choices=["samples", "traces"],
                        help="Pipeline stage to run.")
    parser.add_argument("--dry-run", "-n", action="store_true",
                        help="Print planned actions without submitting / running.")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even when outputs already exist on disk.")
    args = parser.parse_args()

    cfg = load_config()

    print_header(
        f"Data-scaling experiment — mode={args.mode}",
        f"teacher={cfg['teacher']['slug']}  condition={cfg['condition']}  seed={cfg['seed']}"
        + ("  [DRY RUN]" if args.dry_run else "")
        + ("  [FORCE]" if args.force else ""),
    )

    {
        "samples": lambda: mode_samples(cfg, args.dry_run),
        "traces":  lambda: mode_traces(cfg, args.dry_run, args.force),
    }[args.mode]()


if __name__ == "__main__":
    main()
