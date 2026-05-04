#!/usr/bin/env python3
"""
Orchestrator for the data-scaling experiment.

Modes (selected via --mode):

    samples   Build the train sample JSON for the full WG pool (no SLURM).
    traces    Submit the SLURM trace-gen job.
    train     Submit the SLURM training job (1 epoch, full corpus).
    test      Submit the SLURM test-eval job (all 6 EVAL_DATASET_DIRS).

Reads configs/experiments/scaling/data_scaling.yaml.

Usage:
    cd /scratch/s4626451/intention-jailbreak

    python scripts/hpc/scaling/submit_scaling.py --mode samples
    python scripts/hpc/scaling/submit_scaling.py --mode traces
    python scripts/hpc/scaling/submit_scaling.py --mode train
    python scripts/hpc/scaling/submit_scaling.py --mode test

All modes accept --dry-run and --force.
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
DISTILL_TEMPLATE   = str(SCALING_SCRIPT_DIR / "scaling_distill_template.sh")
TEST_EVAL_TEMPLATE = str(SCALING_SCRIPT_DIR / "scaling_test_eval_template.sh")
PREPARE_SAMPLES_SCRIPT = str(SCALING_SCRIPT_DIR / "prepare_scaling_samples.py")

CONDITION_TO_EVAL = {
    "no_intent":        "finetuned_reasoning_classification",
    "synthetic_intent": "finetuned_reasoning_synthetic_intent",
    "human_intent":     "finetuned_reasoning_human_intent",
}


def adapter_run_name(cfg: dict, data_condition_name: str) -> str:
    """Stable filesystem-safe adapter directory name (no `_adapter` suffix)."""
    teacher_slug = cfg["teacher"]["slug"]
    student_slug = cfg["student"]["slug"]
    cond_slug = cfg["condition"].replace("_", "-")
    data_slug = data_condition_name.replace("_", "-")
    lr_str = f"{float(cfg['training']['learning_rate']):.0e}"
    epochs = cfg["training"]["epochs"]
    version = cfg["run_version"]
    return f"{teacher_slug}--{student_slug}--{cond_slug}--{data_slug}--lr{lr_str}--ep{epochs}--{version}"


def adapter_dir(cfg: dict, data_condition_name: str) -> Path:
    return PROJECT_ROOT / cfg["paths"]["adapters_base_dir"] / f"{adapter_run_name(cfg, data_condition_name)}_adapter"


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


# ── Mode: train ──────────────────────────────────────────────────────────────

def mode_train(cfg: dict, dry_run: bool, force: bool):
    student_hf = cfg["student"]["hf_id"]
    cond = cfg["condition"]
    lr = float(cfg["training"]["learning_rate"])
    epochs = cfg["training"]["epochs"]
    val_path = cfg["shared_traces"]["validation_path"]
    test_path = cfg["shared_traces"]["test_path"]

    paths = cfg["paths"]
    wandb_project = cfg["wandb"]["project"]

    if not dry_run:
        create_logs_dir()

    submitted, skipped = [], []
    for entry in cfg["data_conditions"]:
        name = entry["name"]
        train_path = trace_train_path(cfg, name)
        if not train_path.exists():
            print(f"  [error] {name}: train traces missing at {train_path}  (run --mode traces first)")
            continue

        adir = adapter_dir(cfg, name)
        if not force and adir.exists() and (adir / "adapter_config.json").exists():
            print(f"  [skip] {name} — adapter already trained at {adir}")
            skipped.append(name)
            continue

        run_name = adapter_run_name(cfg, name)
        export_vars = {
            "STUDENT_MODEL":       student_hf,
            "LEARNING_RATE":       f"{lr}",
            "EPOCHS":              str(epochs),
            "CONDITION":           cond,
            "TRACES_TRAIN_PATH":   str(train_path.relative_to(PROJECT_ROOT)),
            "TRACES_VAL_PATH":     val_path,
            "TRACES_TEST_PATH":    test_path,
            "RUN_NAME":            run_name,
            "ADAPTER_BASE_DIR":    paths["adapters_base_dir"],
            "TRAIN_RESULTS_DIR":   paths["train_results_dir"],
            "PREDICTIONS_DIR":     paths["predictions_dir"],
            "WANDB_PROJECT":       wandb_project,
            "WANDB_RUN_NAME":      f"{name}--lr{lr:.0e}--ep{epochs}--{cfg['run_version']}",
        }
        label = f"train / {name}"
        if dry_run:
            print(f"  [dry-run] {label}")
            for k, v in export_vars.items():
                print(f"      {k}={v}")
            continue
        try:
            job_id = submit_sbatch(DISTILL_TEMPLATE, export_vars)
            print(f"  ✓ {label} → Job {job_id}")
            submitted.append((label, job_id))
        except subprocess.CalledProcessError as e:
            print(f"  ✗ FAILED {label}: {e.stderr}")

    print(f"\nSubmitted: {len(submitted)}  |  skipped: {len(skipped)}")
    if submitted:
        print_job_summary(submitted)


# ── Mode: test ───────────────────────────────────────────────────────────────

def mode_test(cfg: dict, dry_run: bool, force: bool, eval_config: str):
    student_hf = cfg["student"]["hf_id"]
    eval_cond = CONDITION_TO_EVAL[cfg["condition"]]
    test_base = PROJECT_ROOT / cfg["paths"]["test_eval_base_dir"]

    if not dry_run:
        create_logs_dir()

    submitted, skipped = [], []
    for entry in cfg["data_conditions"]:
        name = entry["name"]
        adir = adapter_dir(cfg, name)
        if not adir.exists():
            print(f"  [error] {name}: adapter missing at {adir}  (run --mode train first)")
            continue

        out_dir = test_base / name
        if not force and out_dir.exists() and any(out_dir.rglob("*.jsonl")):
            print(f"  [skip] {name} — test eval already present at {out_dir}")
            skipped.append(name)
            continue

        run_label = f"scaling-test--{name}"
        export_vars = {
            "STUDENT_MODEL":  student_hf,
            "ADAPTER_PATH":   str(adir),
            "EVAL_CONDITION": eval_cond,
            "EVAL_CONFIG":    eval_config,
            "OUTPUT_DIR":     str(out_dir),
            "WANDB_RUN_NAME": run_label,
        }
        label = f"test / {name}"
        if dry_run:
            print(f"  [dry-run] {label}")
            for k, v in export_vars.items():
                print(f"      {k}={v}")
            continue
        try:
            job_id = submit_sbatch(TEST_EVAL_TEMPLATE, export_vars)
            print(f"  ✓ {label} → Job {job_id}")
            submitted.append((label, job_id))
        except subprocess.CalledProcessError as e:
            print(f"  ✗ FAILED {label}: {e.stderr}")

    print(f"\nSubmitted: {len(submitted)}  |  skipped: {len(skipped)}")
    if submitted:
        print_job_summary(submitted)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", required=True, choices=["samples", "traces", "train", "test"],
                        help="Pipeline stage to run.")
    parser.add_argument("--dry-run", "-n", action="store_true",
                        help="Print planned actions without submitting / running.")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even when outputs already exist on disk.")
    parser.add_argument("--eval-config", default="eval_distillation",
                        help="(test mode) Hydra config name for test eval (default: eval_distillation).")
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
        "train":   lambda: mode_train(cfg, args.dry_run, args.force),
        "test":    lambda: mode_test(cfg, args.dry_run, args.force, args.eval_config),
    }[args.mode]()


if __name__ == "__main__":
    main()
