#!/usr/bin/env python3
"""
Orchestrator for the broader-intent-mix ablation.

Trains a single (teacher, student, lr) point on a mixed training set:
  - annotated `human_intent` train traces (n=1378), plus
  - harm-stratified `synthetic_intent` traces from the scaling pool
    (n=1378, excluding annotated prompts).

Both halves' teacher traces already exist on disk, so no SLURM
trace-generation step is needed. The trace loader in causal.py accepts
a list-valued `reasoning_traces_condition`; per-record condition selects
the intent source (`ground_truth.intent` vs `predicted.prompt_intent`).

Modes (selected via --mode):

    samples   Build the merged parsed_results.json locally (no SLURM).
    train     Submit the SLURM training job.
    ood-val   Submit OOD validation (ToxicChat + Aegis).
    test      Submit test eval on the held-out OOD test suite.

Usage:
    cd /scratch/s4626451/intention-jailbreak
    python scripts/hpc/ablations/submit_broader_intent_mix.py --mode samples
    python scripts/hpc/ablations/submit_broader_intent_mix.py --mode train
    python scripts/hpc/ablations/submit_broader_intent_mix.py --mode ood-val
    python scripts/hpc/ablations/submit_broader_intent_mix.py --mode test
"""

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "hpc"))
from slurm_utils import create_logs_dir, submit_sbatch, print_header, print_job_summary


CONFIG_PATH = PROJECT_ROOT / "configs" / "experiments" / "ablations" / "broader_intent_mix.yaml"

ABLATION_SCRIPT_DIR = PROJECT_ROOT / "scripts" / "hpc" / "ablations"
DISTILL_TEMPLATE   = str(ABLATION_SCRIPT_DIR / "broader_intent_mix_distill_template.sh")
OOD_EVAL_TEMPLATE  = str(ABLATION_SCRIPT_DIR / "ablation_ood_eval_template.sh")
TEST_EVAL_TEMPLATE = str(ABLATION_SCRIPT_DIR / "ablation_test_eval_template.sh")
PREPARE_SAMPLES_SCRIPT = str(ABLATION_SCRIPT_DIR / "prepare_broader_intent_mix_samples.py")

CONDITION_TO_EVAL = {
    "no_intent":        "finetuned_reasoning_classification",
    "synthetic_intent": "finetuned_reasoning_synthetic_intent",
    "human_intent":     "finetuned_reasoning_human_intent",
}


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def adapter_run_name(cfg: dict, data_condition_name: str) -> str:
    teacher_slug = cfg["teacher"]["slug"]
    student_slug = cfg["student"]["slug"]
    data_slug = data_condition_name.replace("_", "-")
    lr_str = f"{float(cfg['learning_rate']):.0e}"
    epochs = cfg["epochs"]
    version = cfg["run_version"]
    return f"{teacher_slug}--{student_slug}--{data_slug}--lr{lr_str}--ep{epochs}--{version}"


def adapter_dir(cfg: dict, data_condition_name: str) -> Path:
    return PROJECT_ROOT / cfg["paths"]["adapters_base_dir"] / f"{adapter_run_name(cfg, data_condition_name)}_adapter"


# ── Mode: samples ────────────────────────────────────────────────────────────

def mode_samples(cfg: dict, dry_run: bool, force: bool):
    seed = cfg["seed"]
    teacher_slug = cfg["teacher"]["slug"]
    out_path = PROJECT_ROOT / cfg["data_conditions"][0]["traces_path"]

    if not force and out_path.exists() and out_path.stat().st_size > 0:
        print(f"  [skip] merged traces already present at {out_path}")
        return

    cmd = [
        sys.executable,
        PREPARE_SAMPLES_SCRIPT,
        "--seed", str(seed),
        "--teacher-slug", teacher_slug,
        "--output-dir", str(out_path.parent),
    ]
    print(f"\nCommand: {' '.join(cmd)}")
    if dry_run:
        print("[dry-run] would build merged traces.")
        return
    subprocess.run(cmd, check=True)


# ── Mode: train ──────────────────────────────────────────────────────────────

def mode_train(cfg: dict, dry_run: bool, force: bool):
    student_hf = cfg["student"]["hf_id"]
    lr = float(cfg["learning_rate"])
    epochs = cfg["epochs"]
    conditions = cfg["conditions"]
    conditions_list = ",".join(conditions)
    val_path = cfg["shared_traces"]["validation_path"]
    test_path = cfg["shared_traces"]["test_path"]

    paths = cfg["paths"]
    wandb_project = cfg["wandb"]["project"]

    if not dry_run:
        create_logs_dir()

    submitted, skipped = [], []
    for entry in cfg["data_conditions"]:
        name = entry["name"]
        train_path = PROJECT_ROOT / entry["traces_path"]
        if not train_path.exists():
            print(f"  [error] {name}: merged train traces missing at {train_path}  (run --mode samples first)")
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
            "CONDITIONS_LIST":     conditions_list,
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


# ── Mode: ood-val ────────────────────────────────────────────────────────────

def _ood_val_done(out_dir: Path) -> bool:
    if not out_dir.exists():
        return False
    return any((out_dir / "toxic-chat").glob("*.jsonl")) and any((out_dir / "aegis").glob("*.jsonl"))


def mode_ood_val(cfg: dict, dry_run: bool, force: bool):
    student_hf = cfg["student"]["hf_id"]
    eval_cond = CONDITION_TO_EVAL[cfg["eval_condition"]]
    ood_base = PROJECT_ROOT / cfg["paths"]["ood_val_base_dir"]

    if not dry_run:
        create_logs_dir()

    submitted, skipped = [], []
    for entry in cfg["data_conditions"]:
        name = entry["name"]
        adir = adapter_dir(cfg, name)
        if not adir.exists():
            print(f"  [error] {name}: adapter missing at {adir}  (run --mode train first)")
            continue

        out_dir = ood_base / name / adir.name
        if not force and _ood_val_done(out_dir):
            print(f"  [skip] {name} — OOD val results already present at {out_dir}")
            skipped.append(name)
            continue

        run_label = f"ablation-ood-val--{name}--{adir.name}"
        export_vars = {
            "STUDENT_MODEL":  student_hf,
            "ADAPTER_PATH":   str(adir),
            "EVAL_CONDITION": eval_cond,
            "OUTPUT_DIR":     str(out_dir),
            "WANDB_RUN_NAME": run_label,
        }
        label = f"ood-val / {name}"
        if dry_run:
            print(f"  [dry-run] {label}")
            for k, v in export_vars.items():
                print(f"      {k}={v}")
            continue
        try:
            job_id = submit_sbatch(OOD_EVAL_TEMPLATE, export_vars)
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
    eval_cond = CONDITION_TO_EVAL[cfg["eval_condition"]]
    test_base = PROJECT_ROOT / cfg["paths"]["test_eval_base_dir"]

    if not dry_run:
        create_logs_dir()

    submitted, skipped = [], []
    for entry in cfg["data_conditions"]:
        name = entry["name"]
        adir = adapter_dir(cfg, name)
        if not adir.exists():
            print(f"  [error] {name}: adapter missing at {adir}")
            continue

        out_dir = test_base / name
        if not force and out_dir.exists() and any(out_dir.rglob("*.jsonl")):
            print(f"  [skip] {name} — test eval already present at {out_dir}")
            skipped.append(name)
            continue

        run_label = f"ablation-test--{name}"
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
    parser.add_argument("--mode", required=True,
                        choices=["samples", "train", "ood-val", "test"],
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
        f"Broader-intent-mix ablation — mode={args.mode}",
        f"teacher={cfg['teacher']['slug']}  student={cfg['student']['slug']}  "
        f"conditions={cfg['conditions']}  lr={cfg['learning_rate']}  "
        f"epochs={cfg['epochs']}  seed={cfg['seed']}"
        + ("  [DRY RUN]" if args.dry_run else "")
        + ("  [FORCE]" if args.force else ""),
    )

    {
        "samples":  lambda: mode_samples(cfg, args.dry_run, args.force),
        "train":    lambda: mode_train(cfg, args.dry_run, args.force),
        "ood-val":  lambda: mode_ood_val(cfg, args.dry_run, args.force),
        "test":     lambda: mode_test(cfg, args.dry_run, args.force, args.eval_config),
    }[args.mode]()


if __name__ == "__main__":
    main()
