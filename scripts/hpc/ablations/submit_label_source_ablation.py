#!/usr/bin/env python3
"""
Orchestrator for the label-source ablation.

Runs the synthetic_intent distillation pipeline for two new data conditions
(hard_original, random_original) — same teacher (gpt-oss-120b), same student
(gemma-3-12b), best per-combo LR (2e-5), one seed. The third comparator
(hard_human) is the existing main result and is NOT re-run here.

Pipeline stages, selected via --mode:

    samples       Build the two train sample JSON files (no SLURM).
    traces        Submit SLURM trace-gen jobs for the two conditions
                  (only those whose train traces are missing).
    train         Submit SLURM training jobs for the two conditions
                  (requires traces present).
    ood-val       Submit OOD validation jobs for trained adapters.
    test          Submit test eval for trained adapters.

All paths and the LR + seed are read from configs/experiments/ablations/label_source_ablation.yaml.

Usage:
    cd /scratch/s4626451/intention-jailbreak

    # 1. Build samples (one-off, runs locally, ~30s)
    python scripts/hpc/ablations/submit_label_source_ablation.py --mode samples

    # 2. Generate teacher traces for the two new train sets
    python scripts/hpc/ablations/submit_label_source_ablation.py --mode traces

    # 3. After traces finish, train the students
    python scripts/hpc/ablations/submit_label_source_ablation.py --mode train

    # 4. After training, run OOD validation (This is not really needed since there is no selection)
    python scripts/hpc/ablations/submit_label_source_ablation.py --mode ood-val

    # 5. Then test eval
    python scripts/hpc/ablations/submit_label_source_ablation.py --mode test

    # Any mode supports --dry-run to preview without submitting.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "hpc"))
from slurm_utils import create_logs_dir, submit_sbatch, print_header, print_job_summary


CONFIG_PATH = PROJECT_ROOT / "configs" / "experiments" / "ablations" / "label_source_ablation.yaml"

ABLATION_SCRIPT_DIR = PROJECT_ROOT / "scripts" / "hpc" / "ablations"
TRACE_GEN_TEMPLATE = str(ABLATION_SCRIPT_DIR / "ablation_trace_gen_template.sh")
DISTILL_TEMPLATE   = str(ABLATION_SCRIPT_DIR / "ablation_distill_template.sh")
OOD_EVAL_TEMPLATE  = str(ABLATION_SCRIPT_DIR / "ablation_ood_eval_template.sh")
TEST_EVAL_TEMPLATE = str(ABLATION_SCRIPT_DIR / "ablation_test_eval_template.sh")

PREPARE_SAMPLES_SCRIPT = str(ABLATION_SCRIPT_DIR / "prepare_ablation_samples.py")

CONDITION_TO_EVAL = {
    "no_intent":        "finetuned_reasoning_classification",
    "synthetic_intent": "finetuned_reasoning_synthetic_intent",
    "human_intent":     "finetuned_reasoning_human_intent",
}


# ── Config loading ───────────────────────────────────────────────────────────

def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def adapter_run_name(cfg: dict, data_condition_name: str) -> str:
    """Stable filesystem-safe adapter directory name (no `_adapter` suffix)."""
    teacher_slug = cfg["teacher"]["slug"]
    student_slug = cfg["student"]["slug"]
    cond_slug = cfg["condition"].replace("_", "-")
    data_slug = data_condition_name.replace("_", "-")
    lr_str = f"{cfg['learning_rate']:.0e}"
    version = cfg["run_version"]
    return f"{teacher_slug}--{student_slug}--{cond_slug}--{data_slug}--lr{lr_str}--{version}"


def adapter_dir(cfg: dict, data_condition_name: str) -> Path:
    return PROJECT_ROOT / cfg["paths"]["adapters_base_dir"] / f"{adapter_run_name(cfg, data_condition_name)}_adapter"


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
    lr = float(cfg["learning_rate"])
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
        # NB: causal.py appends `_adapter` to paths.model_save_dir to form the
        # final adapter directory. Pass RUN_NAME without the suffix so the
        # resulting dir is `<run_name>_adapter`, matching adapter_dir() below
        # and the convention used by the main distillation sweep template.
        export_vars = {
            "STUDENT_MODEL":       student_hf,
            "LEARNING_RATE":       f"{lr}",
            "CONDITION":           cond,
            "TRACES_TRAIN_PATH":   str(train_path.relative_to(PROJECT_ROOT)),
            "TRACES_VAL_PATH":     val_path,
            "TRACES_TEST_PATH":    test_path,
            "RUN_NAME":            run_name,
            "ADAPTER_BASE_DIR":    paths["adapters_base_dir"],
            "TRAIN_RESULTS_DIR":   paths["train_results_dir"],
            "PREDICTIONS_DIR":     paths["predictions_dir"],
            "WANDB_PROJECT":       wandb_project,
            "WANDB_RUN_NAME":      f"{name}--lr{lr:.0e}--{cfg['run_version']}",
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
    eval_cond = CONDITION_TO_EVAL[cfg["condition"]]
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
    eval_cond = CONDITION_TO_EVAL[cfg["condition"]]
    test_base = PROJECT_ROOT / cfg["paths"]["test_eval_base_dir"]

    if not dry_run:
        create_logs_dir()

    # (label, adapter_path, output_dir, run_label) for ablation conditions + baselines.
    jobs: list[tuple[str, Path, Path, str]] = []
    for entry in cfg["data_conditions"]:
        name = entry["name"]
        adir = adapter_dir(cfg, name)
        if not adir.exists():
            print(f"  [error] {name}: adapter missing at {adir}")
            continue
        jobs.append((
            f"test / {name}",
            adir,
            test_base / name,
            f"ablation-test--{name}",
        ))
    for entry in cfg.get("baselines", []) or []:
        name = entry["name"]
        adir = PROJECT_ROOT / entry["adapter_path"]
        if not adir.exists():
            print(f"  [error] baseline {name}: adapter missing at {adir}")
            continue
        jobs.append((
            f"test / baseline:{name}",
            adir,
            PROJECT_ROOT / entry["test_output_dir"],
            f"ablation-test--baseline-{name}",
        ))

    submitted, skipped = [], []
    for label, adir, out_dir, run_label in jobs:
        if not force and out_dir.exists() and any(out_dir.rglob("*.jsonl")):
            print(f"  [skip] {label} — test eval already present at {out_dir}")
            skipped.append(label)
            continue
        export_vars = {
            "STUDENT_MODEL":  student_hf,
            "ADAPTER_PATH":   str(adir),
            "EVAL_CONDITION": eval_cond,
            "EVAL_CONFIG":    eval_config,
            "OUTPUT_DIR":     str(out_dir),
            "WANDB_RUN_NAME": run_label,
        }
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
    parser.add_argument(
        "--mode", required=True,
        choices=["samples", "traces", "train", "ood-val", "test"],
        help="Pipeline stage to run.",
    )
    parser.add_argument("--dry-run", "-n", action="store_true",
                        help="Print planned actions without submitting / running.")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even when outputs already exist on disk.")
    parser.add_argument("--eval-config", default="eval_distillation",
                        help="(test mode) Hydra config name for test eval (default: eval_distillation).")
    args = parser.parse_args()

    cfg = load_config()

    print_header(
        f"Label-source ablation — mode={args.mode}",
        f"teacher={cfg['teacher']['slug']}  student={cfg['student']['slug']}  "
        f"condition={cfg['condition']}  lr={cfg['learning_rate']}  seed={cfg['seed']}"
        + ("  [DRY RUN]" if args.dry_run else "")
        + ("  [FORCE]" if args.force else ""),
    )

    {
        "samples":  lambda: mode_samples(cfg, args.dry_run),
        "traces":   lambda: mode_traces(cfg, args.dry_run, args.force),
        "train":    lambda: mode_train(cfg, args.dry_run, args.force),
        "ood-val":  lambda: mode_ood_val(cfg, args.dry_run, args.force),
        "test":     lambda: mode_test(cfg, args.dry_run, args.force, args.eval_config),
    }[args.mode]()


if __name__ == "__main__":
    main()
