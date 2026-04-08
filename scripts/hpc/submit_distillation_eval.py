#!/usr/bin/env python3
"""
Select best distillation adapters from W&B and submit eval jobs.

For each (teacher × student × condition) combination, queries the corresponding
W&B project for the run with the highest val_harm_f1, then submits a SLURM eval
job using that adapter on all test datasets.

Already-completed combinations (all 5 dataset output files present) are skipped
automatically. Use --force to re-evaluate them.

--cleanup removes every adapter in models/distillation-sweep/ that was not
selected as best across any combination. Combine with --dry-run to preview
deletions before committing.

Usage:
    python scripts/hpc/submit_distillation_eval.py --dry-run
    python scripts/hpc/submit_distillation_eval.py
    python scripts/hpc/submit_distillation_eval.py --force
    python scripts/hpc/submit_distillation_eval.py --traces-version v7
    python scripts/hpc/submit_distillation_eval.py --cleanup --dry-run
    python scripts/hpc/submit_distillation_eval.py --cleanup
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import wandb
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent))
from slurm_utils import create_logs_dir, submit_sbatch, print_header

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DISTILLATION_SWEEP_DIR = PROJECT_ROOT / "models" / "distillation-sweep"

TEMPLATE_PATH = "scripts/hpc/distillation_eval_template.sh"

# ── Sweep dimensions (must match submit_distillation_sweep.py) ────────────────

TEACHER_MODELS = [
    ("cyankiwi/gemma-4-31B-it-AWQ-4bit",             "cyankiwi-gemma-4-31b"),
    # ("moonshotai/kimi-k2.5",                        "moonshotai-kimi-k2.5"),
    # ("openai/gpt-oss-120b",                          "openai-gpt-oss-120b"),
    # ("RedHatAI/gemma-3-27b-it-quantized.w4a16",      "RedHatAI-gemma-3-27b-it-quantized.w4a16"),
]

STUDENT_MODELS = [
    #("meta-llama/Llama-3.1-8B-Instruct",        "llama-3.1-8b"),
    ("google/gemma-3-12b-it",                    "gemma-3-12b"),
    #("mistralai/Ministral-8B-Instruct-2410",     "ministral-8b"),
    #("Qwen/Qwen3-8B",                            "qwen3-8b"),
]

LEARNING_RATES = [1e-5, 2e-5, 5e-5, 1e-4, 2e-4]
CONDITIONS = ["without_intent", "with_intent"]
DEFAULT_TRACES_VERSION = "v6"

# Maps training condition → eval condition name used by eval_safety_classifier.py
CONDITION_TO_EVAL = {
    "without_intent": "finetuned_reasoning_classification",
    "with_intent":    "finetuned_reasoning_generation",
}

# Dataset output subdirectory names — must match eval_distillation.yaml
EVAL_DATASET_DIRS = [
    "annotated_intents",
    "wildguardmix",
    "xstest",
    "toxic_chat",
    "aegis",
]


# ── Completion check ──────────────────────────────────────────────────────────

def _is_done(output_dir: str) -> bool:
    """
    Return True if all eval dataset subdirectories contain at least one .jsonl
    file, indicating this combination has already been fully evaluated.
    """
    base = PROJECT_ROOT / output_dir
    return all(
        any((base / ds_dir).glob("*.jsonl"))
        for ds_dir in EVAL_DATASET_DIRS
    )


# ── W&B query ─────────────────────────────────────────────────────────────────

def fetch_best_adapter(
    teacher_slug: str,
    student_slug: str,
    condition: str,
    traces_version: str,
) -> tuple[Path | None, float | None]:
    """
    Query the W&B project for this (teacher × student) pair and return the
    adapter with the highest val_harm_f1 for the given condition.

    Falls back to deriving the adapter path from run config when adapter_path
    is absent from the W&B summary (older runs logged it only to the filesystem).

    Returns (adapter_path, val_harm_f1) or (None, None) if nothing qualifies.
    """
    project = f"distillation-sweep-{student_slug}--{teacher_slug}"
    api = wandb.Api()
    try:
        runs = api.runs(path=project, filters={"state": "finished"})
    except Exception as e:
        print(f"    [warn] Could not query W&B project '{project}': {e}")
        return None, None

    best_f1 = -1.0
    best_adapter = None

    for run in runs:
        # Filter by condition (summary flag, then run config)
        run_condition = (
            run.summary.get("reasoning_traces_condition")
            or run.config.get("data", {}).get("reasoning_traces_condition")
        )
        if run_condition != condition:
            continue

        # Filter by traces version if present in run name or config
        run_version = run.config.get("data", {}).get("traces_version")
        if run_version is not None and run_version != traces_version:
            continue

        f1 = run.summary.get("val_harm_f1")
        if f1 is None:
            continue

        # Resolve adapter path (summary preferred; fall back to config derivation)
        adapter_path_str = run.summary.get("adapter_path")
        if not adapter_path_str:
            model_save_dir = run.config.get("paths", {}).get("model_save_dir")
            if not model_save_dir:
                continue
            adapter_path_str = model_save_dir + "_adapter"

        adapter_path = Path(adapter_path_str)
        if not adapter_path.is_absolute():
            adapter_path = PROJECT_ROOT / adapter_path

        if not adapter_path.exists():
            continue

        if f1 > best_f1:
            best_f1 = f1
            best_adapter = adapter_path

    if best_adapter is None:
        return None, None

    return best_adapter, best_f1


# ── Cleanup ───────────────────────────────────────────────────────────────────

def cleanup_suboptimal(best_adapters: set[Path], dry_run: bool):
    """
    Delete every directory in models/distillation-sweep/ that is not in
    best_adapters. This removes suboptimal LR adapters and stale version dirs.
    """
    if not DISTILLATION_SWEEP_DIR.exists():
        print("  models/distillation-sweep/ does not exist, nothing to clean.")
        return

    all_dirs = [p for p in DISTILLATION_SWEEP_DIR.iterdir() if p.is_dir()]
    to_delete = [p for p in all_dirs if p not in best_adapters]
    to_keep   = [p for p in all_dirs if p in best_adapters]

    print(f"\n  Cleanup: {len(to_keep)} adapters kept, {len(to_delete)} to delete")

    for path in sorted(to_delete):
        if dry_run:
            print(f"  [dry-run] would delete: {path.name}")
        else:
            shutil.rmtree(path)
            print(f"  deleted: {path.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dry-run", "-n", action="store_true",
        help="Print selected adapters and planned actions without submitting jobs or deleting files.",
    )
    parser.add_argument(
        "--cleanup", action="store_true",
        help="Delete all adapters in models/distillation-sweep/ that were not selected as best.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-submit eval jobs even for combinations that already have output files.",
    )
    parser.add_argument(
        "--traces-version", default=DEFAULT_TRACES_VERSION, metavar="VERSION",
        help=f"Reasoning traces version to select adapters for (default: {DEFAULT_TRACES_VERSION}).",
    )
    args = parser.parse_args()

    traces_version = args.traces_version

    n_combos = len(TEACHER_MODELS) * len(STUDENT_MODELS) * len(CONDITIONS)
    print_header(
        "Distillation Eval — Best Adapter Selection",
        f"{len(TEACHER_MODELS)} teachers × {len(STUDENT_MODELS)} students × "
        f"{len(CONDITIONS)} conditions = {n_combos} combinations"
        f"  |  traces: {traces_version}"
        + (" [DRY RUN]" if args.dry_run else "")
        + (" [FORCE]" if args.force else ""),
    )

    if not args.dry_run:
        create_logs_dir()

    submitted = []
    not_found = []
    skipped = []
    best_adapters: set[Path] = set()

    for teacher_hf, teacher_slug in TEACHER_MODELS:
        print(f"\n── Teacher: {teacher_slug} ──")
        for student_hf, student_slug in STUDENT_MODELS:
            for condition in CONDITIONS:
                label = f"{student_slug} / {condition}"
                print(f"\n  [{label}]")

                best_adapter, best_f1 = fetch_best_adapter(
                    teacher_slug, student_slug, condition, traces_version,
                )

                if best_adapter is None:
                    print(f"    [skip] No qualifying run found in W&B")
                    not_found.append(f"{teacher_slug}/{student_slug}/{condition}")
                    continue

                best_adapters.add(best_adapter)
                eval_condition = CONDITION_TO_EVAL[condition]
                output_dir = (
                    f"data/safety_experiment/distillation"
                    f"/{teacher_slug}/{student_slug}/{condition}"
                )
                run_label = f"distill--{teacher_slug}--{student_slug}--{condition}"

                print(f"    adapter:   {best_adapter.name}")
                print(f"    f1:        {best_f1:.4f}")
                print(f"    condition: {eval_condition}")

                # Skip already-completed combinations
                if not args.force and _is_done(output_dir):
                    print(f"    [done] Output files already present — skipping (use --force to re-run)")
                    skipped.append(label)
                    continue

                if args.dry_run:
                    print(f"    [dry-run] would submit eval job")
                    continue

                export_vars = {
                    "STUDENT_MODEL":  student_hf,
                    "ADAPTER_PATH":   str(best_adapter),
                    "EVAL_CONDITION": eval_condition,
                    "OUTPUT_DIR":     output_dir,
                    "WANDB_RUN_NAME": run_label,
                }
                try:
                    job_id = submit_sbatch(TEMPLATE_PATH, export_vars)
                    print(f"    ✓ Submitted job {job_id}")
                    submitted.append((label, job_id))
                except subprocess.CalledProcessError as e:
                    print(f"    ✗ Submission failed: {e.stderr}")

    print("\n" + "=" * 60)
    if args.dry_run:
        found = n_combos - len(not_found)
        print(f"Dry run: {found}/{n_combos} adapters found, {len(skipped)} already done.")
    else:
        print(f"Submitted {len(submitted)} eval jobs  |  {len(skipped)} skipped (done)  |  {len(not_found)} not found.")
    if not_found:
        print(f"Not found ({len(not_found)}):")
        for label in not_found:
            print(f"  {label}")
    if skipped:
        print(f"Already done ({len(skipped)}):")
        for label in skipped:
            print(f"  {label}")
    print("=" * 60)

    if args.cleanup:
        print("\n── Adapter cleanup ──")
        if not best_adapters and not args.dry_run:
            print("  No best adapters identified — aborting cleanup to avoid data loss.")
        else:
            cleanup_suboptimal(best_adapters, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
