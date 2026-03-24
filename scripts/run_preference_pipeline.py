"""
End-to-end preference-learning pipeline:

  Step 1 – generate_dpo_pairs.py   : sample model outputs, build DPO + contrastive pairs
  Step 2 – train_dpo.py            : DPO fine-tuning on top of the SFT adapter
  Step 3 – train_contrastive.py    : InfoNCE + KL fine-tuning on top of the SFT adapter
  Step 4 – eval_sft_baseline.py    : evaluate SFT adapter on held-out test set
  Step 5 – eval_sft_baseline.py    : evaluate DPO adapter on held-out test set
  Step 6 – eval_sft_baseline.py    : evaluate Contrastive adapter on held-out test set
  Step 7 – compare_models.py       : side-by-side comparison table

Caching: each step is skipped if its output file already exists.
Pass --force to re-run all steps, or --force-from=N to re-run from step N onward.

Usage (from project root):
    python scripts/run_preference_pipeline.py
    python scripts/run_preference_pipeline.py --force
    python scripts/run_preference_pipeline.py --force-from 4
    python scripts/run_preference_pipeline.py --skip-steps 2,3
    python scripts/run_preference_pipeline.py --wandb-project intention-jailbreak
"""

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cached(path: Path) -> bool:
    """Return True if path exists and (for jsonl) is non-empty."""
    if not path.exists():
        return False
    if path.suffix == ".jsonl":
        return path.stat().st_size > 0
    return True


# ---------------------------------------------------------------------------
# Step runners
# ---------------------------------------------------------------------------

def run(cmd: list[str], step_name: str) -> bool:
    """Run a subprocess from the project root. Returns True on success."""
    print(f"\n{'='*70}")
    print(f"  {step_name}")
    print(f"  $ {' '.join(cmd)}")
    print(f"{'='*70}")
    result = subprocess.run(cmd, cwd=str(ROOT))
    if result.returncode != 0:
        print(f"\n[FAILED] {step_name} (exit code {result.returncode})")
        return False
    print(f"\n[OK] {step_name}")
    return True


def step1_generate_pairs(args) -> bool:
    pairs_path = Path(args.pairs_dir) / "dpo_pairs.jsonl"
    contrastive_path = Path(args.pairs_dir) / "contrastive_pairs.jsonl"
    if _cached(pairs_path) and _cached(contrastive_path):
        print(f"\n[SKIP] Step 1 – pairs already exist at {args.pairs_dir}")
        return True
    cmd = [
        sys.executable, "scripts/generate_dpo_pairs.py",
        "--adapter-path",   args.adapter_path,
        "--base-model",     args.base_model,
        "--output-dir",     args.pairs_dir,
        "--temperature",    str(args.temperature),
        "--num-samples",    str(args.k_samples),
    ]
    return run(cmd, "Step 1 – Generate DPO / contrastive pairs")


def step2_train_dpo(args) -> bool:
    adapter_out = Path(args.dpo_output_dir + "_adapter") / "adapter_config.json"
    if _cached(adapter_out):
        print(f"\n[SKIP] Step 2 – DPO adapter already exists at {args.dpo_output_dir}_adapter")
        return True
    cmd = [
        sys.executable, "scripts/train_dpo.py",
        "--pairs-path",           str(Path(args.pairs_dir) / "dpo_pairs.jsonl"),
        "--adapter-path",         args.adapter_path,
        "--base-model",           args.base_model,
        "--output-dir",           args.dpo_output_dir,
        "--beta",                 str(args.dpo_beta),
        "--epochs",               str(args.epochs),
        "--learning-rate",        str(args.learning_rate),
        "--batch-size",           str(args.batch_size),
        "--gradient-accumulation", str(args.grad_accum),
        "--seed",                 str(args.seed),
        "--skip-eval",                          # evaluation done in step 5
        "--wandb-project",        args.wandb_project,
        "--wandb-run",            f"dpo-beta{args.dpo_beta}",
    ]
    return run(cmd, "Step 2 – DPO training")


def step3_train_contrastive(args) -> bool:
    best_adapter = Path(args.contrastive_output_dir + "_adapter_best") / "adapter_config.json"
    if _cached(best_adapter):
        print(f"\n[SKIP] Step 3 – Contrastive adapter already exists at "
              f"{args.contrastive_output_dir}_adapter_best")
        return True
    cmd = [
        sys.executable, "scripts/train_contrastive.py",
        "--pairs-path",     str(Path(args.pairs_dir) / "contrastive_pairs.jsonl"),
        "--adapter-path",   args.adapter_path,
        "--base-model",     args.base_model,
        "--output-dir",     args.contrastive_output_dir,
        "--kl-beta",        str(args.kl_beta),
        "--epochs",         str(args.epochs),
        "--learning-rate",  str(args.learning_rate),
        "--grad-accum",     str(args.grad_accum),
        "--seed",           str(args.seed),
        "--skip-eval",                      # evaluation done in step 6
        "--wandb-project",  args.wandb_project,
        "--wandb-run",      f"contrastive-kl-beta{args.kl_beta}",
    ]
    return run(cmd, "Step 3 – Contrastive (InfoNCE + KL) training")


def step4_eval_sft(args) -> bool:
    pred_path = Path(args.sft_pred_dir) / "test_predictions.jsonl"
    if _cached(pred_path):
        print(f"\n[SKIP] Step 4 – SFT predictions already exist at {pred_path}")
        return True
    cmd = [
        sys.executable, "scripts/eval_sft_baseline.py",
        "--adapter-path",   args.adapter_path,
        "--base-model",     args.base_model,
        "--output-dir",     args.sft_pred_dir,
    ]
    return run(cmd, "Step 4 – Evaluate SFT baseline")


def step5_eval_dpo(args) -> bool:
    pred_path = Path(args.dpo_output_dir) / "predictions" / "test_predictions.jsonl"
    if _cached(pred_path):
        print(f"\n[SKIP] Step 5 – DPO predictions already exist at {pred_path}")
        return True
    adapter_path = args.dpo_output_dir + "_adapter"
    cmd = [
        sys.executable, "scripts/eval_sft_baseline.py",
        "--adapter-path",   adapter_path,
        "--base-model",     args.base_model,
        "--output-dir",     str(Path(args.dpo_output_dir) / "predictions"),
    ]
    return run(cmd, "Step 5 – Evaluate DPO adapter")


def step6_eval_contrastive(args) -> bool:
    pred_path = Path(args.contrastive_output_dir) / "predictions" / "test_predictions.jsonl"
    if _cached(pred_path):
        print(f"\n[SKIP] Step 6 – Contrastive predictions already exist at {pred_path}")
        return True
    adapter_path = args.contrastive_output_dir + "_adapter_best"
    cmd = [
        sys.executable, "scripts/eval_sft_baseline.py",
        "--adapter-path",   adapter_path,
        "--base-model",     args.base_model,
        "--output-dir",     str(Path(args.contrastive_output_dir) / "predictions"),
    ]
    return run(cmd, "Step 6 – Evaluate Contrastive adapter")


def step7_compare(args) -> bool:
    sft_path         = str(Path(args.sft_pred_dir) / "test_predictions.jsonl")
    dpo_path         = str(Path(args.dpo_output_dir) / "predictions" / "test_predictions.jsonl")
    contrastive_path = str(Path(args.contrastive_output_dir) / "predictions" / "test_predictions.jsonl")
    cmd = [
        sys.executable, "scripts/compare_models.py",
        "--sft",         sft_path,
        "--dpo",         dpo_path,
        "--contrastive", contrastive_path,
    ]
    return run(cmd, "Step 7 – Compare SFT vs DPO vs Contrastive")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

STEPS = {
    1: step1_generate_pairs,
    2: step2_train_dpo,
    3: step3_train_contrastive,
    4: step4_eval_sft,
    5: step5_eval_dpo,
    6: step6_eval_contrastive,
    7: step7_compare,
}

STEP_NAMES = {
    1: "generate pairs",
    2: "train DPO",
    3: "train contrastive",
    4: "eval SFT",
    5: "eval DPO",
    6: "eval contrastive",
    7: "compare models",
}


def main():
    args = parse_args()

    # Build the set of steps to run
    skip = set(int(s) for s in args.skip_steps.split(",") if s.strip()) if args.skip_steps else set()
    force_from = args.force_from if args.force_from else (1 if args.force else None)

    results = {}

    for step_num, step_fn in STEPS.items():
        if step_num in skip:
            print(f"\n[SKIP] Step {step_num} – {STEP_NAMES[step_num]} (--skip-steps)")
            results[step_num] = "skipped"
            continue

        # --force / --force-from: delete cache sentinel so the step re-runs
        if force_from and step_num >= force_from:
            _clear_cache(step_num, args)

        ok = step_fn(args)
        results[step_num] = "ok" if ok else "FAILED"

        if not ok and step_num in {2, 3}:
            print(f"\n[ABORT] Training step {step_num} failed — skipping dependent eval steps.")
            break

    # Summary
    print(f"\n{'='*70}")
    print("  Pipeline summary")
    print(f"{'='*70}")
    for step_num, status in results.items():
        icon = "✓" if status == "ok" else ("–" if status == "skipped" else "✗")
        print(f"  {icon}  Step {step_num}: {STEP_NAMES[step_num]}  [{status}]")
    print()


def _clear_cache(step_num: int, args) -> None:
    """Remove the output sentinel for step_num so it re-runs."""
    sentinels = {
        1: [
            Path(args.pairs_dir) / "dpo_pairs.jsonl",
            Path(args.pairs_dir) / "contrastive_pairs.jsonl",
        ],
        2: [Path(args.dpo_output_dir + "_adapter") / "adapter_config.json"],
        3: [Path(args.contrastive_output_dir + "_adapter_best") / "adapter_config.json"],
        4: [Path(args.sft_pred_dir) / "test_predictions.jsonl"],
        5: [Path(args.dpo_output_dir) / "predictions" / "test_predictions.jsonl"],
        6: [Path(args.contrastive_output_dir) / "predictions" / "test_predictions.jsonl"],
        7: [Path("data/comparison/comparison_summary.json")],
    }
    for path in sentinels.get(step_num, []):
        if path.exists():
            path.unlink()
            print(f"  [force] Removed cache: {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Paths
    p.add_argument("--adapter-path",            default="trained_models/causal/hyperparam_sweep/lr_5e-05_e_5_adapter")
    p.add_argument("--base-model",              default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--pairs-dir",               default="data/dpo_pairs/train_t0.8")
    p.add_argument("--dpo-output-dir",          default="trained_models/causal/llama-dpo")
    p.add_argument("--contrastive-output-dir",  default="trained_models/causal/llama-contrastive")
    p.add_argument("--sft-pred-dir",            default="data/predictions/sft_baseline")

    # Pair generation
    p.add_argument("--temperature",  type=float, default=0.8)
    p.add_argument("--k-samples",    type=int,   default=5)

    # Training
    p.add_argument("--seed",            type=int,   default=22)
    p.add_argument("--epochs",          type=int,   default=3)
    p.add_argument("--learning-rate",   type=float, default=5e-5)
    p.add_argument("--batch-size",      type=int,   default=2)
    p.add_argument("--grad-accum",      type=int,   default=8)
    p.add_argument("--dpo-beta",        type=float, default=0.1)
    p.add_argument("--kl-beta",         type=float, default=0.1)

    # Pipeline control
    p.add_argument("--force",       action="store_true",
                   help="Re-run all steps, ignoring cached outputs.")
    p.add_argument("--force-from",  type=int, default=None, metavar="N",
                   help="Re-run from step N onward.")
    p.add_argument("--skip-steps",  type=str, default="",
                   help="Comma-separated step numbers to skip, e.g. '1,4'.")

    # W&B
    p.add_argument("--wandb-project", default="intention-jailbreak")

    return p.parse_args()


if __name__ == "__main__":
    main()
