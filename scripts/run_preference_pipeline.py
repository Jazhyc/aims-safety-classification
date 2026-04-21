"""
End-to-end preference-learning pipeline:

  Step 1 – generate_dpo_pairs.py   : sample model outputs, build DPO pairs
  Step 2 – balance_dpo_pairs.py    : undersample to 50/50 harm distribution
  Step 3 – train_dpo.py            : DPO fine-tuning on top of the SFT adapter
  Step 4 – eval_sft_baseline.py    : evaluate SFT adapter on held-out test set
  Step 5 – eval_safety_classifier.py: evaluate DPO adapter on all benchmarks
  Step 6 – compare_models.py       : side-by-side comparison table

Augmentation mode (--augment):
  Inserts three pre-processing steps before the main pipeline to enrich the
  training set with easy, high-confidence WildGuardMix examples:

  Pre A – filter_wildguard_easy.py     : score WildGuardMix with ModernBERT,
                                         keep high-confidence examples
  Pre B – generate_dpo_pairs.py        : generate DPO pairs for easy examples
           --from-jsonl <candidates>     (uses T=0 synthetic intent as chosen)
  Pre C – merge_dpo_pairs.py           : merge annotated + WildGuardMix pairs;
                                         extract SFT training examples

  Steps 1–6 then run on the merged pairs. Also writes
  data/wildguard_easy/sft_examples.jsonl for SFT augmentation retraining.

Caching: each step is skipped if its output file already exists.
Pass --force to re-run all steps, or --force-from=N to re-run from step N onward.
Use --force-augment to re-run only the augmentation pre-steps A/B/C.

Usage (from project root):
    python scripts/run_preference_pipeline.py
    python scripts/run_preference_pipeline.py --force
    python scripts/run_preference_pipeline.py --force-from 4
    python scripts/run_preference_pipeline.py --skip-steps 2,3
    python scripts/run_preference_pipeline.py --wandb-project intention-jailbreak
    python scripts/run_preference_pipeline.py --augment
    python scripts/run_preference_pipeline.py --augment --force-augment
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
    pairs_path = Path(args.pairs_dir) / args.pairs_filename
    if _cached(pairs_path):
        print(f"\n[SKIP] Step 1 – pairs already exist at {args.pairs_dir}")
        return True
    cmd = [
        sys.executable, "scripts/generate_dpo_pairs.py",
        "--adapter-path",   args.adapter_path,
        "--base-model",     args.base_model,
        "--output-dir",     args.pairs_dir,
        "--temperature",    str(args.temperature),
        "--num-samples",    str(args.k_samples),
        "--max-model-len",  str(args.max_model_len),
        "--seed",           str(args.seed),
    ]
    if args.from_samples:
        cmd += ["--from-samples", args.from_samples]
    if args.intent_filter:
        cmd += ["--intent-filter", "--judge-model", args.judge_model]
    return run(cmd, "Step 1 – Generate DPO pairs")


def step2_balance_pairs(args) -> bool:
    input_dir     = args.augmented_pairs_dir if args.augment else args.pairs_dir
    balanced_path = Path(args.balanced_pairs_dir) / "dpo_pairs.jsonl"
    if _cached(balanced_path):
        print(f"\n[SKIP] Step 2 – balanced pairs already exist at {balanced_path}")
        return True
    cmd = [
        sys.executable, "scripts/balance_dpo_pairs.py",
        "--input",  str(Path(input_dir) / args.pairs_filename),
        "--output", str(balanced_path),
        "--seed",   str(args.seed),
    ]
    return run(cmd, "Step 2 – Balance DPO pairs (undersample majority class)")


def step3_train_dpo(args) -> bool:
    adapter_out = Path(args.dpo_output_dir + "_adapter") / "adapter_config.json"
    if _cached(adapter_out):
        print(f"\n[SKIP] Step 3 – DPO adapter already exists at {args.dpo_output_dir}_adapter")
        return True
    # Unbalanced mode: train directly on raw pairs (no undersampling); use
    # harmful_weight to correct class imbalance in the loss instead.
    pairs_dir = args.pairs_dir if args.unbalanced else args.balanced_pairs_dir
    cmd = [
        sys.executable, "scripts/train_dpo.py",
        "--pairs-path",           str(Path(pairs_dir) / "dpo_pairs.jsonl"),
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
    if args.harmful_weight != 1.0:
        cmd += ["--harmful-weight", str(args.harmful_weight)]
    return run(cmd, "Step 3 – DPO training")


def step4_eval_sft(args) -> bool:
    pred_path = Path(args.sft_pred_dir) / "test_predictions.jsonl"
    if _cached(pred_path):
        print(f"\n[SKIP] Step 4 – SFT predictions already exist at {pred_path}")
        return True
    cmd = [
        sys.executable, "scripts/baselines/eval_sft_baseline.py",
        "--adapter-path",   args.adapter_path,
        "--base-model",     args.base_model,
        "--output-dir",     args.sft_pred_dir,
    ]
    return run(cmd, "Step 4 – Evaluate SFT baseline")


def step5_eval_dpo(args) -> bool:
    pred_dir = Path(args.dpo_output_dir) / "predictions"
    model_slug = args.base_model.replace("/", "_")
    pred_path = pred_dir / "annotated_intents" / f"{model_slug}_finetuned_generation.jsonl"
    if _cached(pred_path):
        print(f"\n[SKIP] Step 5 – DPO predictions already exist at {pred_path}")
        return True
    adapter_path = args.dpo_output_dir + "_adapter"
    cmd = [
        sys.executable, "scripts/eval_safety_classifier.py",
        "--config-name=eval_dpo_condition",
        f"finetuned.generation_adapter={adapter_path}",
        f"paths.output_dir={pred_dir}",
        f"model.name={args.base_model}",
    ]
    return run(cmd, "Step 5 – Evaluate DPO adapter")


def step6_compare(args) -> bool:
    sft_dir = str(args.sft_pred_dir)
    dpo_dir = str(Path(args.dpo_output_dir) / "predictions")
    cmd = [
        sys.executable, "scripts/baselines/compare_models.py",
        "--sft",       sft_dir,
        "--dpo-hard",  dpo_dir,
        "--no-intent",
    ]
    return run(cmd, "Step 6 – Compare SFT vs DPO")


# ---------------------------------------------------------------------------
# Augmentation pre-steps (A / B / C)
# ---------------------------------------------------------------------------

def step_a_filter_wildguard(args) -> bool:
    candidates = Path(args.wildguard_candidates)
    if _cached(candidates):
        print(f"\n[SKIP] Pre A – WildGuardMix candidates already exist at {candidates}")
        return True
    cmd = [
        sys.executable, "scripts/dataset_analysis/filter_wildguard_easy.py",
        "--model-path", args.wildguard_model_path,
        "--output-dir", str(candidates.parent),
    ]
    return run(cmd, "Pre A – Filter WildGuardMix easy examples")


def step_b_generate_wildguard_pairs(args) -> bool:
    pairs_path = Path(args.wildguard_pairs_dir) / "dpo_pairs.jsonl"
    if _cached(pairs_path):
        print(f"\n[SKIP] Pre B – WildGuardMix DPO pairs already exist at {args.wildguard_pairs_dir}")
        return True
    cmd = [
        sys.executable, "scripts/generate_dpo_pairs.py",
        "--from-jsonl",     args.wildguard_candidates,
        "--adapter-path",   args.adapter_path,
        "--base-model",     args.base_model,
        "--output-dir",     args.wildguard_pairs_dir,
        "--temperature",    str(args.temperature),
        "--num-samples",    str(args.k_samples),
        "--max-model-len",  str(args.max_model_len),
        "--seed",           str(args.seed),
    ]
    return run(cmd, "Pre B – Generate DPO pairs for WildGuardMix easy examples")


def step_c_merge_pairs(args) -> bool:
    merged_path = Path(args.augmented_pairs_dir) / "dpo_pairs.jsonl"
    if _cached(merged_path):
        print(f"\n[SKIP] Pre C – Merged pairs already exist at {args.augmented_pairs_dir}")
        return True
    cmd = [
        sys.executable, "scripts/merge_dpo_pairs.py",
        "--annotated-dir", args.pairs_dir,
        "--wildguard-dir", args.wildguard_pairs_dir,
        "--output-dir",    args.augmented_pairs_dir,
        "--sft-output",    args.sft_examples_output,
    ]
    return run(cmd, "Pre C – Merge annotated + WildGuardMix DPO pairs")


def run_augment_preprocessing(args) -> bool:
    for name, fn in [
        ("A – filter wildguard", step_a_filter_wildguard),
        ("B – generate wildguard pairs", step_b_generate_wildguard_pairs),
        ("C – merge pairs", step_c_merge_pairs),
    ]:
        if not fn(args):
            print(f"\n[ABORT] Augmentation pre-step {name} failed.")
            return False
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

STEPS = {
    1: step1_generate_pairs,
    2: step2_balance_pairs,
    3: step3_train_dpo,
    4: step4_eval_sft,
    5: step5_eval_dpo,
    6: step6_compare,
}

STEP_NAMES = {
    1: "generate pairs",
    2: "balance pairs",
    3: "train DPO",
    4: "eval SFT",
    5: "eval DPO",
    6: "compare models",
}


def main():
    args = parse_args()

    # Augmentation pre-processing (runs before main pipeline)
    if args.augment:
        print("\n=== Augmentation mode: running pre-processing steps A / B / C ===")
        if args.force_augment:
            _clear_augment_cache(args)
        if not run_augment_preprocessing(args):
            sys.exit(1)

    # Build the set of steps to run
    skip = set(int(s) for s in args.skip_steps.split(",") if s.strip()) if args.skip_steps else set()
    # --force starts from step 2 to protect pairs; --force-pairs or --force-from 1
    # are the only ways to regenerate pairs intentionally.
    force_from = args.force_from if args.force_from else (1 if args.force_pairs else (2 if args.force else None))

    results = {}

    for step_num, step_fn in STEPS.items():
        if step_num in skip:
            print(f"\n[SKIP] Step {step_num} – {STEP_NAMES[step_num]} (--skip-steps)")
            results[step_num] = "skipped"
            continue

        # --force / --force-from: delete cache sentinel so the step re-runs.
        # Step 1 (pair generation) is only cleared when --force-pairs is set
        # or when force_from explicitly targets step 1.
        if force_from and step_num >= force_from:
            if step_num == 1 and not args.force_pairs and force_from != 1:
                pass  # protect pairs from accidental regeneration
            else:
                _clear_cache(step_num, args)

        ok = step_fn(args)
        results[step_num] = "ok" if ok else "FAILED"

        if not ok and step_num == 3:
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
        1: [Path(args.pairs_dir) / "dpo_pairs.jsonl"],
        2: [Path(args.balanced_pairs_dir) / "dpo_pairs.jsonl"],
        3: [Path(args.dpo_output_dir + "_adapter") / "adapter_config.json"],
        4: [Path(args.sft_pred_dir) / "test_predictions.jsonl"],
        5: [Path(args.dpo_output_dir) / "predictions" / "annotated_intents" /
            f"{args.base_model.replace('/', '_')}_finetuned_generation.jsonl"],
        6: [Path("data/comparison/comparison_summary.json")],
    }
    for path in sentinels.get(step_num, []):
        if path.exists():
            path.unlink()
            print(f"  [force] Removed cache: {path}")


def _clear_augment_cache(args) -> None:
    """Remove augmentation pre-step cache sentinels."""
    sentinels = [
        Path(args.wildguard_candidates),
        Path(args.wildguard_pairs_dir) / "dpo_pairs.jsonl",
        Path(args.augmented_pairs_dir) / "dpo_pairs.jsonl",
    ]
    for path in sentinels:
        if path.exists():
            path.unlink()
            print(f"  [force-augment] Removed cache: {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Paths
    p.add_argument("--adapter-path",       default="trained_models/causal/hyperparam_sweep/lr_5e-05_e_5_adapter")
    p.add_argument("--base-model",         default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--pairs-dir",          default="data/dpo_pairs/train_t0.8")
    p.add_argument("--pairs-filename",     default="dpo_pairs.jsonl",
                   help="Filename within --pairs-dir to use as training pairs. "
                        "Use 'dpo_pairs_decent.jsonl' for bad+decent match conditions.")
    p.add_argument("--balanced-pairs-dir", default="data/dpo_pairs/train_t0.8_balanced")
    p.add_argument("--dpo-output-dir",     default="trained_models/causal/llama-dpo")
    p.add_argument("--sft-pred-dir",       default="data/predictions/sft_baseline")

    # Augmentation paths (only used with --augment)
    p.add_argument("--wildguard-candidates",  default="data/wildguard_easy/candidates.jsonl",
                   help="Output of filter_wildguard_easy.py.")
    p.add_argument("--wildguard-model-path",  default="models/modernbert-ensemble/final_model",
                   help="HF model ID or local path used by filter_wildguard_easy.py.")
    p.add_argument("--wildguard-pairs-dir",   default="data/wildguard_easy/dpo_pairs",
                   help="DPO pairs generated from WildGuardMix easy examples.")
    p.add_argument("--augmented-pairs-dir",   default="data/dpo_pairs/augmented",
                   help="Merged (annotated + WildGuardMix) pairs directory.")
    p.add_argument("--sft-examples-output",   default="data/wildguard_easy/sft_examples.jsonl",
                   help="SFT training examples extracted from WildGuardMix DPO pairs.")

    # Pair generation
    p.add_argument("--temperature",  type=float, default=0.8)
    p.add_argument("--k-samples",    type=int,   default=5)
    p.add_argument("--max-model-len", type=int,  default=4096,
                   help="vLLM max context length used in pair generation.")
    p.add_argument("--from-samples", default=None,
                   help="Path to a pre-generated parsed_samples.jsonl to reuse instead of "
                        "running a fresh T=0.8 generation pass. Ensures all conditions "
                        "start from the same base samples for fair comparison.")
    p.add_argument("--intent-filter", action="store_true",
                   help="Run LLM judge on correct-label samples and add bad_match "
                        "outputs as extra DPO rejected pairs (condition 3).")
    p.add_argument("--judge-model",  default="google/gemma-3-27b-it",
                   help="Judge model for --intent-filter.")

    # Training
    p.add_argument("--seed",            type=int,   default=22)
    p.add_argument("--epochs",          type=int,   default=3)
    p.add_argument("--learning-rate",   type=float, default=5e-5)
    p.add_argument("--batch-size",      type=int,   default=2)
    p.add_argument("--grad-accum",      type=int,   default=8)
    p.add_argument("--dpo-beta",        type=float, default=0.1)
    p.add_argument("--harmful-weight",  type=float, default=1.0,
                   help="Up-weight harmful pairs in DPO loss to correct class imbalance "
                        "(use n_safe/n_harmful, e.g. 1.4 for unbalanced annotated pairs). "
                        "Passed to train_dpo.py --harmful-weight.")
    p.add_argument("--unbalanced",      action="store_true",
                   help="Skip step 2 (pair balancing) and train on raw pairs with "
                        "--harmful-weight to correct imbalance instead.")

    # Pipeline control
    p.add_argument("--augment",       action="store_true",
                   help="Enable WildGuardMix augmentation pre-steps A/B/C.")
    p.add_argument("--force-augment", action="store_true",
                   help="Re-run augmentation pre-steps A/B/C even if cached.")
    p.add_argument("--force",         action="store_true",
                   help="Re-run steps 2–5 (training + eval), ignoring cached outputs. "
                        "Does NOT regenerate pairs (step 1) — use --force-pairs for that.")
    p.add_argument("--force-pairs",   action="store_true",
                   help="Force regeneration of DPO pairs (step 1). "
                        "Only use this intentionally — pairs are stochastic and "
                        "changing them makes results incomparable across runs.")
    p.add_argument("--force-from",    type=int, default=None, metavar="N",
                   help="Re-run from step N onward.")
    p.add_argument("--skip-steps",    type=str, default="",
                   help="Comma-separated step numbers to skip, e.g. '1,4'.")

    # W&B
    p.add_argument("--wandb-project", default="intention-jailbreak")

    return p.parse_args()


if __name__ == "__main__":
    main()
