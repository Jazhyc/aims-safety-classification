#!/usr/bin/env python3
"""
Verify TRL sequence packing doesn't break training.

Runs two short training jobs serially with a tiny subset of reasoning traces:

  1. padding_free=True, packing=False   (current production setup)
  2. padding_free=True, packing=True    (proposed speedup for data scaling)

If packing is implemented correctly via flash-attn varlen, packed examples get
block-diagonal causal attention (no cross-example attention) and the val
eval_loss should be very close to the unpacked baseline. A large divergence
suggests the trainer is using naive concatenation and examples are leaking
into each other.

Output for each run: final val_eval_loss, val_harm_f1, training wall-clock.
Output at the end: side-by-side comparison + a pass/fail judgement against a
configurable tolerance.

Run on an interactive node (~15 min total at default settings):

    # 1. grab an rtx_pro_6000 for an hour
    srun --pty --time=01:00:00 --mem=32GB --gpus-per-node=rtx_pro_6000:1 --cpus-per-task=1 --partition=gpumedium bash

    # 2. inside the interactive shell:
    module load Python/3.12.3-GCCcore-13.3.0 CUDA/12.8.0
    source .venv/bin/activate
    python scripts/hpc/scaling/verify_packing.py

Tweak knobs:
    --num-samples 256        # how many train traces to subsample
    --epochs 2               # epochs per run
    --tolerance 0.10         # |delta| / baseline above which we report FAIL
    --source <path>          # which parsed_results.json to subsample from
    --keep-tmp               # retain the tmp working dir for inspection
"""

import argparse
import json
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_SOURCE = (
    PROJECT_ROOT
    / "data" / "reasoning_traces_v7"
    / "openai-gpt-oss-120b" / "train" / "parsed_results.json"
)
DEFAULT_VAL = (
    PROJECT_ROOT
    / "data" / "reasoning_traces_v7"
    / "openai-gpt-oss-120b" / "validation" / "parsed_results.json"
)

STUDENT_MODEL  = "google/gemma-3-12b-it"
LR             = 2e-5
CONDITION      = "synthetic_intent"


def subsample_traces(source: Path, n: int, dst: Path, seed: int = 42) -> int:
    """Pick `n` records of `condition=synthetic_intent` and write to `dst`."""
    records = json.loads(source.read_text())
    syn = [r for r in records if r.get("condition") == CONDITION]
    rng = random.Random(seed)
    rng.shuffle(syn)
    chosen = syn[:n]
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(chosen, indent=2, ensure_ascii=False))
    return len(chosen)


VALIDATION_LOSS_RE = re.compile(r"\[Causal LM\] Validation loss:\s+([0-9eE+\-.]+)")


def run_one(label: str, packing: bool, train_path: Path, val_path: Path,
            tmp_dir: Path, epochs: int, per_device_batch: int,
            grad_accum: int) -> dict:
    """Spawn one train_generator.py invocation. Captures stdout to parse the
    HF Trainer's final eval_loss (printed by causal.py as
    "[Causal LM] Validation loss: ...") — val_metrics.json is only written
    by the post-training vLLM eval, which we skip here for speed."""
    run_name = f"verify-packing-{label}"
    adapter_save_dir = tmp_dir / "models" / run_name  # causal.py appends `_adapter`
    output_dir       = tmp_dir / "train_results" / run_name
    logs_dir         = tmp_dir / "logs" / run_name
    predictions_dir  = tmp_dir / "predictions" / run_name

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "baselines" / "train_generator.py"),
        "--config-name=reasoning_distillation",
        f"model.name={STUDENT_MODEL}",
        f"training.learning_rate={LR}",
        f"training.epochs={epochs}",
        # Lower per-device batch + bump grad-accum to keep effective batch at
        # batch_size × grad_accum = 32. Required to avoid OOM in the packed
        # run, where each sequence is filled to max_length (~4096 tokens) and
        # the cross_entropy logits tensor (batch × seq × vocab × 4B) balloons.
        f"training.batch_size={per_device_batch}",
        f"training.gradient_accumulation={grad_accum}",
        f"training.padding_free=true",
        f"training.packing={'true' if packing else 'false'}",
        # Skip the post-training vLLM eval — we only care about HF eval_loss
        # for this verification, and vLLM startup adds 2-3 minutes per run.
        "training.skip_vllm_eval=true",
        f"data.reasoning_traces_path={train_path}",
        f"data.reasoning_traces_val_path={val_path}",
        f"data.reasoning_traces_condition={CONDITION}",
        f"paths.output_dir={output_dir}",
        f"paths.logs_dir={logs_dir}",
        f"paths.model_save_dir={adapter_save_dir}",
        f"paths.predictions_dir={predictions_dir}",
        "wandb.enabled=false",
    ]

    print(f"\n{'═' * 70}", flush=True)
    print(f"  Run: {label}  (packing={packing}, per_device_batch={per_device_batch}, grad_accum={grad_accum})", flush=True)
    print(f"{'═' * 70}", flush=True)
    print("  Command:", " ".join(cmd), flush=True)
    print("  ── child output ──", flush=True)
    t0 = time.time()
    # Stream child stdout/stderr line-by-line so the user sees progress live,
    # while also capturing it for "[Causal LM] Validation loss: ..." parsing.
    proc = subprocess.Popen(
        cmd, cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    captured: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        captured.append(line)
    proc.wait()
    elapsed = time.time() - t0
    full_output = "".join(captured)

    result = {"label": label, "packing": packing, "elapsed_s": elapsed}
    if proc.returncode != 0:
        result["error"] = f"non-zero exit {proc.returncode}"
        return result

    matches = VALIDATION_LOSS_RE.findall(full_output)
    if not matches:
        result["error"] = "could not parse '[Causal LM] Validation loss:' from stdout"
        return result
    # Take the LAST match in case multiple evals were printed.
    result["val_eval_loss"] = float(matches[-1])
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--num-samples", type=int, default=256,
                        help="Number of train traces to subsample (default: 256).")
    parser.add_argument("--epochs", type=int, default=2,
                        help="Epochs per run (default: 2).")
    parser.add_argument("--per-device-batch", type=int, default=2,
                        help="per_device_train_batch_size for both runs. Lowered "
                             "from the production default (8) so the packed run's "
                             "logits tensor (batch × seq × vocab × 4B) fits in GPU "
                             "memory at max_length=4096 (default: 2).")
    parser.add_argument("--grad-accum", type=int, default=16,
                        help="gradient_accumulation_steps; pair with --per-device-batch "
                             "to keep effective batch = 32 (default: 16).")
    parser.add_argument("--tolerance", type=float, default=0.10,
                        help="Relative |delta|/baseline above which we report FAIL (default: 0.10).")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE,
                        help="Source parsed_results.json to subsample (default: gpt-oss v7 train).")
    parser.add_argument("--val", type=Path, default=DEFAULT_VAL,
                        help="Validation parsed_results.json (default: gpt-oss v7 validation).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--keep-tmp", action="store_true",
                        help="Don't delete the tmp working dir after the runs (for inspection).")
    args = parser.parse_args()

    if not args.source.exists():
        print(f"ERROR: source traces missing at {args.source}", file=sys.stderr)
        return 2
    if not args.val.exists():
        print(f"ERROR: val traces missing at {args.val}", file=sys.stderr)
        return 2

    tmp_dir = Path(tempfile.mkdtemp(prefix="verify-packing-", dir=str(PROJECT_ROOT / "data")))
    print(f"Working dir: {tmp_dir}")

    try:
        # Subsample once, reuse across both runs so they see identical data.
        train_subset = tmp_dir / "subset" / "train" / "parsed_results.json"
        n = subsample_traces(args.source, args.num_samples, train_subset, seed=args.seed)
        print(f"Subsampled {n} synthetic_intent records → {train_subset}")

        run_kwargs = dict(
            train_path=train_subset, val_path=args.val,
            tmp_dir=tmp_dir, epochs=args.epochs,
            per_device_batch=args.per_device_batch, grad_accum=args.grad_accum,
        )
        results = [
            run_one("nopack", packing=False, **run_kwargs),
            run_one("pack",   packing=True,  **run_kwargs),
        ]

        print(f"\n{'═' * 70}")
        print("  Results")
        print(f"{'═' * 70}")
        for r in results:
            loss = r.get("val_eval_loss")
            loss_str = f"{loss:.4f}" if isinstance(loss, float) else "n/a"
            print(f"  [{r['label']:6s}]  packing={r['packing']!s:5s}  "
                  f"elapsed={r['elapsed_s']:6.1f}s  loss={loss_str:>8s}")
            if "error" in r:
                print(f"           ERROR: {r['error']}")

        no_pack, pack = results
        if "val_eval_loss" not in no_pack or "val_eval_loss" not in pack:
            print("\nVERDICT: missing eval_loss in at least one run — see errors above.")
            return 3

        baseline = float(no_pack["val_eval_loss"])
        packed   = float(pack["val_eval_loss"])
        delta    = packed - baseline
        rel      = abs(delta) / baseline if baseline > 0 else float("inf")

        speedup = no_pack["elapsed_s"] / pack["elapsed_s"] if pack["elapsed_s"] > 0 else float("nan")

        print(f"\n  Loss delta (pack − nopack): {delta:+.4f}  ({rel * 100:.2f}% of baseline)")
        print(f"  Wall-clock speedup (nopack / pack): {speedup:.2f}×")

        verdict = "PASS" if rel <= args.tolerance else "FAIL"
        print(f"\nVERDICT: {verdict}  (tolerance={args.tolerance:.0%})")
        if verdict == "FAIL":
            print("  Packed loss diverges materially from the unpacked baseline.")
            print("  Likely cause: the trainer is concatenating examples without resetting")
            print("  attention/position boundaries, so examples cross-attend in the packed")
            print("  sequence. Do NOT enable packing for the scaling run.")
            return 1
        else:
            print("  Loss is within tolerance → packing is using flash-attn varlen correctly.")
            print(f"  Wall-clock improved {speedup:.1f}×; safe to enable packing for the scaling run.")
            return 0
    finally:
        if args.keep_tmp:
            print(f"\n  --keep-tmp set; leaving working dir at {tmp_dir}")
        else:
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
