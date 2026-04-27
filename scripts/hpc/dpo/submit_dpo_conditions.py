#!/usr/bin/env python3
"""
Submit DPO condition stages on SLURM with dependencies.

Stages:
  hard_dpo      — DPO on hard-only annotated pairs
  judge_dpo     — DPO with LLM-judge intent rejecteds (GPT-OSS 120B on RTX 6000 Pro)

Both stages run independently.

Usage:
  python scripts/hpc/submit_dpo_conditions.py
  python scripts/hpc/submit_dpo_conditions.py --dry-run
  python scripts/hpc/submit_dpo_conditions.py --start-from judge_dpo --judge gpt-oss
  python scripts/hpc/submit_dpo_conditions.py --start-from judge_dpo --judge gemma-27b --judge-from-samples
  python scripts/hpc/submit_dpo_conditions.py --start-from judge_dpo --judge-from-samples --judge-epochs 1 --judge-beta 0.3

Judge presets (--judge <name>): gpt-oss | gemma-27b | gemma-27b-quant | llama-70b
Individual --judge-model / --judge-gpu-type / --judge-tensor-parallel flags override preset values.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from slurm_utils import create_logs_dir, print_header, print_job_summary


TEMPLATE = "scripts/hpc/dpo/dpo_conditions_template.sh"

STAGE_ORDER = ["hard_dpo", "judge_dpo"]

# Judge presets — each entry sets all GPU/model knobs for a known judge.
# Keys are the short names passed to --judge.
JUDGE_PRESETS: dict[str, dict] = {
    "gpt-oss": {
        "model":           "openai/gpt-oss-120b",
        "gpu_type":        "rtx_pro_6000",
        "tensor_parallel": 1,
        "mem":             "32G",
        "cpus":            4,
    },
    "gemma-27b": {
        "model":           "google/gemma-3-27b-it",
        "gpu_type":        "a100",
        "tensor_parallel": 2,
        "mem":             "32G",
        "cpus":            4,
    },
    "gemma-27b-quant": {
        "model":           "RedHatAI/gemma-3-27b-it-quantized.w4a16",
        "gpu_type":        "a100",
        "tensor_parallel": 1,
        "mem":             "32G",
        "cpus":            4,
    },
    "llama-70b": {
        "model":           "meta-llama/Llama-3.1-70B-Instruct",
        "gpu_type":        "a100",
        "tensor_parallel": 2,
        "mem":             "32G",
        "cpus":            4,
    },
}

DEPENDENCIES = {
    "hard_dpo":     None,
    "judge_dpo":    None,
}

# Stages that finish quickly (pair gen / data prep, no full DPO training loop)
SHORT_STAGES = {"hard_dpo"}


def submit_stage(
    stage: str,
    export_vars: dict[str, str],
    sbatch_opts: list[str],
    dependency: str | None = None,
    dry_run: bool = False,
) -> str:
    env = {**export_vars, "STAGE": stage}
    export_str = ",".join(f"{k}={v}" for k, v in env.items())
    cmd = ["sbatch", *sbatch_opts]
    if dependency:
        cmd.append(f"--dependency=afterok:{dependency}")
    cmd += [f"--export={export_str}", TEMPLATE]

    if dry_run:
        print("DRY RUN:", " ".join(shlex.quote(c) for c in cmd))
        return f"DRYRUN_{stage}"

    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return result.stdout.strip().split()[-1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # ── Pipeline control ───────────────────────────────────────────────────
    p.add_argument("--dry-run", action="store_true",
                   help="Print sbatch commands without submitting.")
    p.add_argument("--start-from", choices=STAGE_ORDER, default=None,
                   help="Skip all stages before this one (resume after failure).")
    p.add_argument("--skip-stages", nargs="+", choices=STAGE_ORDER, default=[],
                   metavar="STAGE", help="Stages to exclude from submission (e.g. judge_dpo).")
    p.add_argument("--force", action="store_true",
                   help="Re-run all pipeline steps, ignoring cached outputs.")
    p.add_argument("--force-from", type=int, default=None, metavar="N",
                   help="Re-run from pipeline step N onward.")
    # ── Model ──────────────────────────────────────────────────────────────
    p.add_argument("--base-model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--base-sft-adapter",
                   default="Jazhyc/llama-3.1-8b-sft-generation")
    p.add_argument("--judge", choices=list(JUDGE_PRESETS), default=None, metavar="PRESET",
                   help=f"Judge preset — sets model, GPU type, and tensor parallelism in one flag. "
                        f"Choices: {', '.join(JUDGE_PRESETS)}. "
                        f"Individual --judge-model / --judge-gpu-type / --judge-tensor-parallel "
                        f"override preset values when both are given.")
    p.add_argument("--judge-model", default=None,
                   help="HuggingFace model ID for the judge. Overrides --judge preset.")
    p.add_argument("--judge-tensor-parallel", type=int, default=None,
                   help="GPUs for judge tensor parallelism. Overrides --judge preset.")
    p.add_argument("--judge-gpu-type", default=None,
                   help="SLURM GPU resource name for the judge node (e.g. rtx_pro_6000, a100). "
                        "Overrides --judge preset.")
    p.add_argument("--seed", type=int, default=22)
    p.add_argument("--dpo-beta", type=float, default=0.1,
                   help="DPO beta for hard_dpo and judge_dpo stages.")

    # ── Judge-specific overrides ────────────────────────────────────────────
    p.add_argument("--judge-from-samples", action="store_true",
                   help="Skip sample generation for judge_dpo; reuse cached parsed_samples.jsonl.")
    p.add_argument("--judge-epochs", type=int, default=None,
                   help="Override DPO epochs for judge_dpo only.")
    p.add_argument("--judge-beta", type=float, default=None,
                   help="Override DPO beta for judge_dpo only.")

    # ── Hard DPO overrides ─────────────────────────────────────────────────
    p.add_argument("--unbalanced", action="store_true",
                   help="Train hard_dpo on unbalanced pairs (skip undersampling step 2). "
                        "Use with --harmful-weight to correct imbalance in the loss.")
    p.add_argument("--harmful-weight", type=float, default=1.4,
                   help="Loss up-weight for harmful pairs when --unbalanced is set "
                        "(n_safe/n_harmful ≈ 1.4 for annotated intents).")

    # ── Output path overrides ─────────────────────────────────────────────
    p.add_argument("--judge-pairs-dir", default=None,
                   help="Override JUDGE_PAIRS_DIR (default: data/dpo_pairs/judge).")
    p.add_argument("--judge-balanced-dir", default=None,
                   help="Override JUDGE_BALANCED_DIR.")
    p.add_argument("--judge-dpo-output", default=None,
                   help="Override JUDGE_DPO_OUTPUT (trained model save path).")

    # ── SLURM resources ────────────────────────────────────────────────────
    p.add_argument("--short-partition", default="gpushort")
    p.add_argument("--long-partition", default="gpumedium")
    p.add_argument("--short-time", default="04:00:00")
    p.add_argument("--long-time", default="10:00:00")
    p.add_argument("--short-mem", default="16G")
    p.add_argument("--long-mem", default="24G")
    p.add_argument("--short-cpus", type=int, default=2)
    p.add_argument("--long-cpus", type=int, default=2)
    p.add_argument("--all-gpushort", action="store_true",
                   help="Force all stages onto gpushort (may timeout for training stages).")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not Path(TEMPLATE).exists():
        raise FileNotFoundError(f"Template not found: {TEMPLATE}")

    print_header("DPO Conditions — SLURM Submission")
    if not args.dry_run:
        create_logs_dir()

    # Resolve judge preset → apply defaults, then let explicit flags override.
    preset = JUDGE_PRESETS.get(args.judge, JUDGE_PRESETS["gpt-oss"]) if args.judge else JUDGE_PRESETS["gpt-oss"]
    judge_model           = args.judge_model          or preset["model"]
    judge_tensor_parallel = args.judge_tensor_parallel if args.judge_tensor_parallel is not None else preset["tensor_parallel"]
    judge_gpu_type        = args.judge_gpu_type        or preset["gpu_type"]
    judge_mem             = preset["mem"]
    judge_cpus            = preset["cpus"]

    if args.judge or args.judge_model:
        print(f"  Judge        : {args.judge or 'custom'}")
        print(f"  Judge model  : {judge_model}")
        print(f"  Judge GPUs   : {judge_gpu_type}×{judge_tensor_parallel}")

    # Variables that the bash template cannot sensibly default (model IDs, seeds,
    # pipeline control flags). All path defaults live in the template itself.
    export_vars = {
        "PROJECT_ROOT":      str(Path.cwd()),
        "BASE_MODEL":        args.base_model,
        "BASE_SFT_ADAPTER":  args.base_sft_adapter,
        "JUDGE_MODEL":            judge_model,
        "JUDGE_TENSOR_PARALLEL":  str(judge_tensor_parallel),
        "JUDGE_GPU_TYPE":         judge_gpu_type,
        "SEED":              str(args.seed),
        "DPO_BETA":          str(args.dpo_beta),
        "FORCE":             "1" if args.force else "0",
        "FORCE_FROM":        "" if args.force_from is None else str(args.force_from),
        "JUDGE_FROM_SAMPLES": "1" if args.judge_from_samples else "",
        "JUDGE_EPOCHS":      "" if args.judge_epochs is None else str(args.judge_epochs),
        "JUDGE_BETA":        "" if args.judge_beta is None else str(args.judge_beta),
        "UNBALANCED":        "1" if args.unbalanced else "0",
        "HARMFUL_WEIGHT":    str(args.harmful_weight),
        **({} if args.judge_pairs_dir    is None else {"JUDGE_PAIRS_DIR":    args.judge_pairs_dir}),
        **({} if args.judge_balanced_dir is None else {"JUDGE_BALANCED_DIR": args.judge_balanced_dir}),
        **({} if args.judge_dpo_output   is None else {"JUDGE_DPO_OUTPUT":   args.judge_dpo_output}),
    }

    def sbatch_opts(stage: str) -> list[str]:
        use_short = args.all_gpushort or stage in SHORT_STAGES
        partition = args.short_partition if use_short else args.long_partition
        time      = args.short_time      if use_short else args.long_time
        mem       = args.short_mem       if use_short else args.long_mem
        cpus      = args.short_cpus      if use_short else args.long_cpus
        # judge_dpo loads a large judge model — resource needs come from preset
        gpus = f"{judge_gpu_type}:{judge_tensor_parallel}" if stage == "judge_dpo" else "a100:1"
        if stage == "judge_dpo":
            mem  = judge_mem
            cpus = judge_cpus
        return [
            f"--partition={partition}",
            f"--time={time}",
            f"--mem={mem}",
            f"--cpus-per-task={cpus}",
            f"--gpus-per-node={gpus}",
        ]

    stages = STAGE_ORDER[STAGE_ORDER.index(args.start_from):] if args.start_from else STAGE_ORDER
    stages = [s for s in stages if s not in args.skip_stages]

    dependencies = dict(DEPENDENCIES)

    submitted: dict[str, str] = {}
    jobs: list[tuple[str, str]] = []

    for stage in stages:
        dep_stage  = dependencies[stage]
        # If the dependency stage was skipped, look up its predecessor's job ID instead
        # so the chain still works.
        dep_job_id = submitted.get(dep_stage) if dep_stage else None
        job_id = submit_stage(
            stage, export_vars, sbatch_opts(stage),
            dependency=dep_job_id, dry_run=args.dry_run,
        )
        submitted[stage] = job_id
        jobs.append((stage, job_id))

    print("\nSubmitted stages:")
    for stage, job_id in jobs:
        print(f"  {stage:14s} -> {job_id}")

    if not args.dry_run:
        print_job_summary(jobs)


if __name__ == "__main__":
    main()
