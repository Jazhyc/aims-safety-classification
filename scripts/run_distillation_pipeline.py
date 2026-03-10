"""
Distillation pipeline: teacher trace generation → student fine-tuning → safety evaluation.

Each teacher model is benchmarked through three sequential steps:
  1. Teacher  – generate_reasoning_traces.py produces parsed_results.json
  2. Distill  – train_generator.py (reasoning_distillation config) fine-tunes the student
  3. Evaluate – safety_experiment.py evaluates the distilled student

Caching prevents duplicate work:
  - Step 1: skipped if data/reasoning_traces/<teacher_slug>/parsed_results.json is non-empty
  - Step 2: skipped if trained_models/causal/<run_name>_adapter/adapter_config.json exists
  - Step 3: skipped if data/safety_experiment/pipeline/<teacher_slug>/.done exists

WandB run names are derived from the teacher model name so results are easy to compare.

Usage:
    python scripts/run_distillation_pipeline.py
    python scripts/run_distillation_pipeline.py wandb.enabled=true
    python scripts/run_distillation_pipeline.py "teacher_models=[model/a,model/b]"
    python scripts/run_distillation_pipeline.py conditions=[without_intent]
    python scripts/run_distillation_pipeline.py cuda_visible_devices=MIG-<uuid>
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf


# Maps distillation condition name → safety experiment fields
CONDITION_MAP = {
    "without_intent": {
        "safety_condition": "finetuned_reasoning_classification",
        "adapter_config_key": "finetuned.reasoning_classification_adapter",
    },
    "with_intent": {
        "safety_condition": "finetuned_reasoning_generation",
        "adapter_config_key": "finetuned.reasoning_generation_adapter",
    },
}


def _teacher_slug(teacher_model: str) -> str:
    """Convert a model name to a filesystem-safe slug (mirrors generate_reasoning_traces.py)."""
    return teacher_model.replace("/", "-")


def _condition_slug(condition: str) -> str:
    return condition.replace("_", "-")


# ── Cache helpers ──────────────────────────────────────────────────────────────

def traces_cached(teacher_model: str, traces_dir: Path) -> tuple[bool, Path]:
    """Step 1 cache: non-empty parsed_results.json exists."""
    path = traces_dir / _teacher_slug(teacher_model) / "parsed_results.json"
    if path.exists():
        with open(path) as f:
            data = json.load(f)
        if data:
            return True, path
    return False, path


def _run_name(teacher_model: str, condition: str, suffix: str | None) -> str:
    parts = ["reasoning-distillation", _teacher_slug(teacher_model), _condition_slug(condition)]
    if suffix:
        parts.append(suffix)
    return "-".join(parts)


def _teacher_run_name(teacher_model: str, suffix: str | None) -> str:
    slug = _teacher_slug(teacher_model)
    return f"{slug}-{suffix}" if suffix else slug


def distillation_cached(teacher_model: str, condition: str, models_dir: Path, suffix: str | None = None) -> tuple[bool, Path]:
    """Step 2 cache: LoRA adapter_config.json exists inside the adapter directory."""
    run_name = _run_name(teacher_model, condition, suffix)
    adapter_dir = models_dir / f"{run_name}_adapter"
    if (adapter_dir / "adapter_config.json").exists():
        return True, adapter_dir
    return False, adapter_dir


def safety_cached(teacher_model: str, safety_dir: Path, suffix: str | None = None) -> tuple[bool, Path]:
    """Step 3 cache: .done sentinel file exists."""
    subdir = f"{_teacher_slug(teacher_model)}-{suffix}" if suffix else _teacher_slug(teacher_model)
    sentinel = safety_dir / "pipeline" / subdir / ".done"
    return sentinel.exists(), sentinel


# ── Step runners ──────────────────────────────────────────────────────────────

def _run(cmd: list[str], project_root: Path, label: str, extra_env: dict | None = None) -> bool:
    """Run a subprocess from project_root, return True on success."""
    env = {**os.environ, **(extra_env or {})}
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  $ {' '.join(cmd)}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, cwd=str(project_root), env=env)
    if result.returncode != 0:
        print(f"  FAILED (exit {result.returncode}): {label}")
        return False
    return True


def step1_teacher(
    teacher_model: str,
    conditions: list[str],
    num_samples: int | None,
    traces_dir: Path,
    wandb_cfg: dict,
    extra_env: dict,
    project_root: Path,
    thinking_mode: bool = False,
) -> bool:
    cached, path = traces_cached(teacher_model, traces_dir)
    if cached:
        print(f"[CACHE HIT] Step 1 – traces already exist: {path}")
        return True

    conditions_str = ",".join(conditions)
    cmd = [
        sys.executable, "scripts/generate_reasoning_traces.py",
        f"model.name={teacher_model}",
        f"conditions=[{conditions_str}]",
        f"paths.output_dir={traces_dir}",
        f"thinking_mode={str(thinking_mode).lower()}",
        f"wandb.enabled={str(wandb_cfg.get('enabled', False)).lower()}",
        f"wandb.project={wandb_cfg.get('reasoning_traces_project', 'reasoning-traces-generation')}",
        f"wandb.run_name={_teacher_slug(teacher_model)}",
    ]
    if wandb_cfg.get("entity"):
        cmd.append(f"wandb.entity={wandb_cfg['entity']}")
    if num_samples is not None:
        cmd.append(f"dataset.num_samples={num_samples}")

    return _run(cmd, project_root, f"Step 1 – teacher traces: {teacher_model}", extra_env)


def step2_distill(
    teacher_model: str,
    condition: str,
    traces_path: Path,
    models_dir: Path,
    wandb_cfg: dict,
    extra_env: dict,
    project_root: Path,
    suffix: str | None = None,
) -> tuple[bool, Path]:
    cached, adapter_dir = distillation_cached(teacher_model, condition, models_dir, suffix)
    if cached:
        print(f"[CACHE HIT] Step 2 – adapter already exists: {adapter_dir}")
        return True, adapter_dir

    run_name = _run_name(teacher_model, condition, suffix)
    model_save_dir = models_dir / run_name
    wandb_run_name = f"{_teacher_slug(teacher_model)}-{_condition_slug(condition)}"
    if suffix:
        wandb_run_name += f"-{suffix}"

    cmd = [
        sys.executable, "scripts/train_generator.py",
        "--config-name=reasoning_distillation",
        f"data.reasoning_traces_path={traces_path}",
        f"data.reasoning_traces_condition={condition}",
        f"paths.model_save_dir={model_save_dir}",
        f"paths.output_dir=train_results/causal/{run_name}",
        f"paths.logs_dir=logs/causal/{run_name}",
        f"paths.predictions_dir=data/predictions/causal/{run_name}",
        f"wandb.enabled={str(wandb_cfg.get('enabled', False)).lower()}",
        f"wandb.project={wandb_cfg.get('distillation_project', 'reasoning-distillation')}",
        f"wandb.run_name={wandb_run_name}",
    ]
    if wandb_cfg.get("entity"):
        cmd.append(f"wandb.entity={wandb_cfg['entity']}")

    label = f"Step 2 – distillation: {teacher_model} / {condition}"
    ok = _run(cmd, project_root, label, extra_env)
    # Training may have succeeded even if the subprocess exited non-zero (e.g. post-
    # training vLLM eval OOM).  If the adapter was saved, treat the step as success.
    if not ok and (adapter_dir / "adapter_config.json").exists():
        print(f"  NOTE: subprocess exited with error but adapter was saved — continuing to step 3")
        ok = True
    return ok, adapter_dir


def step3_safety(
    teacher_model: str,
    adapter_map: dict[str, Path],   # condition -> adapter_dir
    safety_dir: Path,
    wandb_cfg: dict,
    extra_env: dict,
    project_root: Path,
    suffix: str | None = None,
) -> bool:
    cached, sentinel = safety_cached(teacher_model, safety_dir, suffix)
    if cached:
        print(f"[CACHE HIT] Step 3 – safety results already exist: {sentinel.parent}")
        return True

    subdir = f"{_teacher_slug(teacher_model)}-{suffix}" if suffix else _teacher_slug(teacher_model)
    output_dir = safety_dir / "pipeline" / subdir
    safety_conditions = [CONDITION_MAP[c]["safety_condition"] for c in adapter_map]
    conditions_str = ",".join(safety_conditions)

    cmd = [
        sys.executable, "scripts/safety_experiment.py",
        f"experiment.conditions=[{conditions_str}]",
        f"paths.output_dir={output_dir}",
        "wandb.enabled=true",  # always log safety results regardless of pipeline wandb.enabled
        f"wandb.project={wandb_cfg.get('safety_project', 'distillation-teacher-sweep')}",
        f"wandb.run_name={_teacher_run_name(teacher_model, suffix)}",
    ]
    if wandb_cfg.get("entity"):
        cmd.append(f"wandb.entity={wandb_cfg['entity']}")
    for condition, adapter_dir in adapter_map.items():
        adapter_key = CONDITION_MAP[condition]["adapter_config_key"]
        cmd.append(f"{adapter_key}={adapter_dir}")

    ok = _run(cmd, project_root, f"Step 3 – safety evaluation: {teacher_model}", extra_env)
    if ok:
        sentinel.parent.mkdir(parents=True, exist_ok=True)
        sentinel.touch()
    return ok


# ── Main ──────────────────────────────────────────────────────────────────────

@hydra.main(version_base=None, config_path="../configs/model_generation", config_name="distillation_pipeline")
def main(cfg: DictConfig) -> None:
    config = OmegaConf.to_container(cfg, resolve=True)
    project_root = Path(__file__).parent.parent.resolve()

    teacher_models: list[str] = config.get("teacher_models", [])
    conditions: list[str] = config.get("conditions", ["without_intent", "with_intent"])
    num_samples = config.get("num_samples")
    cuda_device = config.get("cuda_visible_devices")
    run_suffix: str | None = config.get("run_suffix") or None
    thinking_mode: bool = bool(config.get("thinking_mode", False))
    # Maps teacher model name → existing parsed_results.json path (skips step 1 for that model)
    trace_overrides: dict[str, str] = config.get("teacher_trace_overrides", {}) or {}
    wandb_cfg: dict = config.get("wandb", {})
    cache_cfg: dict = config.get("cache", {})

    traces_dir = project_root / cache_cfg.get("reasoning_traces_dir", "data/reasoning_traces")
    models_dir = project_root / cache_cfg.get("distilled_models_dir", "trained_models/causal")
    safety_dir = project_root / cache_cfg.get("safety_dir", "data/safety_experiment")

    # Validate conditions
    unknown = [c for c in conditions if c not in CONDITION_MAP]
    if unknown:
        raise ValueError(f"Unknown conditions: {unknown}. Valid: {list(CONDITION_MAP)}")

    # MIG / CUDA device is only needed for the large teacher model (step 1).
    # Steps 2 and 3 run the 8B student which fits on the default GPU.
    teacher_env: dict[str, str] = {}
    if cuda_device is not None:
        teacher_env["CUDA_VISIBLE_DEVICES"] = str(cuda_device)
        teacher_env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    print(f"\n{'#'*60}")
    print(f"  Distillation Pipeline")
    print(f"  Teachers  : {len(teacher_models)}")
    print(f"  Conditions: {conditions}")
    if run_suffix:
        print(f"  Run suffix: {run_suffix}")
    if thinking_mode:
        print(f"  Thinking mode: enabled")
    if cuda_device:
        print(f"  CUDA (step 1 only): {cuda_device}")
    print(f"{'#'*60}")

    summary: dict[str, dict] = {}

    for teacher_model in teacher_models:
        slug = _teacher_slug(teacher_model)
        print(f"\n\n{'='*60}")
        print(f"  Teacher: {teacher_model}")
        print(f"{'='*60}")
        summary[slug] = {}

        # ── Step 1: teacher traces (large model — use MIG device if set) ──────
        if teacher_model in trace_overrides:
            traces_path = project_root / trace_overrides[teacher_model]
            print(f"[OVERRIDE] Step 1 – using pre-existing traces: {traces_path}")
            summary[slug]["step1_traces"] = "override"
        else:
            ok1 = step1_teacher(
                teacher_model, conditions, num_samples,
                traces_dir, wandb_cfg, teacher_env, project_root,
                thinking_mode=thinking_mode,
            )
            summary[slug]["step1_traces"] = "ok" if ok1 else "FAILED"
            if not ok1:
                print(f"  Skipping steps 2 & 3 for {teacher_model} — step 1 failed.")
                continue
            _, traces_path = traces_cached(teacher_model, traces_dir)

        # ── Step 2: distillation per condition (8B student — default GPU) ──
        adapter_map: dict[str, Path] = {}
        for condition in conditions:
            ok2, adapter_dir = step2_distill(
                teacher_model, condition, traces_path,
                models_dir, wandb_cfg, {}, project_root, run_suffix,
            )
            summary[slug][f"step2_{condition}"] = "ok" if ok2 else "FAILED"
            if ok2:
                adapter_map[condition] = adapter_dir

        # ── Step 3: safety experiment (8B student — default GPU) ───────────
        if adapter_map:
            ok3 = step3_safety(
                teacher_model, adapter_map,
                safety_dir, wandb_cfg, {}, project_root, run_suffix,
            )
            summary[slug]["step3_safety"] = "ok" if ok3 else "FAILED"
        else:
            print(f"  Skipping step 3 — no adapters available for {teacher_model}.")
            summary[slug]["step3_safety"] = "skipped"

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n\n{'#'*60}")
    print("  Pipeline Summary")
    print(f"{'#'*60}")
    for slug, steps in summary.items():
        print(f"\n  {slug}")
        for step, status in steps.items():
            marker = "✓" if status in ("ok", "override") else ("–" if status == "skipped" else "✗")
            print(f"    {marker} {step}: {status}")


if __name__ == "__main__":
    main()
