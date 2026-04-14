"""
Distillation pipeline: teacher trace generation → student fine-tuning → safety evaluation.

Each teacher model is benchmarked through three sequential steps:
  1. Teacher  – generate_reasoning_traces.py produces per-split parsed_results.json files
  2. Distill  – train_generator.py (reasoning_distillation config) fine-tunes the student
  3. Evaluate – eval_safety_classifier.py evaluates the distilled student

Caching prevents duplicate work:
  - Step 1: skipped if data/reasoning_traces/<teacher_slug>/train/parsed_results.json is non-empty
  - Step 2: skipped if models/distillation/<run_name>_adapter/adapter_config.json exists
  - Step 3: skipped if data/safety_experiment/pipeline/<teacher_slug>/.done exists

WandB run names are derived from the teacher model name so results are easy to compare.

Usage:
    python scripts/run_distillation_pipeline.py
    python scripts/run_distillation_pipeline.py wandb.enabled=true
    python scripts/run_distillation_pipeline.py "teacher_models=[model/a,model/b]"
    python scripts/run_distillation_pipeline.py conditions=[no_intent]
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
    "no_intent": {
        "safety_condition": "finetuned_reasoning_classification",
        "adapter_config_key": "finetuned.reasoning_classification_adapter",
    },
    "synthetic_intent": {
        "safety_condition": "finetuned_reasoning_synthetic_intent",
        "adapter_config_key": "finetuned.reasoning_synthetic_adapter",
    },
    "human_intent": {
        "safety_condition": "finetuned_reasoning_human_intent",
        "adapter_config_key": "finetuned.reasoning_human_intent_adapter",
    },
}


# Per-model slug overrides. Models listed here use the short form instead of the
# default replace("/", "-") to keep paths and W&B run names manageable.
# Must stay in sync with the identical map in generate_reasoning_traces.py.
_TEACHER_SLUG_OVERRIDES = {
    "cyankiwi/gemma-4-31B-it-AWQ-4bit": "cyankiwi-gemma-4-31b",
}


def _teacher_slug(teacher_model: str) -> str:
    """Convert a model name to a filesystem-safe slug (mirrors generate_reasoning_traces.py)."""
    if teacher_model in _TEACHER_SLUG_OVERRIDES:
        return _TEACHER_SLUG_OVERRIDES[teacher_model]
    return teacher_model.replace("/", "-")


def _condition_slug(condition: str) -> str:
    return condition.replace("_", "-")


# ── Cache helpers ──────────────────────────────────────────────────────────────

def traces_cached(teacher_model: str, traces_dir: Path) -> tuple[bool, Path]:
    """Step 1 cache: non-empty train/parsed_results.json exists."""
    path = traces_dir / _teacher_slug(teacher_model) / "train" / "parsed_results.json"
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


def _artifact_name_traces(teacher_model: str) -> str:
    return f"reasoning-traces-{_teacher_slug(teacher_model)}"


def _try_download_traces(teacher_model: str, traces_dir: Path, artifacts_cfg: dict) -> bool:
    """Try to pull traces from W&B. Returns True if traces are now available locally."""
    from intention_jailbreak.model_generation.artifacts import download_traces
    slug = _teacher_slug(teacher_model)
    artifact_name = _artifact_name_traces(teacher_model)
    registry_project = artifacts_cfg.get("registry_project", "intention-jailbreak-models")
    entity = artifacts_cfg.get("entity") or None
    teacher_traces_dir = traces_dir / slug
    try:
        download_traces(artifact_name, registry_project, teacher_traces_dir, entity)
        print(f"[W&B HIT] Step 1 – traces downloaded from W&B: {artifact_name}")
        return True
    except Exception as e:
        print(f"[W&B MISS] Could not download traces '{artifact_name}': {e} — will generate")
        return False


def _upload_traces(teacher_model: str, traces_dir: Path, artifacts_cfg: dict, wandb_cfg: dict) -> None:
    """Upload generated reasoning traces to W&B using a temporary run."""
    import wandb as _wandb
    from intention_jailbreak.model_generation.artifacts import upload_traces
    slug = _teacher_slug(teacher_model)
    artifact_name = _artifact_name_traces(teacher_model)
    registry_project = artifacts_cfg.get("registry_project", "intention-jailbreak-models")
    entity = artifacts_cfg.get("entity") or wandb_cfg.get("entity") or None
    teacher_traces_dir = traces_dir / slug
    run = _wandb.init(
        project=wandb_cfg.get("reasoning_traces_project", "reasoning-traces-generation"),
        entity=entity,
        name=f"upload-traces-{slug}",
        job_type="artifact-upload",
    )
    try:
        upload_traces(teacher_traces_dir, artifact_name, registry_project, entity, run)
    finally:
        run.finish()
    print(f"[artifacts] Traces uploaded: {artifact_name}")


def step1_teacher(
    teacher_model: str,
    conditions: list[str],
    num_samples: int | None,
    traces_dir: Path,
    wandb_cfg: dict,
    extra_env: dict,
    project_root: Path,
    thinking_mode: bool = False,
    artifacts_cfg: dict | None = None,
    backend: str = "vllm",
) -> bool:
    cached, path = traces_cached(teacher_model, traces_dir)
    if cached:
        print(f"[CACHE HIT] Step 1 – traces already exist: {path}")
        return True

    # Not cached locally — try W&B before running expensive generation
    if artifacts_cfg and artifacts_cfg.get("enabled"):
        if _try_download_traces(teacher_model, traces_dir, artifacts_cfg):
            return True

    conditions_str = ",".join(conditions)
    cmd = [
        sys.executable, "scripts/distillation/generate_reasoning_traces.py",
        f"model.name={teacher_model}",
        f"model.backend={backend}",
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

    ok = _run(cmd, project_root, f"Step 1 – teacher traces: {teacher_model}", extra_env)

    # Upload traces to W&B after successful generation
    if ok and artifacts_cfg and artifacts_cfg.get("enabled"):
        try:
            _upload_traces(teacher_model, traces_dir, artifacts_cfg, wandb_cfg)
        except Exception as e:
            print(f"[artifacts] WARNING: Could not upload traces: {e}")

    return ok


def step2_distill(
    teacher_model: str,
    condition: str,
    traces_dir: Path,
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

    slug = _teacher_slug(teacher_model)
    train_traces_path = traces_dir / slug / "train" / "parsed_results.json"
    val_traces_path   = traces_dir / slug / "validation" / "parsed_results.json"

    run_name = _run_name(teacher_model, condition, suffix)
    model_save_dir = models_dir / run_name
    wandb_run_name = f"{_teacher_slug(teacher_model)}-{_condition_slug(condition)}"
    if suffix:
        wandb_run_name += f"-{suffix}"

    cmd = [
        sys.executable, "scripts/baselines/train_generator.py",
        "--config-name=reasoning_distillation",
        f"data.reasoning_traces_path={train_traces_path}",
        f"data.reasoning_traces_val_path={val_traces_path}",
        f"data.reasoning_traces_condition={condition}",
        f"paths.model_save_dir={model_save_dir}",
        f"paths.output_dir=data/train_results/distillation/{run_name}",
        f"paths.logs_dir=logs/distillation/{run_name}",
        f"paths.predictions_dir=data/predictions/{run_name}",
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
        sys.executable, "scripts/eval_safety_classifier.py",
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

@hydra.main(version_base=None, config_path="../../configs/experiments", config_name="distillation_pipeline")
def main(cfg: DictConfig) -> None:
    config = OmegaConf.to_container(cfg, resolve=True)
    project_root = Path(__file__).parent.parent.parent.resolve()

    teacher_models: list[str] = config.get("teacher_models", [])
    conditions: list[str] = config.get("conditions", ["no_intent", "human_intent"])
    num_samples = config.get("num_samples")
    cuda_device = config.get("cuda_visible_devices")
    run_suffix: str | None = config.get("run_suffix") or None
    thinking_mode: bool = bool(config.get("thinking_mode", False))
    stages_cfg: dict = config.get("stages", {})
    run_step1: bool = bool(stages_cfg.get("step1_teacher", True))
    run_step2: bool = bool(stages_cfg.get("step2_distill", True))
    run_step3: bool = bool(stages_cfg.get("step3_safety", True))
    # Maps teacher model name → existing parsed_results.json path (skips step 1 for that model)
    trace_overrides: dict[str, str] = config.get("teacher_trace_overrides", {}) or {}
    # Maps teacher model name → backend ("vllm" | "openrouter"); default is "vllm"
    teacher_backends: dict[str, str] = config.get("teacher_backends", {}) or {}
    wandb_cfg: dict = config.get("wandb", {})
    cache_cfg: dict = config.get("cache", {})
    artifacts_cfg: dict = config.get("artifacts", {})

    traces_dir = project_root / cache_cfg.get("reasoning_traces_dir", "data/reasoning_traces")
    models_dir = project_root / cache_cfg.get("distilled_models_dir", "models/distillation")
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
    active = [s for s, on in [("step1", run_step1), ("step2", run_step2), ("step3", run_step3)] if on]
    print(f"  Stages    : {', '.join(active) if active else 'none'}")
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
            override_train_path = project_root / trace_overrides[teacher_model]
            print(f"[OVERRIDE] Step 1 – using pre-existing traces: {override_train_path}")
            summary[slug]["step1_traces"] = "override"
        elif not run_step1:
            print(f"  [SKIP] Step 1 disabled in config.")
            summary[slug]["step1_traces"] = "skipped"
            cached, train_path = traces_cached(teacher_model, traces_dir)
            if not cached:
                print(f"  No cached traces found — skipping steps 2 & 3.")
                continue
        else:
            backend = teacher_backends.get(teacher_model, "vllm")
            # API-backed models don't need a GPU — use an empty env override
            step1_env = {} if backend == "openrouter" else teacher_env
            ok1 = step1_teacher(
                teacher_model, conditions, num_samples,
                traces_dir, wandb_cfg, step1_env, project_root,
                thinking_mode=thinking_mode,
                artifacts_cfg=artifacts_cfg,
                backend=backend,
            )
            summary[slug]["step1_traces"] = "ok" if ok1 else "FAILED"
            if not ok1:
                print(f"  Skipping steps 2 & 3 for {teacher_model} — step 1 failed.")
                continue

        # ── Step 2: distillation per condition (8B student — default GPU) ──
        # step2_distill constructs train/val paths from traces_dir / slug / split /
        # parsed_results.json.  Override paths are not yet supported for the val split.
        adapter_map: dict[str, Path] = {}
        if not run_step2:
            print(f"  [SKIP] Step 2 disabled in config.")
            for condition in conditions:
                summary[slug][f"step2_{condition}"] = "skipped"
        else:
            for condition in conditions:
                ok2, adapter_dir = step2_distill(
                    teacher_model, condition, traces_dir,
                    models_dir, wandb_cfg, {}, project_root, run_suffix,
                )
                summary[slug][f"step2_{condition}"] = "ok" if ok2 else "FAILED"
                if ok2:
                    adapter_map[condition] = adapter_dir

        # ── Step 3: safety experiment (8B student — default GPU) ───────────
        if not run_step3:
            print(f"  [SKIP] Step 3 disabled in config.")
            summary[slug]["step3_safety"] = "skipped"
        elif adapter_map:
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
