"""
W&B artifact helpers for LoRA adapter upload/download and eval result upload/download.

All adapters are stored in a single registry project (e.g. "intention-jailbreak-models")
independent of the experiment-tracking project used for a given run. This means adapters
can be retrieved on any device without knowing which experiment project trained them.

Artifact naming
---------------
The artifact name is the adapter directory name (the final path component), e.g.:
  - "lr_5e-05_e_5_adapter"                                    (SFT hyperparam sweep)
  - "classification-r16-a32_adapter"                          (LoRA classification sweep)
  - "reasoning-distillation-gemma-3-27b-without-intent_adapter"  (distillation)

These names are already unique across pipelines because causal.py always appends
"_adapter" and the run name encodes the relevant hyperparameters / teacher model.

Checkpoints
-----------
Only root-level files in the adapter directory are uploaded (adapter weights, tokenizer
files, adapter_config.json). Checkpoint subdirectories (checkpoint-N/) are skipped.

Collision policy
----------------
artifact_exists() should be called before training begins. If the artifact already
exists in the registry, the run should not start — this prevents two devices from
racing to produce the same adapter and avoids accidental overwrites.

If a training run crashes mid-upload and leaves a partial artifact, you must manually
delete it from W&B before re-running (go to the registry project → Artifacts → delete).

Usage
-----
    from intention_jailbreak.model_generation.artifacts import (
        artifact_exists, upload_adapter, download_adapter,
    )

    # Before training:
    if artifact_exists(adapter_name, registry_project="intention-jailbreak-models"):
        raise RuntimeError(f"Artifact '{adapter_name}' already exists in the registry.")

    # After training:
    upload_adapter("models/sft/lr_5e-05_e_5_adapter", project="intention-jailbreak-models")

    # Before evaluation on a different device:
    adapter_path = download_adapter(
        "lr_5e-05_e_5_adapter",
        project="intention-jailbreak-models",
        target_dir="models/sft",
    )
"""

from __future__ import annotations

from pathlib import Path

import wandb

ARTIFACT_TYPE = "lora-adapter"
RESULTS_ARTIFACT_TYPE = "eval-results"


def artifact_exists(
    name: str,
    project: str,
    entity: str | None = None,
) -> bool:
    """
    Return True if an artifact with this name already exists in the registry project.

    Args:
        name:    Artifact name, e.g. "lr_5e-05_e_5_adapter".
        project: W&B registry project name.
        entity:  W&B entity (username or team). Defaults to the account used by wandb.Api.
    """
    api = wandb.Api()
    qualifier = f"{entity}/{project}" if entity else project
    try:
        api.artifact(f"{qualifier}/{name}:latest", type=ARTIFACT_TYPE)
        return True
    except wandb.errors.CommError:
        return False


def upload_adapter(
    adapter_path: str | Path,
    project: str,
    entity: str | None = None,
    run: wandb.Run | None = None,
) -> wandb.Artifact:
    """
    Upload a LoRA adapter directory to the W&B registry project.

    Only root-level files are included; checkpoint-N/ subdirectories are skipped.
    Call artifact_exists() before training to ensure no artifact with this name
    exists — upload does not check for duplicates itself.

    Args:
        adapter_path: Path to the *_adapter/ directory.
        project:      W&B registry project name.
        entity:       W&B entity. Defaults to the account used by the active run.
        run:          Active wandb.Run. Defaults to wandb.run.

    Returns:
        The logged wandb.Artifact.

    Raises:
        FileNotFoundError: If adapter_path does not exist.
        RuntimeError:      If no wandb run is active.
    """
    adapter_path = Path(adapter_path)
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter directory not found: {adapter_path}")

    active_run = run or wandb.run
    if active_run is None:
        raise RuntimeError(
            "No active wandb run. Call wandb.init() before upload_adapter()."
        )

    name = adapter_path.name
    artifact = wandb.Artifact(name=name, type=ARTIFACT_TYPE)

    root_files = [p for p in sorted(adapter_path.iterdir()) if p.is_file()]
    if not root_files:
        raise ValueError(f"No files found in adapter directory: {adapter_path}")

    for f in root_files:
        artifact.add_file(str(f), name=f.name)

    qualifier = f"{entity}/{project}" if entity else project
    active_run.log_artifact(artifact, aliases=["latest"], project=qualifier)
    print(f"[artifacts] Uploaded '{name}' to {qualifier} ({len(root_files)} files)")
    return artifact


def download_adapter(
    name: str,
    project: str,
    target_dir: str | Path,
    entity: str | None = None,
    version: str = "latest",
) -> Path:
    """
    Download a LoRA adapter artifact from the W&B registry project.

    Skips the download if target_dir/name already exists and contains files,
    so repeated calls on the same machine are cheap.

    Args:
        name:       Artifact name, e.g. "lr_5e-05_e_5_adapter".
        project:    W&B registry project name.
        target_dir: Local directory to download into. The adapter will be placed
                    at target_dir/name (e.g. "models/sft/lr_5e-05_e_5_adapter").
        entity:     W&B entity. Defaults to the account used by wandb.Api.
        version:    Artifact version alias (default "latest").

    Returns:
        Path to the downloaded adapter directory (target_dir/name).

    Raises:
        wandb.errors.CommError: If the artifact does not exist in the registry.
    """
    dest = Path(target_dir) / name

    if dest.exists() and any(dest.iterdir()):
        print(f"[artifacts] Already present locally, skipping download: {dest}")
        return dest

    api = wandb.Api()
    qualifier = f"{entity}/{project}" if entity else project
    full_ref = f"{qualifier}/{name}:{version}"

    artifact = api.artifact(full_ref, type=ARTIFACT_TYPE)
    artifact.download(root=str(dest))
    print(f"[artifacts] Downloaded '{full_ref}' → {dest}")
    return dest


def upload_results(
    files: list[Path],
    name: str,
    project: str,
    entity: str | None = None,
    run: "wandb.Run | None" = None,
) -> "wandb.Artifact":
    """
    Upload a list of result files (JSONL) as a W&B artifact.

    Unlike adapters, results do not use a collision guard — uploading always
    creates a new version and ``latest`` points to the most recent run. This
    lets notebooks always pull the freshest results with a fixed artifact name.

    Args:
        files:   List of Path objects to include. Relative structure inside the
                 artifact mirrors the paths relative to their common ancestor.
        name:    Artifact name, e.g. "safety-experiment-results".
        project: W&B registry project name.
        entity:  W&B entity. Defaults to the account used by the active run.
        run:     Active wandb.Run. Defaults to wandb.run.

    Returns:
        The logged wandb.Artifact.
    """
    active_run = run or wandb.run
    if active_run is None:
        raise RuntimeError(
            "No active wandb run. Call wandb.init() before upload_results()."
        )
    if not files:
        raise ValueError("No files provided to upload_results().")

    # Use the common ancestor as the root so relative paths are preserved.
    common = Path(files[0].anchor)
    try:
        from pathlib import PurePath
        parts = [list(f.parts) for f in files]
        min_len = min(len(p) for p in parts)
        common_parts = []
        for i in range(min_len):
            if len(set(p[i] for p in parts)) == 1:
                common_parts.append(parts[0][i])
            else:
                break
        common = Path(*common_parts) if common_parts else Path(".")
    except Exception:
        common = Path(".")

    artifact = wandb.Artifact(name=name, type=RESULTS_ARTIFACT_TYPE)
    for f in sorted(files):
        try:
            rel = f.relative_to(common)
        except ValueError:
            rel = Path(f.name)
        artifact.add_file(str(f), name=str(rel))

    qualifier = f"{entity}/{project}" if entity else project
    active_run.log_artifact(artifact, aliases=["latest"], project=qualifier)
    print(f"[artifacts] Uploaded results '{name}' to {qualifier} ({len(files)} files)")
    return artifact


def download_results(
    name: str,
    project: str,
    target_dir: str | Path,
    entity: str | None = None,
    version: str = "latest",
) -> Path:
    """
    Download a results artifact from the W&B registry project.

    Skips the download if target_dir/name already exists and contains .jsonl files.

    Args:
        name:       Artifact name, e.g. "safety-experiment-results".
        project:    W&B registry project name.
        target_dir: Local directory to download into (files placed at target_dir/name/).
        entity:     W&B entity. Defaults to the account used by wandb.Api.
        version:    Artifact version alias (default "latest").

    Returns:
        Path to the downloaded results directory (target_dir/name).
    """
    dest = Path(target_dir) / name
    if dest.exists() and any(dest.rglob("*.jsonl")):
        print(f"[artifacts] Results already present locally, skipping download: {dest}")
        return dest

    api = wandb.Api()
    qualifier = f"{entity}/{project}" if entity else project
    full_ref = f"{qualifier}/{name}:{version}"

    artifact = api.artifact(full_ref, type=RESULTS_ARTIFACT_TYPE)
    artifact.download(root=str(dest))
    print(f"[artifacts] Downloaded results '{full_ref}' → {dest}")
    return dest
