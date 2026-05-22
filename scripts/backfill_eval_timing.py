"""One-shot W&B → local backfill for eval-suite timing.

The local prediction JSONLs for our headline safety models already exist on disk,
but the suite timing (elapsed_s, total_tokens, tokens_per_second) was previously
only logged to W&B. This script fetches those numbers and writes them locally
in the same JSON format that `scripts/eval_safety_classifier.py` now produces
for fresh runs — so the timing notebook can read a single, uniform source.

For each (model, condition), the script:
  1. Lists runs in the configured W&B project, filtered by run-name pattern when given.
  2. Picks the most-recent run that has at least one populated `{ds}/{condition}/elapsed_s` key.
  3. Writes `<model_slug>_<condition>.metrics.json` next to every existing `.jsonl` under
     the model's output directory, and writes a single `<model_slug>_<condition>_suite_summary.json`
     at the directory root capturing the suite-level totals.

Usage:
    .venv/bin/python3.12 scripts/backfill_eval_timing.py            # backfill all 4
    .venv/bin/python3.12 scripts/backfill_eval_timing.py --dry-run  # show what would happen
    .venv/bin/python3.12 scripts/backfill_eval_timing.py --model wildguard
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional

from dotenv import load_dotenv
load_dotenv()

import wandb


# OOD test sets scoped for this analysis. Order is preserved in the summary file.
OOD_DATASETS = ["wildguardmix", "aegis", "toxic-chat", "openai-moderation", "xstest"]


@dataclass
class ModelSpec:
    """A single model+condition to backfill."""
    key: str                      # CLI selector
    model_slug: str               # used in the result-file names
    condition: str                # used in the metric keys: {dataset}/{condition}/elapsed_s
    project: str                  # W&B project
    output_dir: Path              # where the existing .jsonl predictions live
    run_name_contains: List[str] = field(default_factory=list)
    # Optional filters applied to run.config: each (dotted_path, substring) must match.
    # Example: ("model.name", "meta-llama/Llama-3.1-8B-Instruct")
    config_contains: List[tuple[str, str]] = field(default_factory=list)
    description: str = ""


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "data" / "safety_experiment"


MODELS: list[ModelSpec] = [
    ModelSpec(
        key="wildguard",
        model_slug="allenai_wildguard",
        condition="wildguard_classification",
        project="Baselines",
        output_dir=DATA,
        description="allenai/wildguard prior-work baseline",
    ),
    ModelSpec(
        key="nemotron",
        model_slug="nvidia_Nemotron-Content-Safety-Reasoning-4B",
        condition="nemotron_classification",
        project="Baselines",
        output_dir=DATA,
        description="NVIDIA Nemotron Content-Safety Reasoning 4B prior-work baseline",
    ),
    ModelSpec(
        key="safeguard",
        model_slug="openai_gpt-oss-safeguard-120b",
        condition="safeguard_classification",
        project="Baselines",
        output_dir=DATA,
        description="OpenAI gpt-oss-safeguard-120b prior-work baseline",
    ),
    ModelSpec(
        key="guardreasoner",
        model_slug="yueliu1999_GuardReasoner-8B",
        condition="guardreasoner_classification",
        project="Baselines",
        output_dir=DATA,
        description="GuardReasoner 8B prior-work baseline",
    ),
    ModelSpec(
        key="llama_sft",
        model_slug="meta-llama_Llama-3.1-8B-Instruct",
        condition="finetuned_generation",
        project="Baselines",
        output_dir=DATA / "sft",
        # Disambiguates from Gemma SFT (also uses `finetuned_generation` in this project).
        config_contains=[("model.name", "meta-llama/Llama-3.1-8B-Instruct")],
        description="Llama-3.1-8B-Instruct SFT (generation adapter, OOD-val-selected) — test eval",
    ),
    ModelSpec(
        key="gemma_distill",
        model_slug="google_gemma-3-12b-it",
        condition="finetuned_reasoning_human_intent",
        project="Distillation Results",
        output_dir=DATA / "distillation" / "openai-gpt-oss-120b" / "gemma-3-12b" / "human_intent",
        # Released best-distillation classifier: human_intent, gpt-oss-120b teacher,
        # gemma-3-12b student, lr=2e-5, v7.
        config_contains=[
            ("model.name", "google/gemma-3-12b-it"),
            ("finetuned.reasoning_human_intent_adapter", "openai-gpt-oss-120b--gemma-3-12b--human-intent--lr2e-05--v7"),
        ],
        description=(
            "Best-distillation classifier — gpt-oss-120b → gemma-3-12b, human_intent, lr=2e-5, v7"
        ),
    ),
]


def _matches_name_hint(name: str, hints: Iterable[str]) -> bool:
    if not hints:
        return True
    n = name.lower()
    return all(h.lower() in n for h in hints)


def _dotted_get(d, dotted_path: str):
    """Walk a dict via 'a.b.c'-style key. Returns None if any step is missing."""
    cur = d
    for part in dotted_path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def _matches_config(cfg: dict, filters: Iterable[tuple]) -> bool:
    if not filters:
        return True
    for path, needle in filters:
        val = _dotted_get(cfg or {}, path)
        if val is None or needle not in str(val):
            return False
    return True


def _extract_run_timing(run, condition: str) -> dict:
    """Walk run.summary for {dataset}/{condition}/{stat} entries and group by dataset."""
    out: dict[str, dict] = {}
    suffix_map = {
        "elapsed_s": "elapsed_s",
        "total_tokens": "total_tokens",
        "tokens_per_second": "tokens_per_second",
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "correct": "correct",
        "total": "total",
        "tp": "tp", "fp": "fp", "tn": "tn", "fn": "fn",
    }
    # run.summary is a wandb SummarySubDict; iterate keys as strings
    for k in list(run.summary.keys()):
        if not isinstance(k, str):
            continue
        if not k.endswith(tuple(f"/{condition}/{s}" for s in suffix_map)):
            continue
        head, cond, stat = k.rsplit("/", 2)
        # head is the dataset name; verify
        if cond != condition or stat not in suffix_map:
            continue
        bucket = out.setdefault(head, {})
        try:
            bucket[stat] = run.summary[k]
        except Exception:
            continue
    return out


def find_run(api: wandb.Api, spec: ModelSpec, entity: Optional[str] = None) -> Optional["wandb.apis.public.Run"]:
    """Find the most recent finished run in `spec.project` that has the expected timing keys."""
    path = f"{entity}/{spec.project}" if entity else spec.project
    # `state` filter accepts only finished runs; relax if no candidates.
    filters = {"$or": [{"state": "finished"}, {"state": "crashed"}]}
    print(
        f"  [search] project='{spec.project}'  condition='{spec.condition}'  "
        f"name_hint={spec.run_name_contains or '<none>'}  "
        f"config_filter={spec.config_contains or '<none>'}"
    )
    # Collect all matching runs and pick the most recent deterministically. The
    # `order="-created_at"` hint isn't always strict across pages, so don't rely on
    # the first iterate being the newest.
    matched = []
    for run in api.runs(path=path, filters=filters, order="-created_at"):
        if not _matches_name_hint(run.name or "", spec.run_name_contains):
            continue
        # api.runs() returns lazily-loaded runs whose .config is empty until .load() is called.
        if spec.config_contains:
            try:
                run.load(force=True)
            except Exception:
                pass
            if not _matches_config(run.config or {}, spec.config_contains):
                continue
        timing = _extract_run_timing(run, spec.condition)
        if not timing:
            continue
        matched.append((run.created_at, len(timing), run))

    if not matched:
        return None
    # Pick the run with the most datasets covered, breaking ties by most recent.
    matched.sort(key=lambda t: (t[1], t[0]), reverse=True)
    return matched[0][2]


def write_backfill(spec: ModelSpec, run, dry_run: bool) -> None:
    """Read timing from `run.summary` and write local JSON files matching the live-eval format."""
    timing = _extract_run_timing(run, spec.condition)
    if not timing:
        print(f"  [skip] no timing metrics for condition={spec.condition} in run {run.id}")
        return

    print(f"  [picked] run='{run.name}' id={run.id} created={run.created_at}")
    print(f"  [found]  datasets: {sorted(timing.keys())}")

    suite_runs: dict[str, dict] = {}
    suite_elapsed = 0.0
    suite_tokens = 0
    suite_total = 0
    suite_datasets: list[str] = []

    for dataset_name, stats in sorted(timing.items()):
        # Locate the prediction .jsonl this metrics file should sit next to.
        jsonl = spec.output_dir / dataset_name / f"{spec.model_slug}_{spec.condition}.jsonl"
        if not jsonl.exists():
            print(f"    [warn] no local prediction file for {dataset_name}: {jsonl}")
            continue

        payload = {
            "model_slug": spec.model_slug,
            "condition": spec.condition,
            "dataset_name": dataset_name,
            "source": f"wandb:{run.project}/{run.id}",
            **{k: stats.get(k) for k in [
                "accuracy", "precision", "recall", "f1",
                "correct", "total", "tp", "fp", "tn", "fn",
                "total_tokens", "elapsed_s", "tokens_per_second",
            ]},
        }
        out_path = jsonl.with_name(f"{spec.model_slug}_{spec.condition}.metrics.json")
        if dry_run:
            print(f"    [dry] would write {out_path}")
        else:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            print(f"    ✓ wrote {out_path}")

        suite_key = f"{dataset_name}/{spec.condition}"
        suite_runs[suite_key] = payload
        try:
            suite_elapsed += float(stats.get("elapsed_s") or 0.0)
            suite_tokens += int(stats.get("total_tokens") or 0)
            suite_total += int(stats.get("total") or 0)
        except (TypeError, ValueError):
            pass
        suite_datasets.append(dataset_name)

    suite_tps = suite_tokens / suite_elapsed if suite_elapsed > 0 else 0.0
    summary = {
        "model_slug": spec.model_slug,
        "condition": spec.condition,
        "source": f"wandb:{run.project}/{run.id}",
        "wandb_run_name": run.name,
        "datasets": sorted(suite_datasets),
        "conditions": [spec.condition],
        "runs": suite_runs,
        "suite": {
            "elapsed_s": suite_elapsed,
            "total_tokens": suite_tokens,
            "total_examples": suite_total,
            "tokens_per_second": suite_tps,
        },
    }
    summary_path = spec.output_dir / f"{spec.model_slug}_{spec.condition}_suite_summary.json"
    if dry_run:
        print(f"  [dry] would write {summary_path}")
    else:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"  ✓ wrote suite summary: {summary_path}")
        print(f"    suite: n={suite_total:,}  tokens={suite_tokens:,}  time={suite_elapsed:.1f}s  tps={suite_tps:.1f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=[m.key for m in MODELS] + ["all"], default="all",
                        help="Which model to backfill (default: all 4)")
    parser.add_argument("--dry-run", action="store_true", help="Don't write files; print intended writes.")
    parser.add_argument("--entity", default=None, help="W&B entity (defaults to user's default).")
    args = parser.parse_args()

    selected = MODELS if args.model == "all" else [m for m in MODELS if m.key == args.model]
    api = wandb.Api()
    # Prefer WANDB_ENTITY (set in .env for this project) over the personal default.
    entity = args.entity or os.environ.get("WANDB_ENTITY") or api.default_entity
    print(f"W&B entity: {entity}")

    for spec in selected:
        print(f"\n=== {spec.key}: {spec.description} ===")
        run = find_run(api, spec, entity=entity)
        if run is None:
            print(f"  [error] could not find a finished W&B run with timing for condition={spec.condition}")
            continue
        write_backfill(spec, run, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
