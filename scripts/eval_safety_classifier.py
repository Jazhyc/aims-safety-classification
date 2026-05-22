"""
Safety classification experiment comparing prompting baselines and prior-work models.

Condition groups
----------------
Prompting baselines (single main model instance):
  vanilla_classification              prompt -> harm label
  vanilla_generation                  prompt -> intent + harm
  vanilla_with_human_intent           prompt + human intent -> harm
  vanilla_with_model_intent           prompt + model intent -> harm
  zeroshot_cot_classification         CoT -> reasoning + harm (JSON)
  zeroshot_cot_generation             CoT -> reasoning + intent + harm (JSON)
  zeroshot_cot_classification_with_intent  CoT + human intent -> reasoning + harm (JSON)

Prior-work baselines (each loads its own model after the main model is freed):
  llamaguard_classification           LlamaGuard 4 (12B) native classification
  wildguard_classification            WildGuard (7B) native classification

Fine-tuned (separate run, adapters must be present locally or in artifact registry):
  finetuned_generation                SFT adapter: prompt -> intent + harm
  finetuned_classification            SFT adapter: prompt -> harm
  finetuned_reasoning_classification  Distilled adapter (MODE D): reasoning + harm
  finetuned_reasoning_generation      Distilled adapter (MODE E): reasoning + intent + harm

Datasets:
  Jazhyc/wildguard-annotated-intents  annotated test split (has human intents)
  allenai/wildguardmix (wildguardtest) WildGuardTest benchmark
  walledai/XSTest                     XSTest benchmark

Usage:
    python scripts/eval_safety_classifier.py
    python scripts/eval_safety_classifier.py --config-name=eval_safety_classifier
    python scripts/eval_safety_classifier.py experiment.conditions=[vanilla_classification]
"""

import os
import json
import time
import warnings
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

# Suppress pydantic warnings from vllm internals
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

from datasets import load_dataset
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
import hydra
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer

try:
    import wandb
except ImportError:
    wandb = None

from dotenv import load_dotenv
load_dotenv()

from intention_jailbreak.model_generation.artifacts import download_adapter, upload_results
from intention_jailbreak.model_generation.safety_prompts import VALID_CONDITIONS
from intention_jailbreak.model_generation.safety_classifier import (
    map_harm_to_binary,
    generate_model_intents,
    run_llamaguard_classification,
    run_wildguard_classification,
    run_safeguard_classification,
    run_guardreasoner_classification,
    run_shieldgemma_classification,
    run_nemotron_classification,
    run_grpo_classification,
    run_vanilla_generation_openrouter,
    run_condition_on_dataset,
)


PRIOR_WORK_CONDITIONS = {
    "llamaguard_classification",
    "wildguard_classification",
    "safeguard_classification",
    "guardreasoner_classification",
    "shieldgemma_classification",
    "nemotron_classification",
    "grpo_classification",
}


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_test_dataset(data_cfg: dict) -> Tuple[Any, str, bool]:
    """
    Load and prepare a test dataset from config.

    Returns:
        (test_dataset, harm_column, has_intent)
    """
    dataset_name = data_cfg["dataset_name"]
    subset = data_cfg.get("subset")
    split = data_cfg.get("split", "train")

    print(f"Loading dataset: {dataset_name}")
    if subset:
        print(f"  Subset: {subset}")
        dataset = load_dataset(dataset_name, subset, split=split)
    else:
        dataset = load_dataset(dataset_name, split=split)

    rename_map = {}
    if "ID" in dataset.column_names:
        rename_map["ID"] = "id"
    if "Prompt" in dataset.column_names:
        rename_map["Prompt"] = "prompt"
    if "Intent" in dataset.column_names:
        rename_map["Intent"] = "intent"
    if "user_input" in dataset.column_names and "prompt" not in dataset.column_names:
        rename_map["user_input"] = "prompt"
    existing_renames = {k: v for k, v in rename_map.items() if k in dataset.column_names}
    if existing_renames:
        dataset = dataset.rename_columns(existing_renames)

    harm_column = data_cfg.get("harm_column", "Annotator Harm")

    label_map = data_cfg.get("label_map")
    if label_map and harm_column in dataset.column_names:
        str_map = {str(k): v for k, v in label_map.items()}
        dataset = dataset.map(lambda x: {**x, harm_column: str_map.get(str(x[harm_column]), str(x[harm_column]))})

    if data_cfg.get("all_harmful", False) and harm_column not in dataset.column_names:
        print(f"  Creating synthetic harm column '{harm_column}' with all 'harmful' labels")
        dataset = dataset.map(lambda x: {**x, harm_column: "harmful"})

    has_intent = "intent" in dataset.column_names

    if data_cfg.get("use_manual_split", True):
        test_size = data_cfg.get("test_size", 0.1)
        seed = data_cfg.get("seed", 22)
        test_dataset = dataset.train_test_split(test_size=test_size, seed=seed)["test"]
    else:
        test_dataset = dataset

    original_size = len(test_dataset)
    test_dataset = test_dataset.filter(lambda x: x.get(harm_column) is not None)
    filtered_count = original_size - len(test_dataset)
    if filtered_count > 0:
        print(f"  Filtered {filtered_count} samples with null harm labels")

    print(f"Test dataset size: {len(test_dataset)}")
    print(f"Harm column: {harm_column}")
    print(f"Has intent: {has_intent}")
    return test_dataset, harm_column, has_intent


# ---------------------------------------------------------------------------
# Metrics and I/O
# ---------------------------------------------------------------------------

def compute_metrics(results: List[dict]) -> dict:
    """Compute accuracy, precision, recall, and F1 from results."""
    correct = total = tp = fp = tn = fn = 0

    for r in results:
        true_harm = r.get("true_harm_binary")
        pred_harm = r.get("predicted_harm")
        if true_harm is None or pred_harm is None:
            continue
        true_lower = true_harm.lower()
        pred_lower = pred_harm.lower()
        if true_lower == pred_lower:
            correct += 1
        total += 1
        if true_lower == "harmful" and pred_lower == "harmful":
            tp += 1
        elif true_lower == "safe" and pred_lower == "harmful":
            fp += 1
        elif true_lower == "safe" and pred_lower == "safe":
            tn += 1
        elif true_lower == "harmful" and pred_lower == "safe":
            fn += 1

    accuracy = correct / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1,
        "correct": correct, "total": total,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "missing_predictions": len(results) - total,
    }


def save_results(results: List[dict], output_path: Path, condition: str):
    """Save results to JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"✓ Saved {len(results)} results to {output_path}")


# ---------------------------------------------------------------------------
# Infrastructure helpers for main()
# ---------------------------------------------------------------------------

def _free_vllm() -> None:
    """Release GPU memory (call immediately after `del <model>`)."""
    import gc
    import torch
    gc.collect()
    torch.cuda.empty_cache()


def _load_vllm(model_name: str, vllm_cfg: dict, **extra_kwargs) -> LLM:
    """Construct a vLLM LLM instance from common config, with optional overrides."""
    kwargs: Dict[str, Any] = {
        "model": model_name,
        "gpu_memory_utilization": vllm_cfg.get("gpu_memory_utilization", 0.90),
        "max_model_len": vllm_cfg.get("max_model_len", 2048),
        "dtype": vllm_cfg.get("dtype", "bfloat16"),
        "enforce_eager": vllm_cfg.get("enforce_eager", True),
    }
    flash_attn_version = vllm_cfg.get("flash_attn_version", None)
    if flash_attn_version is not None:
        kwargs["attention_config"] = {"flash_attn_version": flash_attn_version}
    kwargs.update(extra_kwargs)
    return LLM(**kwargs)


def _process_results(
    results: List[dict],
    condition: str,
    dataset_name: str,
    data_cfg: dict,
    paths_cfg: dict,
    model_slug: str,
    wandb_run,
    all_metrics: dict,
    result_files_written: List[Path],
    elapsed_s: float = 0.0,
) -> None:
    """Compute metrics, print summary, save to disk, and log to W&B."""
    if not results:
        return

    metrics = compute_metrics(results)
    total_tokens = sum(r.get("num_tokens", 0) for r in results)
    tokens_per_second = total_tokens / elapsed_s if elapsed_s > 0 else 0.0
    metrics["total_tokens"] = total_tokens
    metrics["elapsed_s"] = elapsed_s
    metrics["tokens_per_second"] = tokens_per_second

    metric_key = f"{dataset_name}/{condition}"
    all_metrics[metric_key] = {**metrics, "model_slug": model_slug, "dataset_name": dataset_name, "condition": condition}

    print(f"\n  Results for {condition}:")
    print(f"    Accuracy: {metrics['accuracy']:.4f} ({metrics['correct']}/{metrics['total']})")
    print(f"    Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}")
    print(f"    Missing predictions: {metrics['missing_predictions']}")
    print(f"    Tokens generated: {total_tokens:,}  |  Time: {elapsed_s:.1f}s  |  Throughput: {tokens_per_second:.1f} tok/s")

    pipeline_output_dir = paths_cfg.get("output_dir")
    output_dir = (
        Path(pipeline_output_dir) / dataset_name
        if pipeline_output_dir
        else Path(data_cfg.get("output_dir", "data/safety_experiment"))
    )
    output_path = output_dir / f"{model_slug}_{condition}.jsonl"
    save_results(results, output_path, condition)
    result_files_written.append(output_path)

    metrics_path = output_dir / f"{model_slug}_{condition}.metrics.json"
    metrics_payload = {
        "model_slug": model_slug,
        "condition": condition,
        "dataset_name": dataset_name,
        **metrics,
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)
    result_files_written.append(metrics_path)

    if wandb_run is not None:
        wandb.log({
            f"{metric_key}/accuracy": metrics["accuracy"],
            f"{metric_key}/precision": metrics["precision"],
            f"{metric_key}/recall": metrics["recall"],
            f"{metric_key}/f1": metrics["f1"],
            f"{metric_key}/correct": metrics["correct"],
            f"{metric_key}/total": metrics["total"],
            f"{metric_key}/total_tokens": total_tokens,
            f"{metric_key}/elapsed_s": elapsed_s,
            f"{metric_key}/tokens_per_second": tokens_per_second,
        })


def _run_baseline_on_datasets(
    run_fn,
    condition: str,
    model_slug: str,
    datasets_cfg: list,
    sampling_params: SamplingParams,
    paths_cfg: dict,
    wandb_run,
    all_metrics: dict,
    result_files_written: List[Path],
) -> None:
    """Run a single-model baseline on every dataset and record results."""
    for dataset_idx, data_cfg in enumerate(datasets_cfg):
        dataset_name = data_cfg.get("name", f"dataset_{dataset_idx}")
        print(f"\n{'='*80}")
        print(f"=== {condition} on Dataset: {dataset_name} ===")
        print(f"{'='*80}")

        test_dataset, harm_column, _ = load_test_dataset(data_cfg)
        t0 = time.monotonic()
        results = run_fn(test_dataset, sampling_params, harm_column)
        elapsed_s = time.monotonic() - t0

        _process_results(
            results, condition, dataset_name, data_cfg, paths_cfg,
            model_slug, wandb_run, all_metrics, result_files_written,
            elapsed_s=elapsed_s,
        )


def _validate_finetuned_adapters(conditions: List[str], finetuned_cfg: dict) -> None:
    """Raise ValueError if any requested finetuned condition is missing its adapter."""
    if finetuned_cfg.get("use_merged", False):
        return  # merged model is loaded as model.name; no adapter paths needed
    checks = {
        "finetuned_generation": ("generation_adapter", "finetuned.generation_adapter"),
        "finetuned_classification": ("classification_adapter", "finetuned.classification_adapter"),
        "finetuned_reasoning_classification": ("reasoning_classification_adapter", "finetuned.reasoning_classification_adapter"),
        "finetuned_reasoning_human_intent": ("reasoning_human_intent_adapter", "finetuned.reasoning_human_intent_adapter"),
        "finetuned_reasoning_synthetic_intent": ("reasoning_synthetic_adapter", "finetuned.reasoning_synthetic_adapter"),
    }
    for cond, (key, cfg_path) in checks.items():
        if cond in conditions:
            path = finetuned_cfg.get(key)
            if not path or not os.path.exists(path):
                raise ValueError(f"{cond} requires valid {cfg_path} path. Got: {path}")


def _setup_lora_requests(
    finetuned_cfg: dict,
) -> Tuple[Optional[LoRARequest], Optional[LoRARequest], Optional[LoRARequest], Optional[LoRARequest], Optional[LoRARequest]]:
    """Create LoRARequest objects for each configured fine-tuned adapter."""
    def _make(key: str, name: str, uid: int) -> Optional[LoRARequest]:
        path = finetuned_cfg.get(key)
        if path and os.path.exists(path):
            print(f"Loaded {name} adapter: {path}")
            return LoRARequest(f"{name}_lora", uid, path)
        return None

    return (
        _make("generation_adapter", "generation", 1),
        _make("classification_adapter", "classification", 2),
        _make("reasoning_classification_adapter", "reasoning_classification", 3),
        _make("reasoning_human_intent_adapter", "reasoning_human_intent", 4),
        _make("reasoning_synthetic_adapter", "reasoning_synthetic", 5),
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="../configs/experiments", config_name="eval_safety_classifier")
def main(cfg: DictConfig):
    """Main experiment function."""
    config = OmegaConf.to_container(cfg, resolve=True)

    model_cfg = config.get("model", {})
    experiment_cfg = config.get("experiment", {})
    datasets_cfg = config.get("datasets", [])
    gen_cfg = config.get("generation", {})
    paths_cfg = config.get("paths", {})
    vllm_cfg = config.get("vllm", {})
    wandb_cfg = config.get("wandb", {})
    lora_cfg = config.get("lora", {})
    finetuned_cfg = config.get("finetuned", {})
    llamaguard_cfg = config.get("llamaguard", {})
    wildguard_cfg = config.get("wildguard", {})
    safeguard_cfg = config.get("safeguard", {})
    guardreasoner_cfg = config.get("guardreasoner", {})
    shieldgemma_cfg = config.get("shieldgemma", {})
    nemotron_cfg = config.get("nemotron", {})
    grpo_cfg = config.get("grpo", {})
    artifacts_cfg = config.get("artifacts", {})

    conditions = experiment_cfg.get("conditions", ["vanilla_classification"])
    if isinstance(conditions, str):
        conditions = [conditions]

    for cond in conditions:
        if cond not in VALID_CONDITIONS:
            raise ValueError(f"Invalid condition: {cond}. Valid: {VALID_CONDITIONS}")

    use_merged = finetuned_cfg.get("use_merged", False)

    finetuned_conditions = {
        "finetuned_generation", "finetuned_classification",
        "finetuned_reasoning_classification", "finetuned_reasoning_generation",
        "finetuned_reasoning_human_intent", "finetuned_reasoning_synthetic_intent",
    }
    needs_finetuned = bool(set(conditions) & finetuned_conditions)
    needs_finetuned_lora = needs_finetuned and not use_merged
    needs_llamaguard = "llamaguard_classification" in conditions
    needs_wildguard = "wildguard_classification" in conditions
    needs_safeguard = "safeguard_classification" in conditions
    needs_guardreasoner = "guardreasoner_classification" in conditions
    needs_shieldgemma = "shieldgemma_classification" in conditions
    needs_nemotron = "nemotron_classification" in conditions
    needs_grpo = "grpo_classification" in conditions
    needs_main_model = bool(set(conditions) - PRIOR_WORK_CONDITIONS)
    needs_prior_work = (
        needs_llamaguard or needs_wildguard or needs_safeguard
        or needs_guardreasoner or needs_shieldgemma or needs_nemotron
        or needs_grpo
    )

    openrouter_cfg = config.get("openrouter", {}) or {}
    openrouter_models = openrouter_cfg.get("models", []) if openrouter_cfg.get("enabled", False) else []
    openrouter_conditions = openrouter_cfg.get("conditions", ["vanilla_generation"])
    if isinstance(openrouter_conditions, str):
        openrouter_conditions = [openrouter_conditions]
    for cond in openrouter_conditions:
        if cond != "vanilla_generation":
            raise ValueError(
                f"OpenRouter backend currently only supports 'vanilla_generation', got: {cond}"
            )

    llamaguard_model = llamaguard_cfg.get("name", "meta-llama/Llama-Guard-4-12B")
    wildguard_model = wildguard_cfg.get("name", "allenai/wildguard")
    safeguard_model = safeguard_cfg.get("name", "openai/gpt-oss-safeguard-120b")
    guardreasoner_model = guardreasoner_cfg.get("name", "yueliu1999/GuardReasoner-8B")
    shieldgemma_model = shieldgemma_cfg.get("name", "google/shieldgemma-27b")
    nemotron_model = nemotron_cfg.get("name", "nvidia/Nemotron-Content-Safety-Reasoning-4B")
    grpo_model = grpo_cfg.get("name", "iustinsirbu/llama-3.1-8b-grpo-intent-safety")

    # Download missing fine-tuned adapters from W&B registry
    if artifacts_cfg.get("enabled", False) and needs_finetuned_lora:
        registry_project = artifacts_cfg["registry_project"]
        artifact_entity = artifacts_cfg.get("entity", None)
        adapter_keys = {
            "finetuned_generation": finetuned_cfg.get("generation_adapter"),
            "finetuned_classification": finetuned_cfg.get("classification_adapter"),
            "finetuned_reasoning_classification": finetuned_cfg.get("reasoning_classification_adapter"),
            "finetuned_reasoning_human_intent": finetuned_cfg.get("reasoning_human_intent_adapter"),
            "finetuned_reasoning_synthetic_intent": finetuned_cfg.get("reasoning_synthetic_adapter"),
        }
        for cond in conditions:
            adapter_path = adapter_keys.get(cond)
            if adapter_path and not os.path.exists(adapter_path):
                download_adapter(
                    name=Path(adapter_path).name,
                    project=registry_project,
                    target_dir=Path(adapter_path).parent,
                    entity=artifact_entity,
                )

    # Validate adapters for finetuned conditions (after potential download)
    _validate_finetuned_adapters(conditions, finetuned_cfg)

    print(f"\n{'='*60}")
    print(f"Safety Experiment")
    print(f"Conditions: {conditions}")
    print(f"Datasets: {len(datasets_cfg)}")
    for key, label in [
        ("generation_adapter", "Generation adapter"),
        ("classification_adapter", "Classification adapter"),
        ("reasoning_classification_adapter", "Reasoning classification adapter"),
        ("reasoning_generation_adapter", "Reasoning generation adapter"),
    ]:
        path = finetuned_cfg.get(key)
        if path:
            print(f"{label}: {path}")
    if needs_llamaguard:
        print(f"LlamaGuard model: {llamaguard_model}")
    if needs_wildguard:
        print(f"WildGuard model: {wildguard_model}")
    if needs_safeguard:
        print(f"Safeguard model: {safeguard_model}")
    if needs_guardreasoner:
        print(f"GuardReasoner model: {guardreasoner_model}")
    if needs_shieldgemma:
        print(f"ShieldGemma model: {shieldgemma_model}")
    if needs_nemotron:
        print(f"Nemotron model: {nemotron_model} (thinking={nemotron_cfg.get('thinking', True)})")
    if needs_grpo:
        print(f"GRPO model: {grpo_model}")
    print(f"{'='*60}")

    # Initialize wandb
    wandb_run = None
    if wandb_cfg.get("enabled", False) and wandb is not None:
        wandb_run = wandb.init(
            entity=wandb_cfg.get("entity"),
            project=wandb_cfg.get("project", "safety-experiment"),
            name=wandb_cfg.get("run_name"),
            tags=wandb_cfg.get("tags", []) + conditions,
            dir=str(Path(hydra.utils.get_original_cwd()) / wandb_cfg.get("dir", "logs/wandb")),
            mode=wandb_cfg.get("mode", "online"),
            config=config,
        )

    # Shared state across main / prior-work / OpenRouter phases
    all_metrics: Dict[str, dict] = {}
    result_files_written: List[Path] = []
    sampling_params = SamplingParams(
        max_tokens=gen_cfg.get("max_new_tokens", 64),
        temperature=gen_cfg.get("temperature", 0.0),
        top_p=gen_cfg.get("top_p", 1.0),
        top_k=gen_cfg.get("top_k", -1),
        skip_special_tokens=True,
        # Stop on chat-format markers from any model family. The trained format ends
        # with `Prompt harm: harmful|safe`; if the model continues past that into a new
        # turn, it's gone off-script and we want to truncate. These strings never appear
        # inside a valid structured output, so this is a no-op for well-behaved students.
        stop=["<|eot_id|>", "<|end_header_id|>", "<|start_header_id|>", "[INST]", "[/INST]"],
    )

    llm = None
    tokenizer = None
    generation_lora_request = None
    classification_lora_request = None
    reasoning_classification_lora_request = None
    reasoning_human_intent_lora_request = None
    reasoning_synthetic_lora_request = None
    model_slug = model_cfg["name"].replace("/", "_")

    if needs_main_model:
        # Load main model
        print(f"\n=== Loading Model: {model_cfg['name']} ===")
        if use_merged:
            print("Merged model mode: LoRA disabled (weights already baked in)")
        llm_extra: Dict[str, Any] = {"limit_mm_per_prompt": {"image": 0}}
        if needs_finetuned_lora:
            llm_extra["enable_lora"] = True
            llm_extra["max_lora_rank"] = lora_cfg.get("rank", 16)
            llm_extra["max_loras"] = 4
            print("LoRA enabled for fine-tuned conditions")
        llm = _load_vllm(model_cfg["name"], vllm_cfg, **llm_extra)

        (
            generation_lora_request,
            classification_lora_request,
            reasoning_classification_lora_request,
            reasoning_human_intent_lora_request,
            reasoning_synthetic_lora_request,
        ) = _setup_lora_requests(finetuned_cfg) if not use_merged else (None, None, None, None, None)

        # Always load the tokenizer from the base model. LoRA adapters share the
        # base tokenizer, and pulling it from `generation_adapter` (a default Llama
        # SFT path in eval_ood_validation.yaml) silently applies the Llama chat
        # template to every student, producing empty / refusal outputs on Qwen3
        # and Ministral whose native templates use <|im_start|> / [INST] markers.
        tokenizer = AutoTokenizer.from_pretrained(model_cfg["name"])
    else:
        print("\n=== Skipping main model load (no main-model conditions requested) ===")

    if needs_main_model:
        for dataset_idx, data_cfg in enumerate(datasets_cfg):
            dataset_name = data_cfg.get("name", f"dataset_{dataset_idx}")
            print(f"\n{'='*80}")
            print(f"=== Dataset: {dataset_name} ===")
            print(f"{'='*80}")

            test_dataset, harm_column, has_intent = load_test_dataset(data_cfg)

            model_intents = None
            if "vanilla_with_model_intent" in conditions and generation_lora_request is not None:
                model_intents = generate_model_intents(llm, generation_lora_request, test_dataset, sampling_params)

            for condition in conditions:
                print(f"\n--- Condition: {condition} ---")

                if condition == "finetuned_generation" and generation_lora_request is None and not use_merged:
                    print(f"  Skipping {condition} (no generation adapter)")
                    continue
                if condition == "finetuned_classification" and classification_lora_request is None and not use_merged:
                    print(f"  Skipping {condition} (no classification adapter)")
                    continue
                if condition == "finetuned_reasoning_classification" and reasoning_classification_lora_request is None and not use_merged:
                    print(f"  Skipping {condition} (no reasoning classification adapter)")
                    continue
                if condition == "finetuned_reasoning_human_intent" and reasoning_human_intent_lora_request is None and not use_merged:
                    print(f"  Skipping {condition} (no reasoning human intent adapter)")
                    continue
                if condition == "finetuned_reasoning_synthetic_intent" and reasoning_synthetic_lora_request is None and not use_merged:
                    print(f"  Skipping {condition} (no reasoning synthetic adapter)")
                    continue
                if condition in ("vanilla_with_human_intent", "zeroshot_cot_classification_with_intent") and not has_intent:
                    print(f"  Skipping {condition} (no intent column)")
                    continue
                if condition == "vanilla_with_model_intent" and model_intents is None:
                    print(f"  Skipping {condition} (no model intents)")
                    continue

                t0 = time.monotonic()
                results = run_condition_on_dataset(
                    condition=condition,
                    llm=llm,
                    tokenizer=tokenizer,
                    test_dataset=test_dataset,
                    sampling_params=sampling_params,
                    harm_column=harm_column,
                    has_intent=has_intent,
                    model_intents=model_intents,
                    generation_lora_request=generation_lora_request,
                    classification_lora_request=classification_lora_request,
                    reasoning_classification_lora_request=reasoning_classification_lora_request,
                    reasoning_human_intent_lora_request=reasoning_human_intent_lora_request,
                    reasoning_synthetic_lora_request=reasoning_synthetic_lora_request,
                )
                elapsed_s = time.monotonic() - t0

                _process_results(
                    results, condition, dataset_name, data_cfg, paths_cfg,
                    model_slug, wandb_run, all_metrics, result_files_written,
                    elapsed_s=elapsed_s,
                )

    # Free the main model before loading any prior-work baseline
    if needs_main_model and needs_prior_work:
        print(f"\n{'='*60}")
        print("Cleaning up main model before loading baseline models...")
        print(f"{'='*60}")
        del llm
        _free_vllm()

    # Prior-work baselines — each loads its own model, runs on all datasets, then frees
    if needs_llamaguard:
        print(f"\n=== Loading LlamaGuard: {llamaguard_model} ===")
        llamaguard_llm = _load_vllm(llamaguard_model, vllm_cfg, limit_mm_per_prompt={"image": 0})
        llamaguard_tokenizer = AutoTokenizer.from_pretrained(llamaguard_model)
        _run_baseline_on_datasets(
            run_fn=lambda ds, sp, hc: run_llamaguard_classification(llamaguard_llm, llamaguard_tokenizer, ds, sp, hc),
            condition="llamaguard_classification",
            model_slug=llamaguard_model.replace("/", "_"),
            datasets_cfg=datasets_cfg,
            sampling_params=sampling_params,
            paths_cfg=paths_cfg,
            wandb_run=wandb_run,
            all_metrics=all_metrics,
            result_files_written=result_files_written,
        )
        del llamaguard_llm
        _free_vllm()

    if needs_wildguard:
        print(f"\n=== Loading WildGuard: {wildguard_model} ===")
        wildguard_llm = _load_vllm(wildguard_model, vllm_cfg)
        _run_baseline_on_datasets(
            run_fn=lambda ds, sp, hc: run_wildguard_classification(wildguard_llm, ds, sp, hc),
            condition="wildguard_classification",
            model_slug=wildguard_model.replace("/", "_"),
            datasets_cfg=datasets_cfg,
            sampling_params=sampling_params,
            paths_cfg=paths_cfg,
            wandb_run=wandb_run,
            all_metrics=all_metrics,
            result_files_written=result_files_written,
        )
        del wildguard_llm
        _free_vllm()

    if needs_safeguard:
        print(f"\n=== Loading GPT-OSS-Safeguard: {safeguard_model} ===")
        safeguard_llm = _load_vllm(safeguard_model, vllm_cfg, limit_mm_per_prompt={"image": 0})
        safeguard_tokenizer = AutoTokenizer.from_pretrained(safeguard_model)
        _run_baseline_on_datasets(
            run_fn=lambda ds, sp, hc: run_safeguard_classification(safeguard_llm, safeguard_tokenizer, ds, sp, hc),
            condition="safeguard_classification",
            model_slug=safeguard_model.replace("/", "_"),
            datasets_cfg=datasets_cfg,
            sampling_params=sampling_params,
            paths_cfg=paths_cfg,
            wandb_run=wandb_run,
            all_metrics=all_metrics,
            result_files_written=result_files_written,
        )
        del safeguard_llm
        _free_vllm()

    if needs_guardreasoner:
        print(f"\n=== Loading GuardReasoner: {guardreasoner_model} ===")
        guardreasoner_llm = _load_vllm(guardreasoner_model, vllm_cfg)
        _run_baseline_on_datasets(
            run_fn=lambda ds, sp, hc: run_guardreasoner_classification(guardreasoner_llm, ds, sp, hc),
            condition="guardreasoner_classification",
            model_slug=guardreasoner_model.replace("/", "_"),
            datasets_cfg=datasets_cfg,
            sampling_params=sampling_params,
            paths_cfg=paths_cfg,
            wandb_run=wandb_run,
            all_metrics=all_metrics,
            result_files_written=result_files_written,
        )
        del guardreasoner_llm
        _free_vllm()

    if needs_shieldgemma:
        print(f"\n=== Loading ShieldGemma: {shieldgemma_model} ===")
        shieldgemma_llm = _load_vllm(shieldgemma_model, vllm_cfg, limit_mm_per_prompt={"image": 0})
        _run_baseline_on_datasets(
            run_fn=lambda ds, sp, hc: run_shieldgemma_classification(shieldgemma_llm, ds, sp, hc),
            condition="shieldgemma_classification",
            model_slug=shieldgemma_model.replace("/", "_"),
            datasets_cfg=datasets_cfg,
            sampling_params=sampling_params,
            paths_cfg=paths_cfg,
            wandb_run=wandb_run,
            all_metrics=all_metrics,
            result_files_written=result_files_written,
        )
        del shieldgemma_llm
        _free_vllm()

    if needs_nemotron:
        nemotron_thinking = nemotron_cfg.get("thinking", True)
        print(f"\n=== Loading Nemotron: {nemotron_model} ===")
        nemotron_llm = _load_vllm(nemotron_model, vllm_cfg, limit_mm_per_prompt={"image": 0})
        nemotron_tokenizer = AutoTokenizer.from_pretrained(nemotron_model)
        _run_baseline_on_datasets(
            run_fn=lambda ds, sp, hc: run_nemotron_classification(nemotron_llm, nemotron_tokenizer, ds, sp, hc, thinking=nemotron_thinking),
            condition="nemotron_classification",
            model_slug=nemotron_model.replace("/", "_"),
            datasets_cfg=datasets_cfg,
            sampling_params=sampling_params,
            paths_cfg=paths_cfg,
            wandb_run=wandb_run,
            all_metrics=all_metrics,
            result_files_written=result_files_written,
        )
        del nemotron_llm
        _free_vllm()

    if needs_grpo:
        print(f"\n=== Loading GRPO: {grpo_model} ===")
        grpo_llm = _load_vllm(grpo_model, vllm_cfg)
        grpo_tokenizer = AutoTokenizer.from_pretrained(grpo_model)
        _run_baseline_on_datasets(
            run_fn=lambda ds, sp, hc: run_grpo_classification(grpo_llm, grpo_tokenizer, ds, sp, hc),
            condition="grpo_classification",
            model_slug=grpo_model.replace("/", "_"),
            datasets_cfg=datasets_cfg,
            sampling_params=sampling_params,
            paths_cfg=paths_cfg,
            wandb_run=wandb_run,
            all_metrics=all_metrics,
            result_files_written=result_files_written,
        )
        del grpo_llm
        _free_vllm()

    # OpenRouter closed-source baselines (vanilla_generation only) — no GPU usage
    if openrouter_models:
        or_max_tokens = openrouter_cfg.get("max_tokens", gen_cfg.get("max_new_tokens", 2048))
        or_temperature = openrouter_cfg.get("temperature", gen_cfg.get("temperature", 0.0))
        or_max_workers = openrouter_cfg.get("max_workers", 8)
        or_max_retries = openrouter_cfg.get("max_retries", 3)
        or_site_url = openrouter_cfg.get("site_url")
        or_app_name = openrouter_cfg.get("app_name")

        print(f"\n{'='*60}")
        print(f"OpenRouter baselines: {len(openrouter_models)} model(s) × "
              f"{len(openrouter_conditions)} condition(s)")
        print(f"{'='*60}")

        for or_model in openrouter_models:
            print(f"\n=== OpenRouter model: {or_model} ===")
            for condition in openrouter_conditions:
                _run_baseline_on_datasets(
                    run_fn=lambda ds, sp, hc, _m=or_model: run_vanilla_generation_openrouter(
                        model_name=_m,
                        test_dataset=ds,
                        harm_column=hc,
                        max_tokens=or_max_tokens,
                        temperature=or_temperature,
                        max_workers=or_max_workers,
                        max_retries=or_max_retries,
                        site_url=or_site_url,
                        app_name=or_app_name,
                    ),
                    condition=condition,
                    model_slug=or_model.replace("/", "_"),
                    datasets_cfg=datasets_cfg,
                    sampling_params=sampling_params,
                    paths_cfg=paths_cfg,
                    wandb_run=wandb_run,
                    all_metrics=all_metrics,
                    result_files_written=result_files_written,
                )

    # Per-(model, condition) suite summary written locally so downstream notebooks
    # don't need W&B. Written before artifact upload so it gets bundled too.
    # Grouping by (model_slug, condition) matches the W&B backfill convention
    # (scripts/backfill_eval_timing.py) and keeps eval-by-adapter distinct when
    # a single run evaluates multiple conditions on the same base model.
    if all_metrics:
        per_group: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for key, metrics in all_metrics.items():
            ms = metrics.get("model_slug", "unknown")
            cond = metrics.get("condition", "unknown")
            per_group.setdefault((ms, cond), {"runs": {}, "datasets": set()})
            per_group[(ms, cond)]["runs"][key] = metrics
            per_group[(ms, cond)]["datasets"].add(metrics.get("dataset_name", "unknown"))

        summary_dir = Path(paths_cfg.get("output_dir") or "data/safety_experiment")
        summary_dir.mkdir(parents=True, exist_ok=True)
        for (model_slug, condition), payload in per_group.items():
            runs = payload["runs"]
            suite_elapsed = sum(float(m.get("elapsed_s", 0.0)) for m in runs.values())
            suite_tokens = sum(int(m.get("total_tokens", 0)) for m in runs.values())
            suite_total = sum(int(m.get("total", 0)) for m in runs.values())
            suite_tps = suite_tokens / suite_elapsed if suite_elapsed > 0 else 0.0
            summary = {
                "model_slug": model_slug,
                "condition": condition,
                "datasets": sorted(payload["datasets"]),
                "conditions": [condition],
                "runs": runs,
                "suite": {
                    "elapsed_s": suite_elapsed,
                    "total_tokens": suite_tokens,
                    "total_examples": suite_total,
                    "tokens_per_second": suite_tps,
                },
            }
            summary_path = summary_dir / f"{model_slug}_{condition}_suite_summary.json"
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            print(f"✓ Wrote suite summary for {model_slug} / {condition}: {summary_path}")
            print(f"    Suite: {suite_total:,} ex  |  {suite_tokens:,} tok  |  {suite_elapsed:.1f}s  |  {suite_tps:.1f} tok/s")
            result_files_written.append(summary_path)

    # Upload results artifact if configured
    results_name = artifacts_cfg.get("results_name")
    if results_name and result_files_written and wandb_run is not None:
        upload_results(
            files=result_files_written,
            name=results_name,
            project=artifacts_cfg["registry_project"],
            entity=artifacts_cfg.get("entity"),
            run=wandb_run,
        )

    # Summary
    print(f"\n{'='*80}")
    print("=== EXPERIMENT SUMMARY ===")
    print(f"{'='*80}")
    for key, metrics in all_metrics.items():
        print(f"\n{key}:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")
        print(f"  Tokens: {metrics.get('total_tokens', 0):,}  |  Time: {metrics.get('elapsed_s', 0):.1f}s  |  Throughput: {metrics.get('tokens_per_second', 0):.1f} tok/s")

    if wandb_run is not None:
        wandb.finish()

    print(f"\n{'='*60}")
    print("Experiment complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
