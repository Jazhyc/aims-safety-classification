"""
Evaluation script for intent generation from hyperparameter sweep model.

This script:
1. Loads a LORA model using VLLM
2. Evaluates it on the test set of Jazhyc/wildguard-annotated-intents
3. Generates predictions and extracts harm labels
4. Saves results with prompts, generated intents, true intents, predicted labels, and true labels

Usage:
    python scripts/eval_intent_generation.py --config-name=eval_intent_generation
"""

import os
import json
from pathlib import Path
from datasets import load_dataset
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from transformers import AutoTokenizer
import hydra
from omegaconf import DictConfig, OmegaConf
import re

try:
    import wandb
except ImportError:
    wandb = None

from dotenv import load_dotenv
load_dotenv()


def extract_intent_and_harm(generated_text):
    """
    Extract intent and harm label from generated text.
    
    The model generates binary labels automatically in formats like:
    - "Intent: <text>; Harm: <label>"
    - "Intent: <text>"
    - Just the intent text directly
    
    Args:
        generated_text: Raw generated text from model
    
    Returns:
        tuple: (intent, harm_label)
    """
    generated_text = generated_text.strip()
    
    # Try to parse structured format: "Intent: <text>; Harm: <label>"
    intent_harm_pattern = r"Intent:\s*(.+?);\s*Harm:\s*(.+?)(?:\n|$)"
    match = re.search(intent_harm_pattern, generated_text, re.IGNORECASE | re.DOTALL)
    
    if match:
        intent = match.group(1).strip()
        harm = match.group(2).strip().lower()
        return intent, harm
    
    # Try to parse intent-only format: "Intent: <text>"
    intent_pattern = r"Intent:\s*(.+?)(?:\n|$)"
    match = re.search(intent_pattern, generated_text, re.IGNORECASE | re.DOTALL)
    
    if match:
        intent = match.group(1).strip()
        return intent, None
    
    # If no pattern matches, treat entire text as intent
    return generated_text, None


def load_and_split_dataset(dataset_name, subset, split, harm_column, intent_column, 
                          use_manual_split, seed, test_size, val_size):
    """
    Load dataset and create train/val/test splits.
    
    NOTE: No label processing is applied to test data.
    The model generates binary labels automatically, and we compare with raw test labels.
    
    Args:
        dataset_name: HuggingFace dataset name
        subset: Dataset subset (e.g., 'wildguardtest' for wildguardmix, None for annotated-intents)
        split: Dataset split to load (e.g., 'test' for wildguardmix, 'train' for annotated-intents)
        harm_column: Name of the harm label column
        intent_column: Name of the intent column (can be None)
        use_manual_split: Whether to manually create train/val/test splits
        seed: Random seed for splitting
        test_size: Test set proportion (if manual split)
        val_size: Validation set proportion (if manual split)
    
    Returns:
        tuple: (test_dataset, harm_column_name)
    """
    print(f"Loading dataset: {dataset_name}")
    if subset:
        print(f"  Subset: {subset}")
        dataset = load_dataset(dataset_name, subset, split=split)
    else:
        dataset = load_dataset(dataset_name, split=split)
    
    # Rename columns to match expected format
    rename_map = {
        "ID": "id",
        "Prompt": "prompt",
    }
    
    # Add intent column to rename map if specified
    if intent_column and intent_column in dataset.column_names:
        rename_map[intent_column] = "intent"
    
    existing_renames = {k: v for k, v in rename_map.items() if k in dataset.column_names}
    if existing_renames:
        dataset = dataset.rename_columns(existing_renames)
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Harm label column: {harm_column}")
    
    if use_manual_split:
        # Create splits manually (for datasets without dedicated test split)
        print("Using manual train/val/test split")
        
        # First split: train+val vs test
        train_test_split = dataset.train_test_split(test_size=test_size, seed=seed)
        test_dataset = train_test_split["test"]
        
        print(f"Test split size: {len(test_dataset)}")
    else:
        # Use the dataset as-is (already a dedicated test split)
        print("Using dedicated test split from dataset")
        test_dataset = dataset
    
    return test_dataset, harm_column


@hydra.main(version_base=None, config_path="../configs/model_generation", config_name="eval_intent_generation")
def main(cfg: DictConfig):
    """
    Main evaluation function.
    
    Args:
        cfg: Hydra configuration object
    """
    # Convert config to dict for easier access
    config = OmegaConf.to_container(cfg, resolve=True)
    
    # Extract configuration
    datasets_cfg = config.get("datasets", [])
    model_cfg = config.get("model", {})
    lora_cfg = config.get("lora", {})
    gen_cfg = config.get("generation", {})
    paths_cfg = config.get("paths", {})
    vllm_cfg = config.get("vllm", {})
    wandb_cfg = config.get("wandb", {})
    
    # Initialize wandb if enabled
    wandb_run = None
    if wandb_cfg.get("enabled", False) and wandb is not None:
        wandb_run = wandb.init(
            entity=wandb_cfg.get("entity"),
            project=wandb_cfg.get("project", "intention-jailbreak-eval"),
            tags=wandb_cfg.get("tags", []),
            dir=wandb_cfg.get("dir", "logs/wandb"),
            mode=wandb_cfg.get("mode", "online"),
            config=config,
        )
        print("\n✓ WandB initialized")
    
    # Load model once (used for all datasets)
    print(f"\n=== Loading Model ===")
    print(f"Base model: {model_cfg['name']}")
    print(f"LORA adapter: {paths_cfg['model_path']}")
    
    # Initialize VLLM with LORA support
    llm = LLM(
        model=model_cfg["name"],
        enable_lora=True,
        max_lora_rank=lora_cfg["rank"],
        max_loras=1,
        limit_mm_per_prompt={"image": 0},
        gpu_memory_utilization=vllm_cfg["gpu_memory_utilization"],
        max_model_len=vllm_cfg["max_model_len"],
        dtype=vllm_cfg["dtype"],
        enforce_eager=vllm_cfg["enforce_eager"],
    )
    
    # Create LORA request
    lora_request = LoRARequest("intent_lora", 1, paths_cfg["model_path"])
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(paths_cfg["model_path"])
    
    # Set up sampling parameters
    sampling_params = SamplingParams(
        max_tokens=gen_cfg["max_new_tokens"],
        temperature=gen_cfg["temperature"],
        top_p=gen_cfg["top_p"],
        top_k=gen_cfg["top_k"],
        skip_special_tokens=True,
    )
    
    # Iterate through all datasets
    for dataset_idx, data_cfg in enumerate(datasets_cfg):
        dataset_name = data_cfg.get("name", f"dataset_{dataset_idx}")
        print(f"\n{'='*80}")
        print(f"=== Evaluating on: {dataset_name} ===")
        print(f"{'='*80}")
        
        # Create output directory
        output_dir = Path(data_cfg["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load and split dataset (NO label processing)
        print("\n=== Loading Dataset ===")
        test_dataset, harm_column = load_and_split_dataset(
            data_cfg["dataset_name"],
            data_cfg.get("subset"),
            data_cfg.get("split", "train"),
            data_cfg.get("harm_column", "Annotator Harm"),
            data_cfg.get("intent_column", "Intent"),
            data_cfg.get("use_manual_split", True),
            data_cfg["seed"],
            data_cfg["test_size"],
            data_cfg["val_size"]
        )
        
        # Prepare prompts
        print(f"\n=== Generating Predictions ===")
        print(f"Test set size: {len(test_dataset)}")
        test_examples = list(test_dataset)
        prompts = [f"{ex['prompt']}\n" for ex in test_examples]
        
        # Generate predictions with VLLM
        outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)
        
        # Process and save results
        results_path = output_dir / data_cfg["predictions_file"]
        print(f"\nSaving results to {results_path}")
        
        results = []
        with open(results_path, "w", encoding="utf-8") as f:
            for ex, output in zip(test_examples, outputs):
                # Extract generated text
                generated_text = output.outputs[0].text.strip()
                
                # Parse intent and harm from generated text (NO processing)
                predicted_intent, predicted_harm = extract_intent_and_harm(generated_text)
                
                # Get true values (NO processing - keep original labels)
                true_intent = ex.get("intent", None)  # May be None for datasets without intent column
                true_harm = ex.get(harm_column, None)
                
                # Create result entry
                result = {
                    "id": ex.get("id", ""),
                    "prompt": ex["prompt"],
                    "generated_intent": predicted_intent,
                    "true_intent": true_intent,
                    "predicted_harm": predicted_harm,
                    "true_harm": true_harm,
                    "raw_generation": generated_text,
                }
                
                results.append(result)
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
        
        print(f"✓ Saved {len(results)} predictions")
        
        # Compute and display basic statistics
        print(f"\n=== Evaluation Statistics for {dataset_name} ===")
        print(f"Total examples: {len(results)}")
        
        # Count how many have harm predictions
        harm_predictions = [r for r in results if r["predicted_harm"] is not None]
        print(f"Examples with harm prediction: {len(harm_predictions)} ({len(harm_predictions)/len(results)*100:.1f}%)")
        
        # Count harm label distribution (if available)
        stats = {
            f"{dataset_name}/total_examples": len(results),
            f"{dataset_name}/examples_with_harm_prediction": len(harm_predictions),
            f"{dataset_name}/harm_prediction_rate": len(harm_predictions)/len(results) if results else 0,
        }
        
        if harm_predictions:
            from collections import Counter
            predicted_harm_dist = Counter([r["predicted_harm"] for r in harm_predictions])
            true_harm_dist = Counter([r["true_harm"] for r in results if r["true_harm"] is not None])
            
            print("\nPredicted harm distribution:")
            for label, count in sorted(predicted_harm_dist.items()):
                print(f"  {label}: {count} ({count/len(harm_predictions)*100:.1f}%)")
                stats[f"{dataset_name}/predicted_{label}_count"] = count
                stats[f"{dataset_name}/predicted_{label}_pct"] = count/len(harm_predictions)
            
            print("\nTrue harm distribution (original labels):")
            for label, count in sorted(true_harm_dist.items()):
                total = len([r for r in results if r["true_harm"] is not None])
                print(f"  {label}: {count} ({count/total*100:.1f}%)")
                stats[f"{dataset_name}/true_{label}_count"] = count
                stats[f"{dataset_name}/true_{label}_pct"] = count/total if total > 0 else 0
        
        # Log to wandb if enabled
        if wandb_run is not None:
            print(f"\n=== Logging {dataset_name} to WandB ===")
            
            # Log statistics
            wandb.log(stats)
            
            # Save predictions as artifact
            artifact = wandb.Artifact(
                name=f"predictions_{dataset_name}",
                type="predictions",
                description=f"Intent generation predictions on {data_cfg['dataset_name']}",
                metadata={
                    "model_path": paths_cfg["model_path"],
                    "base_model": model_cfg["name"],
                    "dataset": data_cfg["dataset_name"],
                    "subset": data_cfg.get("subset"),
                    "split": data_cfg.get("split"),
                    "total_examples": len(results),
                }
            )
            artifact.add_file(str(results_path))
            wandb_run.log_artifact(artifact)
            
            # Create wandb Table for visualization
            table_data = []
            for r in results[:100]:  # First 100 examples for visualization
                table_data.append([
                    r["id"],
                    r["prompt"][:200],  # Truncate long prompts
                    r["generated_intent"][:200] if r["generated_intent"] else "",
                    r["true_intent"][:200] if r["true_intent"] else "",
                    r["predicted_harm"],
                    r["true_harm"],
                ])
            
            predictions_table = wandb.Table(
                columns=["ID", "Prompt", "Generated Intent", "True Intent", "Predicted Harm", "True Harm"],
                data=table_data
            )
            wandb.log({f"{dataset_name}/predictions_sample": predictions_table})
            
            print(f"✓ Logged predictions and statistics to WandB")
    
    # Finish wandb run after all datasets
    if wandb_run is not None:
        wandb.finish()
    
    print(f"\n{'='*80}")
    print("Evaluation complete for all datasets!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
