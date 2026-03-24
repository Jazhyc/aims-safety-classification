"""
Training script for WildGuardMix binary classifier using ModernBERT.

Usage:
    python scripts/train.py
    python scripts/train.py training.per_device_train_batch_size=128
    python scripts/train.py --multirun training.learning_rate=5e-5,1e-4,2e-4
"""

import os

from intention_jailbreak.ensemble.deepensembleclassifier import DeepEnsembleClassifier
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import torch
import wandb
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

from intention_jailbreak.dataset import wildguardmix
from intention_jailbreak.training import (
    set_all_seeds,
    prepare_classification_data,
    tokenize_dataset,
    compute_classification_metrics,
    compute_sample_weights,
    print_gpu_info,
    WeightedTrainer,
)


@hydra.main(version_base=None, config_path="../configs", config_name="train_config")
def main(cfg: DictConfig):
    """Main training function."""
    
    set_all_seeds(cfg.seed)
    print_gpu_info()
    
    # Initialize WandB
    if cfg.training.report_to == "wandb":
        wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            dir=cfg.wandb.dir,
            name=cfg.wandb.run_name,
            config=OmegaConf.to_container(cfg, resolve=True),
            tags=cfg.wandb.tags,
        )
    
    # Load data (filtering happens before split if enabled)
    print("Loading dataset...")
    train_df, annotation_df = wildguardmix.load_and_split(
        subset=cfg.dataset.subset,
        test_size=cfg.dataset.test_size,
        random_state=cfg.dataset.random_state,
        filter_english=cfg.dataset.get('filter_english', False),
        text_column=cfg.dataset.text_column,
        language_cache_dir=cfg.dataset.get('language_cache_dir', 'data/cache')
    )
    
    print(f"Training split: {len(train_df)} samples")
    print(f"Annotation split: {len(annotation_df)} samples")
    
    # Prepare data - split train_df into train/val/test (90:10:10 of the 90% training portion)
    print("Preparing data...")
    train_dataset, val_dataset, test_dataset, train_df_processed, val_df_processed, test_df_processed = prepare_classification_data(
        train_df=train_df,
        test_df=None,
        val_size=cfg.dataset.val_size,
        test_size=cfg.dataset.get('test_size_from_train', 0.1),
        label_column=cfg.dataset.label_column,
        text_column=cfg.dataset.text_column,
        positive_label=cfg.dataset.positive_label,
        random_state=cfg.dataset.random_state
    )
    
    # Load model
    print(f"Loading tokenizer for: {cfg.model.name}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)
    
    # Compute sample weights if enabled (before tokenization)
    use_label_weights = cfg.dataset.get('use_label_weights', False)
    use_subcategory_weights = cfg.dataset.get('use_subcategory_weights', False)
    use_any_weights = use_label_weights or use_subcategory_weights
    
    if use_any_weights:
        weight_desc = []
        if use_label_weights:
            weight_desc.append("labels")
        if use_subcategory_weights:
            weight_desc.append(f"'{cfg.dataset.class_weight_column}'")
        print(f"Computing sample weights based on {' and '.join(weight_desc)}...")
        
        train_weights = compute_sample_weights(
            train_df_processed,
            label_column='label',
            weight_column=cfg.dataset.class_weight_column if use_subcategory_weights else None,
            use_label_weights=use_label_weights,
            use_subcategory_weights=use_subcategory_weights
        )
        # Add weights to dataset
        train_dataset = train_dataset.add_column("weight", train_weights.tolist())
        # For validation, we typically don't use weights, but add uniform weights
        val_weights = np.ones(len(val_dataset))
        val_dataset = val_dataset.add_column("weight", val_weights.tolist())
    
    # Save test set for later evaluation
    print("Saving test set for evaluation...")
    test_output_dir = Path("data/test_predictions")
    test_output_dir.mkdir(parents=True, exist_ok=True)
    test_df_processed.to_parquet(test_output_dir / "held_out_test_set.parquet", index=False)
    print(f"Test set saved to: {test_output_dir / 'held_out_test_set.parquet'}")
    
    # Tokenize (no test set tokenization needed here)
    print("Tokenizing...")
    train_dataset = tokenize_dataset(train_dataset, tokenizer, cfg.model.max_length, cfg.dataset.text_column, num_proc=cfg.dataset.num_proc)
    val_dataset = tokenize_dataset(val_dataset, tokenizer, cfg.model.max_length, cfg.dataset.text_column, num_proc=cfg.dataset.num_proc)
    
    print(f"Loading model for: {cfg.model.name}")
    model = None
    if cfg.ensemble.enabled:
        print(f"Training an ensemble of {cfg.ensemble.num_models} models")
        model = DeepEnsembleClassifier(
            model_fn=lambda: AutoModelForSequenceClassification.from_pretrained(
                    cfg.model.name,
                    dtype=torch.bfloat16,
                    **{k: v for k, v in cfg.model.items() if k != 'name' and k != 'max_length'}
                ),
            num_models=cfg.ensemble.num_models
    )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            cfg.model.name,
            dtype=torch.bfloat16,
            **{k: v for k, v in cfg.model.items() if k != 'name' and k != 'max_length'}
        )

    # Setup training
    training_args = TrainingArguments(**cfg.training)
    
    # Use WeightedTrainer if weights are specified, otherwise standard Trainer
    trainer_class = WeightedTrainer if use_any_weights else Trainer
    # This is the simplest approach just train each model using the same class and then combine
    if cfg.ensemble.enabled:
        for idx, ensemble_member in enumerate(model.models):
            trainer = trainer_class(
                    model=ensemble_member,
                    args=training_args,
                    train_dataset=train_dataset,
                    eval_dataset=val_dataset,
                    processing_class=tokenizer,
                    compute_metrics=compute_classification_metrics,
                )
            print(f"Training model {idx}")
            trainer.train()
            trainer.evaluate()
    else:
        trainer = trainer_class(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                processing_class=tokenizer,
                compute_metrics=compute_classification_metrics,
            )
        print("Training...")
        trainer.train()
        trainer.evaluate()
    
    # Save model
    print("Saving model...")
    final_model_dir = Path(cfg.training.output_dir) / "final_model"
    
    # Use safe serialization only if not an ensemble
    if cfg.ensemble.enabled:
        model.save_pretrained(final_model_dir)
    else:
        model.save_pretrained(final_model_dir, safe_serialization=True)
    tokenizer.save_pretrained(final_model_dir)
    print(f"Model saved to: {final_model_dir}")
    
    if cfg.training.report_to == "wandb":
        wandb.finish()
    
    print("Training complete!")


if __name__ == "__main__":
    main()
