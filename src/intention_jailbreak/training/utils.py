"""Utility functions for setting seeds and printing system info."""

import numpy as np
import torch
from typing import Dict, Optional
from transformers import set_seed, Trainer

from intention_jailbreak.ensemble.deepensembleclassifier import DeepEnsembleClassifier

torch.set_float32_matmul_precision('high')

def set_all_seeds(seed: int):
    """
    Set all random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    set_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def print_gpu_info():
    """Print GPU information if available."""
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"CUDA: {torch.version.cuda}")
    else:
        print("⚠️  No GPU available, training will use CPU")


class WeightedTrainer(Trainer):
    """
    Custom Trainer that applies per-sample weights to the loss function.
    
    Expects the dataset to have a 'weight' column with per-sample weights.
    
    Args:
        *args, **kwargs: Passed to parent Trainer
    """
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute weighted cross-entropy loss using per-sample weights from dataset.
        """
        labels = inputs.pop("labels")
        # Extract weights if present in inputs
        weights = inputs.pop("weight", None)
        
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # Compute per-sample loss (reduction='none')
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        per_sample_loss = loss_fct(logits, labels)
        
        # Apply sample weights if provided
        if weights is not None:
            weights = weights.to(per_sample_loss.device)
            weighted_loss = per_sample_loss * weights
            loss = weighted_loss.mean()
        else:
            loss = per_sample_loss.mean()
        
        return (loss, outputs) if return_outputs else loss
