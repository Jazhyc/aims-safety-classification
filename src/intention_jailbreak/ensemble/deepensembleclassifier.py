import json
from pathlib import Path
from typing import Callable, Optional
import torch
from torch import nn
from dataclasses import dataclass

@dataclass
class EnsembleOutput:
    """Output class matching transformers.modeling_outputs.SequenceClassifierOutput"""
    logits: torch.Tensor
    probs: torch.Tensor
    individual_logits: Optional[torch.Tensor] = None  # Shape: (num_models, batch, num_classes)


class DeepEnsembleClassifier(nn.Module):
    """
    An implementation of an ensemble for providing a rough estimate of the Bayesian predictive posterior
    distribution p(y|x, D).
    It is a simple wrapper class that instantiates N models and averages their predictions.
    """
    def __init__(self, model_fn: Callable, num_models: int):
        super().__init__()
        self.models = nn.ModuleList([model_fn() for _ in range(num_models)])
        self.num_models = num_models
    
    def forward(self, **kwargs):
        """
        Forward pass through all models in the ensemble.

        Returns: EnsembleOutput
        """
        # Get outputs from all models
        all_logits = []
        all_probs = []
        for model in self.models:
            outputs = model(**kwargs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            all_logits.append(outputs.logits)
            all_probs.append(probs)
        
        # Stack and average logits
        stacked_logits = torch.stack(all_logits)  # Shape: (num_models, batch, num_classes)
        mean_logits = stacked_logits.mean(dim=0)  # Shape: (batch, num_classes) -- for compatibility, not sure how theoretically grounded this is.
        mean_probs = torch.stack(all_probs).mean(dim=0)
        
        return EnsembleOutput(
            logits=mean_logits,
            individual_logits=stacked_logits,
            probs=mean_probs
        )

    def save_pretrained(self, save_directory):
        """Save each model in the ensemble to separate directories."""
        Path(save_directory).mkdir(parents=True, exist_ok=True)
        
        for i, model in enumerate(self.models):
            model_dir = Path(save_directory) / f"model_{i}"
            model.save_pretrained(model_dir, safe_serialization=True)
        
        # Save ensemble configuration
        config = {
            "num_models": self.num_models,
            "ensemble_type": "DeepEnsembleClassifier"
        }
        with open(Path(save_directory) / "ensemble_config.json", "w") as f:
            json.dump(config, f)
    
    @classmethod
    def from_pretrained(cls, save_directory, model_class, **model_kwargs):
        """
        Load ensemble from saved models.
        
        Args:
            save_directory: Directory containing the saved ensemble models
            model_class: The model class to use (e.g., AutoModelForSequenceClassification)
            **model_kwargs: Additional keyword arguments to pass to model.from_pretrained()
                           (e.g., dtype=torch.bfloat16)
        """
        save_path = Path(save_directory)
        
        # Load configuration
        with open(save_path / "ensemble_config.json", "r") as f:
            config = json.load(f)
        
        num_models = config["num_models"]
        
        # Create ensemble with a dummy model_fn (will be replaced)
        dummy_fn = lambda: None
        ensemble = cls(dummy_fn, num_models)
        
        # Load each model with the provided kwargs
        for i in range(num_models):
            model_dir = save_path / f"model_{i}"
            ensemble.models[i] = model_class.from_pretrained(model_dir, **model_kwargs)
        
        return ensemble
