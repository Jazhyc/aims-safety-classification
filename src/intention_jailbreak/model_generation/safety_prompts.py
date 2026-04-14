"""
Constants, schemas, and condition registry for the safety classifier evaluation.

System prompt strings are loaded from configs/prompt_templates.yaml (safety_classifier section)
via prompt_templates.py and re-exported here for backwards compatibility.
"""

from .prompt_templates import (
    CLASSIFICATION_SYSTEM_PROMPT,
    CLASSIFICATION_WITH_INTENT_SYSTEM_PROMPT,
    GENERATION_SYSTEM_PROMPT,
    GENERATION_JSON_SYSTEM_PROMPT,
    COT_CLASSIFICATION_SYSTEM_PROMPT,
    COT_GENERATION_SYSTEM_PROMPT,
    COT_CLASSIFICATION_WITH_INTENT_SYSTEM_PROMPT,
    WILDGUARD_TEMPLATE as _WILDGUARD_TEMPLATE,
    SAFEGUARD_SYSTEM_PROMPT as _SAFEGUARD_SYSTEM_PROMPT,
)

HARM_CATEGORIES = [
    "Privacy",
    "Misinformation",
    "Harmful Language",
    "Malicious Uses",
    "Other harms",
    "Benign",
]

BINARY_LABELS = ["harmful", "safe"]

# ---------------------------------------------------------------------------
# JSON schemas for CoT structured output
# ---------------------------------------------------------------------------

COT_CLASSIFICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string", "description": "Step-by-step analysis of the prompt's potential intent and harms"},
        "harm": {"type": "string", "enum": BINARY_LABELS, "description": "Safety classification: harmful or safe"},
    },
    "required": ["reasoning", "harm"],
    "additionalProperties": False,
}

COT_GENERATION_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string", "description": "Step-by-step analysis of the prompt"},
        "intent": {"type": "string", "description": "The inferred intent behind the user's prompt"},
        "harm": {"type": "string", "enum": BINARY_LABELS, "description": "Safety classification: harmful or safe"},
    },
    "required": ["reasoning", "intent", "harm"],
    "additionalProperties": False,
}

# ---------------------------------------------------------------------------
# Supported conditions registry
# ---------------------------------------------------------------------------

VALID_CONDITIONS = [
    "finetuned_generation",                    # Fine-tuned: prompt -> intent + harm
    "finetuned_classification",                # Fine-tuned: prompt -> harm only
    "vanilla_classification",                  # Vanilla: prompt -> harm (with system prompt)
    "vanilla_generation",                      # Vanilla: prompt -> intent + harm (with system prompt)
    "vanilla_with_human_intent",               # Vanilla: prompt + human intent -> harm
    "vanilla_with_model_intent",               # Vanilla: prompt + model intent -> harm
    "llamaguard_classification",               # LlamaGuard 4 baseline: prompt -> harm
    "wildguard_classification",                # WildGuard baseline: prompt -> harm
    "safeguard_classification",                # GPT-OSS-Safeguard baseline: prompt -> harm
    "zeroshot_cot_classification",             # CoT: prompt -> reasoning + harm
    "zeroshot_cot_generation",                 # CoT: prompt -> reasoning + intent + harm
    "zeroshot_cot_classification_with_intent", # CoT: prompt + human intent -> reasoning + harm
    "finetuned_reasoning_classification",      # Distilled (MODE D): prompt -> reasoning + harm
    "finetuned_reasoning_human_intent",        # Distilled (MODE E): prompt -> reasoning + human-annotated intent + harm
    "finetuned_reasoning_synthetic_intent",    # Distilled (MODE F): prompt -> reasoning + model-generated intent + harm
]
