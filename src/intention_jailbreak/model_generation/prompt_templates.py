"""
Shared prompt templates for student model training and inference.

All template strings live in configs/prompt_templates.yaml.  This module
loads them once at import time and exposes:

  - TAXONOMY          -- raw taxonomy string
  - PREAMBLE          -- preamble with {taxonomy} already substituted
  - OUTPUT_FORMAT_WITHOUT_INTENT
  - OUTPUT_FORMAT_WITH_INTENT
  - TEACHER_GROUND_TRUTH   -- dict with keys 'without_intent' and 'with_intent'

  - build_student_prompt(user_prompt, with_intent) -- full input for the student model

Safety classifier prompts (from safety_classifier section):
  - CLASSIFICATION_SYSTEM_PROMPT
  - CLASSIFICATION_WITH_INTENT_SYSTEM_PROMPT
  - GENERATION_SYSTEM_PROMPT
  - GENERATION_JSON_SYSTEM_PROMPT
  - COT_CLASSIFICATION_SYSTEM_PROMPT
  - COT_GENERATION_SYSTEM_PROMPT
  - COT_CLASSIFICATION_WITH_INTENT_SYSTEM_PROMPT
  - WILDGUARD_TEMPLATE
  - SAFEGUARD_SYSTEM_PROMPT

The student model (fine-tuned via reasoning distillation) receives the same
context at both training time (causal.py) and evaluation time (safety_experiment.py).

Usage
-----
from intention_jailbreak.model_generation.prompt_templates import build_student_prompt

# Classification student (without_intent): Reasoning + Prompt harm
prompt = build_student_prompt(user_prompt, with_intent=False)

# Generation student (with_intent): Reasoning + Intent + Prompt harm
prompt = build_student_prompt(user_prompt, with_intent=True)
"""

from pathlib import Path
import yaml

# Locate the YAML file relative to this module (project_root/configs/prompt_templates.yaml)
_HERE = Path(__file__).resolve()
_PROJECT_ROOT = _HERE.parent.parent.parent.parent  # src/intention_jailbreak/model_generation/ -> project root
_TEMPLATES_PATH = _PROJECT_ROOT / "configs" / "prompt_templates.yaml"


def _load() -> dict:
    with open(_TEMPLATES_PATH, "r") as f:
        return yaml.safe_load(f)


_t = _load()

# ── Public constants ──────────────────────────────────────────────────────────

TAXONOMY: str = _t["taxonomy"].strip()

# PREAMBLE has {taxonomy} substituted so callers don't need to do it.
PREAMBLE: str = _t["preamble"].format(taxonomy=TAXONOMY).strip()

OUTPUT_FORMAT_WITHOUT_INTENT: str = _t["output_format_without_intent"].strip()
OUTPUT_FORMAT_WITH_INTENT: str = _t["output_format_with_intent"].strip()

# Teacher-facing ground-truth instructions (used by generate_reasoning_traces.py).
# Keys: 'without_intent', 'with_intent'.  Each value is a format string with
# placeholders {annotator_harmful_label} and optionally {intent}.
TEACHER_GROUND_TRUTH: dict = _t["teacher_ground_truth"]

# ── Safety classifier prompts (used by eval_safety_classifier.py) ────────────
_sc = _t["safety_classifier"]

CLASSIFICATION_SYSTEM_PROMPT: str = _sc["classification_system_prompt"].strip()
CLASSIFICATION_WITH_INTENT_SYSTEM_PROMPT: str = _sc["classification_with_intent_system_prompt"].strip()
GENERATION_SYSTEM_PROMPT: str = _sc["generation_system_prompt"].strip()
GENERATION_JSON_SYSTEM_PROMPT: str = _sc["generation_json_system_prompt"].strip()
COT_CLASSIFICATION_SYSTEM_PROMPT: str = _sc["cot_classification_system_prompt"].strip()
COT_GENERATION_SYSTEM_PROMPT: str = _sc["cot_generation_system_prompt"].strip()
COT_CLASSIFICATION_WITH_INTENT_SYSTEM_PROMPT: str = _sc["cot_classification_with_intent_system_prompt"].strip()
WILDGUARD_TEMPLATE: str = _sc["wildguard_template"].strip()
SAFEGUARD_SYSTEM_PROMPT: str = _sc["safeguard_system_prompt"].strip()


# ── Public helpers ────────────────────────────────────────────────────────────

def build_student_prompt(user_prompt: str, with_intent: bool = False) -> str:
    """
    Build the full input prompt for the student model.

    This is the complete context the student sees at both training and inference
    time.  The model is expected to produce the completion:

        Reasoning: <step-by-step analysis>
        [Prompt intent: <intent>]   <- only when with_intent=True
        Prompt harm: <harmful/unharmful>

    Args:
        user_prompt:  The raw human request to classify.
        with_intent:  If True, the output format includes a Prompt intent field.

    Returns:
        The complete formatted input string (no trailing newline).
    """
    output_format = OUTPUT_FORMAT_WITH_INTENT if with_intent else OUTPUT_FORMAT_WITHOUT_INTENT
    return f"{PREAMBLE}\n\nHuman user:\n{user_prompt}\n\n{output_format}"
