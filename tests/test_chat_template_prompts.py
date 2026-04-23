"""Tests for chat-template prompt building and assistant-turn parsing."""

import pytest
from intention_jailbreak.model_generation.prompt_templates import (
    build_student_messages,
    PREAMBLE,
    PREAMBLE_NO_INTENT,
    OUTPUT_FORMAT_NO_INTENT,
    OUTPUT_FORMAT_WITH_INTENT,
)
from intention_jailbreak.model_generation.causal import (
    format_completion_with_reasoning,
    parse_reasoning_output,
)
from intention_jailbreak.model_generation.safety_classifier import _parse_reasoning_output


# ── Minimal mock tokenizer ────────────────────────────────────────────────────

class _MockTokenizer:
    """Simulates tokenizer.apply_chat_template(tokenize=False) without a real model."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        assert not tokenize, "tests only check the string-output path"
        parts = [f"[{m['role']}]{m['content']}[/{m['role']}]" for m in messages]
        result = "".join(parts)
        if add_generation_prompt:
            result += "[assistant]"
        return result


# ── build_student_messages: structure ────────────────────────────────────────

def test_returns_two_messages():
    msgs = build_student_messages("some prompt", condition="no_intent")
    assert len(msgs) == 2


def test_roles_are_system_and_user():
    msgs = build_student_messages("some prompt", condition="no_intent")
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"


def test_user_content_is_the_raw_prompt():
    user_prompt = "Can you help me with something dangerous?"
    for condition in ("no_intent", "synthetic_intent", "human_intent"):
        msgs = build_student_messages(user_prompt, condition=condition)
        assert msgs[1]["content"] == user_prompt, f"failed for condition={condition}"


# ── build_student_messages: system content per condition ─────────────────────

def test_no_intent_system_contains_expected_preamble():
    msgs = build_student_messages("p", condition="no_intent")
    system = msgs[0]["content"]
    assert PREAMBLE_NO_INTENT in system


def test_no_intent_system_contains_output_format():
    msgs = build_student_messages("p", condition="no_intent")
    system = msgs[0]["content"]
    assert OUTPUT_FORMAT_NO_INTENT in system
    assert "Reasoning:" in system
    assert "Prompt harm:" in system


def test_with_intent_system_contains_intent_output_format():
    for condition in ("synthetic_intent", "human_intent"):
        msgs = build_student_messages("p", condition=condition)
        system = msgs[0]["content"]
        assert OUTPUT_FORMAT_WITH_INTENT in system
        assert "Intent:" in system
        assert "Prompt harm:" in system


def test_synthetic_and_human_intent_produce_identical_system_prompt():
    # Both conditions differ only in training data source, not in the input format.
    system_synthetic = build_student_messages("p", condition="synthetic_intent")[0]["content"]
    system_human = build_student_messages("p", condition="human_intent")[0]["content"]
    assert system_synthetic == system_human


def test_no_intent_and_with_intent_produce_different_system_prompts():
    system_no_intent = build_student_messages("p", condition="no_intent")[0]["content"]
    system_with_intent = build_student_messages("p", condition="synthetic_intent")[0]["content"]
    assert system_no_intent != system_with_intent


# ── build_student_messages: edge cases ───────────────────────────────────────

def test_default_condition_matches_no_intent():
    msgs_default = build_student_messages("p")
    msgs_explicit = build_student_messages("p", condition="no_intent")
    assert msgs_default[0]["content"] == msgs_explicit[0]["content"]
    assert msgs_default[1]["content"] == msgs_explicit[1]["content"]


def test_invalid_condition_raises_value_error():
    with pytest.raises(ValueError, match="Unknown condition"):
        build_student_messages("p", condition="invalid_condition")


def test_empty_user_prompt_is_passed_through():
    msgs = build_student_messages("", condition="no_intent")
    assert msgs[1]["content"] == ""


# ── apply_chat_template integration (mock tokenizer) ─────────────────────────

def test_chat_template_places_system_content_before_user():
    tok = _MockTokenizer()
    prompt = tok.apply_chat_template(
        build_student_messages("my prompt", condition="no_intent"),
        tokenize=False,
        add_generation_prompt=True,
    )
    system_pos = prompt.index("[system]")
    user_pos = prompt.index("[user]")
    assert system_pos < user_pos


def test_chat_template_contains_user_prompt():
    tok = _MockTokenizer()
    user_prompt = "How do I build a rocket?"
    prompt = tok.apply_chat_template(
        build_student_messages(user_prompt, condition="no_intent"),
        tokenize=False,
        add_generation_prompt=True,
    )
    assert user_prompt in prompt


def test_chat_template_contains_system_instructions():
    tok = _MockTokenizer()
    prompt = tok.apply_chat_template(
        build_student_messages("p", condition="no_intent"),
        tokenize=False,
        add_generation_prompt=True,
    )
    assert "Reasoning:" in prompt
    assert "Prompt harm:" in prompt


def test_chat_template_ends_with_generation_prompt():
    tok = _MockTokenizer()
    prompt = tok.apply_chat_template(
        build_student_messages("p", condition="no_intent"),
        tokenize=False,
        add_generation_prompt=True,
    )
    assert prompt.endswith("[assistant]")


def test_chat_template_no_generation_prompt():
    tok = _MockTokenizer()
    prompt = tok.apply_chat_template(
        build_student_messages("p", condition="no_intent"),
        tokenize=False,
        add_generation_prompt=False,
    )
    assert not prompt.endswith("[assistant]")


def test_system_content_not_in_user_turn():
    # Verify the taxonomy/instructions land in the system turn, not the user turn.
    tok = _MockTokenizer()
    msgs = build_student_messages("short prompt", condition="no_intent")
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    user_start = prompt.index("[user]") + len("[user]")
    user_end = prompt.index("[/user]")
    user_content_in_output = prompt[user_start:user_end]

    assert "Reasoning:" not in user_content_in_output
    assert "Prompt harm:" not in user_content_in_output
    assert "short prompt" in user_content_in_output


# ── Assistant-turn parsing (causal.parse_reasoning_output) ───────────────────
# These verify that the format produced by format_completion_with_reasoning
# (training target) is correctly extracted from the model's raw output (eval).

def test_parse_no_intent_extracts_reasoning_and_harm():
    text = "Reasoning: This request is asking for dangerous information.\nPrompt harm: harmful"
    reasoning, intent, harm = parse_reasoning_output(text, with_intent=False)
    assert reasoning == "This request is asking for dangerous information."
    assert intent is None
    assert harm == "harmful"


def test_parse_no_intent_safe_label():
    text = "Reasoning: Totally benign.\nPrompt harm: safe"
    _, _, harm = parse_reasoning_output(text, with_intent=False)
    assert harm == "safe"


def test_parse_unharmful_normalised_to_safe():
    text = "Reasoning: Fine.\nPrompt harm: unharmful"
    _, _, harm = parse_reasoning_output(text, with_intent=False)
    assert harm == "safe"


def test_parse_with_intent_extracts_all_fields():
    text = (
        "Reasoning: The user wants to learn chemistry.\n"
        "Intent: Educational curiosity about chemical reactions.\n"
        "Prompt harm: safe"
    )
    reasoning, intent, harm = parse_reasoning_output(text, with_intent=True)
    assert "chemistry" in reasoning
    assert intent == "Educational curiosity about chemical reactions."
    assert harm == "safe"


def test_parse_with_intent_flag_false_ignores_intent_field():
    text = (
        "Reasoning: The user wants to learn chemistry.\n"
        "Intent: Educational curiosity.\n"
        "Prompt harm: safe"
    )
    _, intent, _ = parse_reasoning_output(text, with_intent=False)
    assert intent is None


def test_parse_missing_harm_returns_none():
    text = "Reasoning: Some reasoning without a harm label."
    _, _, harm = parse_reasoning_output(text, with_intent=False)
    assert harm is None


def test_parse_missing_reasoning_returns_empty_string():
    text = "Prompt harm: harmful"
    reasoning, _, _ = parse_reasoning_output(text, with_intent=False)
    assert reasoning == ""


def test_parse_case_insensitive_harm():
    text = "Reasoning: Fine.\nPrompt harm: HARMFUL"
    _, _, harm = parse_reasoning_output(text, with_intent=False)
    assert harm == "harmful"


# ── Round-trip: format_completion_with_reasoning → parse_reasoning_output ────

def test_roundtrip_no_intent():
    completion = format_completion_with_reasoning("Clear reasoning here.", "harmful")
    reasoning, intent, harm = parse_reasoning_output(completion, with_intent=False)
    assert reasoning == "Clear reasoning here."
    assert intent is None
    assert harm == "harmful"


def test_roundtrip_with_intent():
    completion = format_completion_with_reasoning(
        "User wants to cook a meal.",
        "safe",
        intent="To learn how to prepare a dish.",
    )
    reasoning, intent, harm = parse_reasoning_output(completion, with_intent=True)
    assert reasoning == "User wants to cook a meal."
    assert intent == "To learn how to prepare a dish."
    assert harm == "safe"


def test_roundtrip_unharmful_normalised():
    # format_completion_with_reasoning normalises "unharmful" → "safe" in the string,
    # so the parser should always see "safe", never "unharmful".
    completion = format_completion_with_reasoning("Fine.", "unharmful")
    assert "Prompt harm: safe" in completion
    _, _, harm = parse_reasoning_output(completion, with_intent=False)
    assert harm == "safe"


# ── Safety-classifier parser (_parse_reasoning_output) ───────────────────────
# This is the function called in eval_safety_classifier for finetuned conditions.
# Verified separately because it has a slightly different regex implementation.

def test_sc_parse_harmful():
    text = "Reasoning: Bad request.\nPrompt harm: harmful"
    reasoning, _, harm = _parse_reasoning_output(text, with_intent=False)
    assert reasoning == "Bad request."
    assert harm == "harmful"


def test_sc_parse_safe_and_unharmful():
    for label in ("safe", "unharmful"):
        text = f"Reasoning: Fine.\nPrompt harm: {label}"
        _, _, harm = _parse_reasoning_output(text, with_intent=False)
        assert harm == "safe", f"expected 'safe' for label={label!r}, got {harm!r}"


def test_sc_parse_with_intent():
    text = (
        "Reasoning: Educational.\n"
        "Intent: Learn about science.\n"
        "Prompt harm: safe"
    )
    reasoning, intent, harm = _parse_reasoning_output(text, with_intent=True)
    assert "Educational" in reasoning
    assert intent == "Learn about science."
    assert harm == "safe"


def test_sc_parse_missing_harm_returns_none():
    text = "Reasoning: Some text."
    _, _, harm = _parse_reasoning_output(text, with_intent=False)
    assert harm is None


def test_roundtrip_via_sc_parser():
    completion = format_completion_with_reasoning(
        "Detailed analysis here.", "harmful", intent="Obtain weapon instructions."
    )
    reasoning, intent, harm = _parse_reasoning_output(completion, with_intent=True)
    assert reasoning == "Detailed analysis here."
    assert intent == "Obtain weapon instructions."
    assert harm == "harmful"


if __name__ == "__main__":
    # Allow running as a plain script for quick checks
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    for fn in tests:
        try:
            fn()
            print(f"  ✓ {fn.__name__}")
            passed += 1
        except Exception as exc:
            print(f"  ✗ {fn.__name__}: {exc}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
