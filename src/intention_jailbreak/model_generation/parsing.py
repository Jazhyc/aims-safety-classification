"""Shared parsing utilities for intent-generation style outputs."""

from __future__ import annotations

import re
from typing import Optional


def normalize_harm_label_binary(label: str | None, safe_label: str = "safe") -> str | None:
    """Normalize a raw harm label to binary values.

    Returns:
        - "harmful"
        - safe_label (default: "safe")
        - None (if unparseable)
    """
    if not label:
        return None

    value = str(label).strip().lower().rstrip(".").rstrip(";")

    if "unharmful" in value or "safe" in value or value in {"harmless", "benign", "s", "b"}:
        return safe_label
    if "harmful" in value or value in {"unsafe", "dangerous", "h", "u"}:
        return "harmful"
    return None


def extract_intent_and_harm(raw_text: str, safe_label: str = "safe") -> tuple[Optional[str], Optional[str]]:
    """Parse completion text into (intent, harm).

    Handles common formats:
      - "Intent: <text>; Harm: harmful"
      - "Intent: <text>\\nHarm: harmful"
      - "Intent: <text> Harm: harmful"
      - "<text>; Harm: harmful"  (missing Intent prefix)
      - "<text>"                 (intent only)

    Notes:
      - Mirrors the distillation fallback parser style by preferring the LAST
        regex match when multiple template-like blocks appear in the output.
      - Strips `<think>...</think>` blocks when present.
    """
    text = (raw_text or "").strip()
    if not text:
        return None, None

    if "<think>" in text and "</think>" in text:
        text = text.split("</think>")[-1].strip()

    # Pattern 1: "Intent: <text>; Harm: <label>"
    matches = list(
        re.finditer(r"Intent:\s*(.+?);\s*Harm:\s*(\S+)", text, re.IGNORECASE | re.DOTALL)
    )
    if matches:
        m = matches[-1]
        return m.group(1).strip(), normalize_harm_label_binary(m.group(2), safe_label=safe_label)

    # Pattern 2: "Intent: <text>\nHarm: <label>"
    matches = list(
        re.finditer(r"Intent:\s*(.+?)\n\s*Harm:\s*(\S+)", text, re.IGNORECASE | re.DOTALL)
    )
    if matches:
        m = matches[-1]
        return m.group(1).strip().rstrip("."), normalize_harm_label_binary(m.group(2), safe_label=safe_label)

    # Pattern 3: "Intent: <text> Harm: <label>"
    matches = list(
        re.finditer(r"Intent:\s*(.+?)\s+Harm:\s*(\S+)", text, re.IGNORECASE | re.DOTALL)
    )
    if matches:
        m = matches[-1]
        return m.group(1).strip().rstrip("."), normalize_harm_label_binary(m.group(2), safe_label=safe_label)

    # Pattern 4: "<text>; Harm: <label>"
    matches = list(
        re.finditer(r"^(.+?);\s*Harm:\s*(\S+)", text, re.IGNORECASE | re.DOTALL)
    )
    if matches:
        m = matches[-1]
        return m.group(1).strip(), normalize_harm_label_binary(m.group(2), safe_label=safe_label)

    # Pattern 5: Harm label after semicolon/newline anywhere
    matches = list(re.finditer(r"[;\n]\s*Harm:\s*(\S+)", text, re.IGNORECASE))
    if matches:
        m = matches[-1]
        harm = normalize_harm_label_binary(m.group(1), safe_label=safe_label)
        intent = text[:m.start()].strip()
        intent = re.sub(r"^Intent:\s*", "", intent, flags=re.IGNORECASE).strip()
        return (intent if intent else text), harm

    # Pattern 6: Intent only
    matches = list(re.finditer(r"^Intent:\s*(.+?)(?:;|$)", text, re.IGNORECASE | re.DOTALL))
    if matches:
        return matches[-1].group(1).strip(), None

    return text, None
