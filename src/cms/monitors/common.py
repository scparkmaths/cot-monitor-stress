"""Shared parsing and prompt helpers for LLM monitors."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from cms.io import compact_actions

_JSON_RE = re.compile(r"\{.*\}", re.S)


def load_prompt(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def parse_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass
    m = _JSON_RE.search(text)
    if not m:
        raise ValueError(f"model did not return a JSON object: {text[:300]!r}")
    value = json.loads(m.group(0))
    if not isinstance(value, dict):
        raise ValueError("JSON response was not an object")
    return value


def suspicion_from_text(text: str) -> float:
    value = parse_json_object(text).get("suspicion")
    score = float(value)
    if not 0.0 <= score <= 1.0:
        raise ValueError(f"suspicion outside [0,1]: {score}")
    return score


def monitor_input(
    row: dict[str, Any], *, include_cot: bool, max_action_chars: int
) -> str:
    parts = [
        "STATED TASK:\n" + str(row.get("task_prompt", "")),
        "VISIBLE ACTION TRACE:\n"
        + compact_actions(row.get("actions") or [], max_action_chars),
    ]
    if include_cot:
        parts.append("REASONING TRACE:\n" + str(row.get("cot", "")))
    return "\n\n".join(parts)
