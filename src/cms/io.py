"""Small I/O helpers shared by the post-collection pipeline."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as ex:
                raise ValueError(f"{path}:{line_no}: invalid JSON") from ex
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    tmp.replace(p)


def sha256_json(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def row_uid(row: dict[str, Any]) -> str:
    """Stable trajectory key. Inspect sample ids can repeat across epochs; UUIDs should not."""
    if row.get("uid"):
        return str(row["uid"])
    return sha256_json(
        {
            "id": row.get("id"),
            "epoch": row.get("epoch"),
            "log": row.get("log"),
            "agent": row.get("agent"),
        }
    )[:24]


def compact_actions(actions: list[dict[str, Any]], max_chars: int = 120_000) -> str:
    """Deterministic text view of visible actions, retaining both the head and tail if capped."""
    chunks: list[str] = []
    for i, a in enumerate(actions):
        role = a.get("role", "?")
        if role == "assistant":
            chunks.append(f"[{i}] assistant text:\n{a.get('text', '')}")
            for tc in a.get("tool_calls") or []:
                chunks.append(
                    f"[{i}] tool_call {tc.get('function')}: "
                    + json.dumps(
                        tc.get("arguments"), ensure_ascii=False, sort_keys=True
                    )
                )
        elif role == "tool":
            chunks.append(
                f"[{i}] tool_result {a.get('function')}:\n{a.get('text', '')}"
            )
        else:
            chunks.append(f"[{i}] {role}:\n{a.get('text', '')}")
    text = "\n\n".join(chunks)
    if len(text) <= max_chars:
        return text
    half = max(1, (max_chars - 80) // 2)
    return (
        text[:half]
        + "\n\n...[ACTION TRACE TRUNCATED IN THE MIDDLE]...\n\n"
        + text[-half:]
    )
