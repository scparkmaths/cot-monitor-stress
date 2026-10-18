"""Disk cache for every paid model call.

Cache key = sha256(model slug, pinned provider, prompt, generation/model params).
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import yaml
from inspect_ai.model import GenerateConfig, get_model

from cms.io import sha256_json

_LOCKS: dict[str, asyncio.Lock] = {}


def load_models(path: str | Path = "config/models.yaml") -> dict[str, dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)["openrouter"]


def _cache_path(root: str | Path, key: str) -> Path:
    root = Path(root)
    return root / key[:2] / f"{key}.json"


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False, sort_keys=True)
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


async def cached_generate(
    role: str,
    prompt: str,
    *,
    generate_kwargs: dict[str, Any] | None = None,
    cache_extra: dict[str, Any] | None = None,
    cache_root: str | Path = "cache/llm",
    models_path: str | Path = "config/models.yaml",
    cache_prompt: bool | None = None,
) -> tuple[str, bool]:
    """Generate once, then replay from JSON forever unless the key changes.

    `cache_extra` affects only the cache key. It is useful for independent judge
    replicates without passing unsupported parameters to the provider.
    """
    generate_kwargs = dict(generate_kwargs or {})
    cache_extra = dict(cache_extra or {})
    entry = load_models(models_path)[role]
    key_payload = {
        "slug": entry["slug"],
        "provider": entry["provider"],
        "prompt": prompt,
        "params": {
            "generate": generate_kwargs,
            "cache_extra": cache_extra,
            "cache_prompt": cache_prompt,
        },
    }
    key = sha256_json(key_payload)
    path = _cache_path(cache_root, key)
    if path.exists():
        value = json.loads(path.read_text(encoding="utf-8"))
        return str(value["completion"]), True

    lock = _LOCKS.setdefault(key, asyncio.Lock())
    async with lock:
        if path.exists():
            value = json.loads(path.read_text(encoding="utf-8"))
            return str(value["completion"]), True
        model_args: dict[str, Any] = {
            "provider": {"order": [entry["provider"]], "allow_fallbacks": False},
        }
        model = get_model(f"openrouter/{entry['slug']}", **model_args)
        config_kwargs = dict(generate_kwargs)
        if cache_prompt is not None:
            config_kwargs["cache_prompt"] = cache_prompt
        result = await model.generate(prompt, config=GenerateConfig(**config_kwargs))
        completion = result.completion
        _atomic_json(
            path,
            {
                "key": key,
                "slug": entry["slug"],
                "provider": entry["provider"],
                "params": key_payload["params"],
                "completion": completion,
            },
        )
        return completion, False
