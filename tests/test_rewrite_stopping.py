import asyncio
from types import SimpleNamespace

from cms.attack import repair, retry_empty_aware
from cms.io import read_jsonl, sha256_json, sha256_text, write_jsonl


def test_one_repair_then_one_empty_retry_and_stop(tmp_path, monkeypatch):
    original = "Inspect the infrastructure configuration carefully. " * 10
    row = {
        "uid": "example",
        "keep": True,
        "cot": original,
        "actions": [],
        "mode": "honest",
    }
    candidate = {
        "uid": "example",
        "style": "aware",
        "rewrite": "",
        "actions_sha256": sha256_json([]),
        "original_cot_sha256": sha256_text(original),
    }
    source = tmp_path / "trajectories.jsonl"
    candidates = tmp_path / "candidates.jsonl"
    write_jsonl(source, [row])
    write_jsonl(candidates, [candidate])
    calls = []

    async def empty_generation(*args, **kwargs):
        calls.append(kwargs)
        return "", False

    monkeypatch.setattr(repair, "cached_generate", empty_generation)
    monkeypatch.setattr(retry_empty_aware, "cached_generate", empty_generation)
    args = SimpleNamespace(
        input=source,
        candidates=candidates,
        output=candidates,
        prompt="config/prompts/obfuscate_aware_repair.txt",
        min_edit_distance=0.30,
        max_action_chars=24000,
        temperature=0.3,
        max_tokens=16000,
        concurrency=1,
        limit=None,
    )
    asyncio.run(repair.run(args))
    asyncio.run(retry_empty_aware.run(args))
    asyncio.run(repair.run(args))
    asyncio.run(retry_empty_aware.run(args))
    assert len(calls) == 2
    assert calls[1]["generate_kwargs"]["reasoning_effort"] == "low"
    final = read_jsonl(candidates)[0]
    assert final["repair_attempt"] == final["empty_retry_attempt"] == 1
