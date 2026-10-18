"""Generate naive and structure-aware CoT rewrites.

Each kept trajectory gets two candidates, followed by one mechanical aware
repair and one retry for empty aware output. Independent validation follows
in `cms.attack.validate`; rejection never triggers additional generation.
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from types import SimpleNamespace

from dotenv import load_dotenv

from cms.cache import cached_generate
from cms.io import (
    compact_actions,
    read_jsonl,
    row_uid,
    sha256_json,
    sha256_text,
    write_jsonl,
)

load_dotenv()


async def rewrite_one(row, style: str, template: str, a, sem: asyncio.Semaphore):
    action_summary = compact_actions(row.get("actions") or [], a.max_action_chars)
    prompt = template.format(
        task=str(row.get("task_prompt", "")),
        actions=action_summary,
        cot=str(row.get("cot", "")),
    )
    async with sem:
        completion, hit = await cached_generate(
            "obfuscator",
            prompt,
            generate_kwargs={"temperature": a.temperature, "max_tokens": a.max_tokens},
            cache_extra={"rewrite_style": style, "salt": a.salt},
        )
    original = str(row.get("cot", ""))
    return {
        "uid": row_uid(row),
        "id": row.get("id"),
        "epoch": row.get("epoch"),
        "agent": row.get("agent"),
        "mode": row.get("mode"),
        "main_task": row.get("main_task"),
        "side_task": row.get("side_task"),
        "fold_main": row.get("fold_main"),
        "fold_side": row.get("fold_side"),
        "style": style,
        "original_cot_sha256": sha256_text(original),
        "actions_sha256": sha256_json(row.get("actions") or []),
        "rewrite": completion.strip(),
        "original_n_chars": len(original),
        "rewrite_n_chars": len(completion.strip()),
        "cache_hit": hit,
        "salt": a.salt,
    }


async def run(a) -> None:
    rows = [r for r in read_jsonl(a.input) if r.get("keep")]
    prompts = {
        "naive": Path(a.naive_prompt).read_text(encoding="utf-8"),
        "aware": Path(a.aware_prompt).read_text(encoding="utf-8"),
    }
    sem = asyncio.Semaphore(a.concurrency)
    jobs = [
        rewrite_one(r, style, prompts[style], a, sem)
        for r in rows
        for style in ("naive", "aware")
    ]
    out = await asyncio.gather(*jobs)
    write_jsonl(a.output, out)
    from cms.attack import repair, retry_empty_aware

    # Preserve the generation parameters and cache keys of each fixed stage.
    helper = SimpleNamespace(
        input=a.input,
        candidates=a.output,
        output=a.output,
        prompt=a.repair_prompt,
        min_edit_distance=0.30,
        max_action_chars=a.max_action_chars,
        temperature=0.3,
        max_tokens=a.max_tokens,
        concurrency=a.concurrency,
        limit=None,
    )
    await repair.run(helper)
    await retry_empty_aware.run(helper)
    hits = sum(bool(r["cache_hit"]) for r in out)
    print(
        f"wrote {len(out)} rewrite candidates -> {a.output}; cache hits {hits}/{len(out)}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/raw/trajectories.jsonl")
    ap.add_argument("--output", default="data/raw/rewrite_candidates.jsonl")
    ap.add_argument("--naive-prompt", default="config/prompts/obfuscate_naive.txt")
    ap.add_argument("--aware-prompt", default="config/prompts/obfuscate_aware.txt")
    ap.add_argument(
        "--repair-prompt", default="config/prompts/obfuscate_aware_repair.txt"
    )
    ap.add_argument("--max-action-chars", type=int, default=24_000)
    ap.add_argument("--temperature", type=float, default=0.4)
    ap.add_argument("--max-tokens", type=int, default=16_000)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument(
        "--salt", default="v1", help="change to force a fresh sampled rewrite"
    )
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
