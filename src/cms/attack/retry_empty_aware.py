"""Retry only aware rewrites whose previous generation returned no final text.

This is a generation-failure retry, not adaptive resampling of non-empty
rewrites. Uses the same GLM obfuscator with low reasoning effort so the model
has ample output budget left for the final rewrite.
"""

from __future__ import annotations

import argparse
import asyncio
import math
from pathlib import Path

from dotenv import load_dotenv

from cms.cache import cached_generate
from cms.io import compact_actions, read_jsonl, row_uid, sha256_text, write_jsonl

load_dotenv()


async def retry_one(
    row: dict,
    candidate: dict,
    template: str,
    args,
    sem: asyncio.Semaphore,
) -> dict:
    original = str(row.get("cot", ""))
    target = len(original)

    low = math.ceil(0.90 * target)
    high = math.floor(1.10 * target)

    actions = compact_actions(
        row.get("actions") or [],
        args.max_action_chars,
    )

    prompt = template.format(
        task=str(row.get("task_prompt", "")),
        actions=actions,
        cot=original,
        current="[EMPTY PREVIOUS OUTPUT — produce the rewrite from the original reasoning]",
        original_n_chars=target,
        min_chars=low,
        max_chars=high,
    )

    async with sem:
        completion, hit = await cached_generate(
            "obfuscator",
            prompt,
            generate_kwargs={
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
                "reasoning_effort": "low",
            },
            cache_extra={
                "rewrite_style": "aware",
                "repair": "empty-output-low-reasoning-v1",
            },
        )

    rewrite = completion.strip()

    out = dict(candidate)
    out.update(
        {
            "rewrite": rewrite,
            "rewrite_n_chars": len(rewrite),
            "empty_retry_attempt": 1,
            "empty_retry_reasoning_effort": "low",
            "empty_retry_cache_hit": hit,
            "empty_retry_tag": "empty-output-low-reasoning-v1",
            "parent_rewrite_sha256": sha256_text(str(candidate.get("rewrite", ""))),
        }
    )

    return out


async def run(args) -> None:
    trajectories = [r for r in read_jsonl(args.input) if r.get("keep")]

    by_uid = {row_uid(r): r for r in trajectories}

    candidates = read_jsonl(args.candidates)

    selected = [
        i
        for i, c in enumerate(candidates)
        if (
            c.get("style") == "aware"
            and not str(c.get("rewrite", "")).strip()
            and c.get("empty_retry_attempt", 0) < 1
        )
    ]

    print(f"empty aware candidates found: {len(selected)}")

    if args.limit is not None:
        selected = selected[: args.limit]
        print(f"retrying this run: {len(selected)}")

    template = Path(args.prompt).read_text(encoding="utf-8")

    sem = asyncio.Semaphore(args.concurrency)

    jobs = []

    for i in selected:
        candidate = candidates[i]
        uid = str(candidate["uid"])

        if uid not in by_uid:
            raise SystemExit(f"candidate uid missing from trajectories: {uid}")

        jobs.append(
            retry_one(
                by_uid[uid],
                candidate,
                template,
                args,
                sem,
            )
        )

    replacements = await asyncio.gather(*jobs)

    out = list(candidates)

    for i, replacement in zip(
        selected,
        replacements,
    ):
        out[i] = replacement

    write_jsonl(args.output, out)

    nonempty = sum(bool(str(r.get("rewrite", "")).strip()) for r in replacements)

    hits = sum(bool(r.get("empty_retry_cache_hit")) for r in replacements)

    print(f"non-empty retry outputs: " f"{nonempty}/{len(replacements)}")

    print(f"retry cache hits: " f"{hits}/{len(replacements)}")

    print(f"wrote {len(out)} candidates -> {args.output}")


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--input",
        default="data/raw/trajectories.jsonl",
    )

    ap.add_argument(
        "--candidates",
        default="data/raw/rewrite_candidates.jsonl",
    )

    ap.add_argument(
        "--output",
        default="data/raw/rewrite_candidates.jsonl",
    )

    ap.add_argument(
        "--prompt",
        default="config/prompts/obfuscate_aware_repair.txt",
    )

    ap.add_argument(
        "--max-action-chars",
        type=int,
        default=24_000,
    )

    ap.add_argument(
        "--temperature",
        type=float,
        default=0.3,
    )

    ap.add_argument(
        "--max-tokens",
        type=int,
        default=16_000,
    )

    ap.add_argument(
        "--concurrency",
        type=int,
        default=4,
    )

    ap.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
