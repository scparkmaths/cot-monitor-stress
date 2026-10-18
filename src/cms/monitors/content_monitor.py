"""Content monitors over stated task + actions + CoT."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from cms.cache import cached_generate
from cms.io import read_jsonl
from cms.monitors.common import load_prompt, monitor_input, suspicion_from_text

load_dotenv()

ROLE_MAP = {
    "mid": ("monitor_mid", "content_mid"),
}


async def one(
    row, role: str, template: str, a, sem: asyncio.Semaphore
) -> tuple[float, int]:
    body = monitor_input(row, include_cot=True, max_action_chars=a.max_action_chars)
    prompt = template.format(input=body)
    async with sem:
        text, hit = await cached_generate(
            role, prompt, generate_kwargs={"temperature": 0, "max_tokens": 80}
        )
    return suspicion_from_text(text), int(hit)


async def run(a) -> None:
    rows = read_jsonl(a.input)
    template = load_prompt(a.prompt)
    sem = asyncio.Semaphore(a.concurrency)
    selected = [ROLE_MAP[name] for name in a.models]

    jobs = []
    meta = []
    for row in rows:
        for role, name in selected:
            jobs.append(one(row, role, template, a, sem))
            meta.append((row, name))
    results = await asyncio.gather(*jobs)

    out = []
    hits = 0
    for (row, name), (score, nhit) in zip(meta, results):
        hits += nhit
        out.append(
            {
                "uid": row.get("uid"),
                "id": row.get("id"),
                "agent": row.get("agent"),
                "mode": row.get("mode"),
                "main_task": row.get("main_task"),
                "side_task": row.get("side_task"),
                "condition": row.get("condition"),
                "scope": row.get("scope"),
                "monitor": name,
                "cv": "fixed",
                "fold_main": int(row.get("fold_main")),
                "fold_side": row.get("fold_side"),
                "score": float(score),
            }
        )

    new = pd.DataFrame(out)
    output = Path(a.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    if not new.empty:
        sort_cols = [
            c
            for c in ("monitor", "agent", "fold_main", "uid", "scope", "condition")
            if c in new.columns
        ]
        if sort_cols:
            new = new.sort_values(sort_cols, kind="stable").reset_index(drop=True)
    new.to_parquet(output, index=False)
    print(
        f"wrote {len(out)} newly scored rows ({', '.join(a.models)}) -> {a.output}; "
        f"paid-call cache hits={hits}"
    )
    print(f"content score file now contains {len(new)} rows")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/raw/conditions.jsonl")
    ap.add_argument("--output", default="data/raw/content_scores.parquet")
    ap.add_argument("--prompt", default="config/prompts/content_monitor.txt")
    ap.add_argument("--max-action-chars", type=int, default=120_000)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument(
        "--models",
        nargs="+",
        choices=sorted(ROLE_MAP),
        default=["mid"],
        help="score the evaluated Qwen content monitor",
    )
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
