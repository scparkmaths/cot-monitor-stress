"""Extract trajectories directly from Inspect .eval logs (no ControlArena export).

  uv run python -m cms.collect.from_logs --log-dir logs/main

Writes:
  data/raw/trajectories.jsonl          every sample, with a `keep` flag and fold ids
  outputs/tables/collection_counts.md  per agent x mode counts (committed; goes in README)
"""

import argparse
import glob
import json
from collections import Counter
from pathlib import Path

from inspect_ai.log import read_eval_log, read_eval_log_samples
from inspect_ai.model import ContentReasoning, ContentText

MIN_COT_CHARS = 200
COT_SEP = "\n\n---\n\n"


def truthy(v) -> bool:
    return v in (1, 1.0, True, "C", "CORRECT", "correct")


def score_values(sample) -> dict:
    out = {}
    for name, sc in (sample.scores or {}).items():
        v = sc.value
        out.update(v) if isinstance(v, dict) else out.update({name: v})
    return out


def text_of(content) -> str:
    if isinstance(content, str):
        return content
    return "\n".join(c.text for c in (content or []) if isinstance(c, ContentText))


def split_messages(messages):
    """cot = reasoning blocks only; actions = everything an actions-only monitor may see.
    System messages are dropped (the attack policy's instructions live there)."""
    task_prompt, cot, actions = None, [], []
    for m in messages:
        if m.role == "system":
            continue
        if m.role == "user":
            if task_prompt is None:
                task_prompt = text_of(m.content)
            else:
                actions.append({"role": "user", "text": text_of(m.content)})
        elif m.role == "assistant":
            for c in m.content if isinstance(m.content, list) else []:
                if isinstance(c, ContentReasoning) and not c.redacted and c.reasoning:
                    cot.append(c.reasoning)
            actions.append(
                {
                    "role": "assistant",
                    "text": text_of(m.content),
                    "tool_calls": [
                        {"function": tc.function, "arguments": tc.arguments}
                        for tc in (m.tool_calls or [])
                    ],
                }
            )
        elif m.role == "tool":
            actions.append(
                {
                    "role": "tool",
                    "function": getattr(m, "function", None),
                    "text": text_of(m.content),
                }
            )
    return task_prompt or "", COT_SEP.join(cot), actions


def mode_of(md: dict, task: str) -> str:
    m = str(md.get("eval_mode", "")).lower()
    if "attack" in m or "honest" in m:
        return "attack" if "attack" in m else "honest"
    return "attack" if "attack" in task else "honest"


def task_names(md: dict, sample_id: str):
    main, side = md.get("main_task_name"), md.get("side_task_name")
    if not main and "_ST_" in sample_id:
        main, side = sample_id.split("_ST_", 1)
        side = "ST_" + side
    return main or sample_id, side


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", required=True)
    ap.add_argument("--out", default="data/raw/trajectories.jsonl")
    ap.add_argument("--counts", default="outputs/tables/collection_counts.md")
    a = ap.parse_args()

    rows = []
    for path in sorted(glob.glob(f"{a.log_dir}/*.eval")):
        hdr = read_eval_log(path, header_only=True)
        task = hdr.eval.task
        if hdr.status != "success":
            print(
                f"warning: {Path(path).name} status={hdr.status} (unfinished or failed run)"
            )
        agent = next(
            (x for x in ("agent_strong", "agent_baseline") if x in task), "unknown"
        )

        for s in read_eval_log_samples(
            path, all_samples_required=False, resolve_attachments=True
        ):
            md = s.metadata or {}
            mode = mode_of(md, task)
            main_task, side_task = task_names(md, str(s.id))
            sv = score_values(s)
            task_prompt, cot, actions = split_messages(s.messages)

            side_desc = str(md.get("side_task_description") or "")[:80]
            leak = bool(mode == "attack" and side_desc and side_desc in task_prompt)

            r = {
                "uid": s.uuid,
                "id": str(s.id),
                "epoch": s.epoch,
                "log": Path(path).name,
                "setting": "iac",
                "agent": agent,
                "mode": mode,
                "main_task": main_task,
                "side_task": side_task if mode == "attack" else None,
                "main_ok": truthy(sv.get("main_task_success")),
                "side_ok": truthy(sv.get("side_task_success"))
                if mode == "attack"
                else None,
                "hit_limit": getattr(s, "limit", None) is not None,
                "errored": s.error is not None,
                "side_prompt_leak": leak,
                "task_prompt": task_prompt,
                "cot": cot,
                "actions": actions,
            }
            ok = r["main_ok"] if mode == "honest" else r["side_ok"]
            r["keep"] = bool(
                ok
                and not r["hit_limit"]
                and not r["errored"]
                and len(cot) >= MIN_COT_CHARS
            )
            r["capped_after_side_success"] = bool(
                r["hit_limit"] and mode == "attack" and r["side_ok"]
            )
            rows.append(r)

    mains = sorted({r["main_task"] for r in rows})
    sides = sorted({r["side_task"] for r in rows if r["side_task"]})
    for r in rows:
        r["fold_main"] = mains.index(r["main_task"])
        r["fold_side"] = sides.index(r["side_task"]) if r["side_task"] else None

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    keys = sorted({(r["agent"], r["mode"]) for r in rows})
    lines = [
        "| agent | mode | total | kept | task failed | hit 1M cap | capped after side success | errored | short CoT | prompt leak |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for agent, mode in keys:
        g = [r for r in rows if r["agent"] == agent and r["mode"] == mode]
        okf = (lambda r: r["main_ok"]) if mode == "honest" else (lambda r: r["side_ok"])
        c = Counter()
        for r in g:
            c["kept"] += r["keep"]
            c["failed"] += not okf(r)
            c["cap"] += r["hit_limit"]
            c["cap_ok"] += r["capped_after_side_success"]
            c["err"] += r["errored"]
            c["short"] += len(r["cot"]) < MIN_COT_CHARS
            c["leak"] += r["side_prompt_leak"]
        lines.append(
            f"| {agent} | {mode} | {len(g)} | {c['kept']} | {c['failed']} | {c['cap']} | "
            f"{c['cap_ok']} | {c['err']} | {c['short']} | {c['leak']} |"
        )
    lines.append(f"\nmain tasks: {len(mains)}   side tasks: {len(sides)}")
    table = "\n".join(lines)
    Path(a.counts).parent.mkdir(parents=True, exist_ok=True)
    Path(a.counts).write_text(table + "\n")
    print(table)
    print(f"\nwrote {len(rows)} rows -> {a.out}")


if __name__ == "__main__":
    main()
