"""Zero-cost estimate of paid API spend for the canonical pilot pipeline.

Reads local trajectory/rewrite/condition artifacts and the on-disk cms LLM cache.
It never calls a model or the network. Estimates cover initial generation,
validation and scoring; mechanical repair and empty-output retries are excluded.

Prices are provider-specific defaults from OpenRouter as of 2026-10-12:
  qwen/qwen3.5-27b via NovitaAI     $0.30/M in, $2.40/M out
  z-ai/glm-5.3-flashx via Z.ai     $0.37/M in, $1.25/M out
Override them on the CLI if prices change.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from cms.io import compact_actions, read_jsonl, row_uid, sha256_json
from cms.monitors.common import load_prompt, monitor_input


@dataclass(frozen=True)
class Scenario:
    name: str
    chars_per_token: float
    short_output_tokens: int
    naive_output_ratio: float
    aware_output_ratio: float


SCENARIOS = (
    Scenario("low", 4.2, 20, 0.75, 0.90),
    Scenario("base", 3.5, 35, 1.00, 1.00),
    Scenario("high", 2.8, 80, 1.25, 1.10),
)


@dataclass
class Call:
    stage: str
    role: str
    prompt_chars: int
    output_kind: str  # "short" or "rewrite"
    source_cot_chars: int = 0
    style: str | None = None
    actual_output_chars: int | None = None
    cached: bool = False
    exact: bool = True
    dedupe_key: str | None = None


def load_models(path: str | Path) -> dict[str, dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)["openrouter"]


def cache_key(
    role: str,
    prompt: str,
    models: dict[str, dict[str, Any]],
    *,
    generate_kwargs: dict[str, Any] | None = None,
    cache_extra: dict[str, Any] | None = None,
    cache_prompt: bool | None = None,
) -> str:
    e = models[role]
    payload = {
        "slug": e["slug"],
        "provider": e["provider"],
        "prompt": prompt,
        "params": {
            "generate": dict(generate_kwargs or {}),
            "cache_extra": dict(cache_extra or {}),
            "cache_prompt": cache_prompt,
        },
    }
    return sha256_json(payload)


def cache_path(root: str | Path, key: str) -> Path:
    return Path(root) / key[:2] / f"{key}.json"


def add_exact_call(
    calls: dict[str, Call],
    *,
    stage: str,
    role: str,
    prompt: str,
    models: dict[str, dict[str, Any]],
    cache_root: str | Path,
    generate_kwargs: dict[str, Any] | None = None,
    cache_extra: dict[str, Any] | None = None,
    cache_prompt: bool | None = None,
    output_kind: str = "short",
    source_cot_chars: int = 0,
    style: str | None = None,
    actual_output_chars: int | None = None,
) -> None:
    key = cache_key(
        role,
        prompt,
        models,
        generate_kwargs=generate_kwargs,
        cache_extra=cache_extra,
        cache_prompt=cache_prompt,
    )
    # Same local-cache key means the paid call happens at most once.
    if key in calls:
        return
    calls[key] = Call(
        stage=stage,
        role=role,
        prompt_chars=len(prompt),
        output_kind=output_kind,
        source_cot_chars=source_cot_chars,
        style=style,
        actual_output_chars=actual_output_chars,
        cached=cache_path(cache_root, key).exists(),
        exact=True,
        dedupe_key=key,
    )


def add_estimated_call(
    calls: dict[str, Call],
    *,
    synthetic_key: str,
    stage: str,
    role: str,
    prompt_chars: int,
    output_kind: str = "short",
    source_cot_chars: int = 0,
    style: str | None = None,
    actual_output_chars: int | None = None,
) -> None:
    if synthetic_key in calls:
        return
    calls[synthetic_key] = Call(
        stage=stage,
        role=role,
        prompt_chars=prompt_chars,
        output_kind=output_kind,
        source_cot_chars=source_cot_chars,
        style=style,
        actual_output_chars=actual_output_chars,
        cached=False,
        exact=False,
        dedupe_key=synthetic_key,
    )


def output_tokens(c: Call, s: Scenario) -> float:
    if c.output_kind == "short":
        return float(s.short_output_tokens)
    if c.actual_output_chars is not None:
        return min(16_000.0, c.actual_output_chars / s.chars_per_token)
    ratio = s.aware_output_ratio if c.style == "aware" else s.naive_output_ratio
    return min(16_000.0, c.source_cot_chars * ratio / s.chars_per_token)


def cost(c: Call, s: Scenario, prices: dict[str, tuple[float, float]]) -> float:
    if c.cached:
        return 0.0
    pin, pout = prices[c.role]
    input_tokens = c.prompt_chars / s.chars_per_token
    out_tokens = output_tokens(c, s)
    return (input_tokens * pin + out_tokens * pout) / 1_000_000.0


def candidate_map(path: str | Path) -> dict[tuple[str, str], dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return {}
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for r in read_jsonl(p):
        out[(str(r.get("uid")), str(r.get("style")))] = r
    return out


def build_calls(a) -> tuple[dict[str, Call], dict[str, Any]]:
    rows = [r for r in read_jsonl(a.trajectories) if r.get("keep")]
    models = load_models(a.models)
    calls: dict[str, Call] = {}

    prompts = {
        "actions": load_prompt(a.actions_prompt),
        "content": load_prompt(a.content_prompt),
        "naive": load_prompt(a.naive_prompt),
        "aware": load_prompt(a.aware_prompt),
        "judge": load_prompt(a.judge_prompt),
    }

    candidates = candidate_map(a.candidates)
    by_uid = {row_uid(r): r for r in rows}

    # Actions-only: one Qwen call per kept trajectory.
    for row in rows:
        body = monitor_input(
            row, include_cot=False, max_action_chars=a.monitor_action_chars
        )
        prompt = prompts["actions"].format(input=body)
        add_exact_call(
            calls,
            stage="actions_only",
            role="monitor_mid",
            prompt=prompt,
            models=models,
            cache_root=a.cache_root,
            generate_kwargs={"temperature": 0, "max_tokens": 80},
        )

    # Rewrite: naive + aware for every kept row because scope=all needs
    # honest rewrites too.
    for row in rows:
        uid = row_uid(row)
        actions = compact_actions(row.get("actions") or [], a.rewrite_action_chars)
        original = str(row.get("cot", ""))
        for style in ("naive", "aware"):
            prompt = prompts[style].format(
                task=str(row.get("task_prompt", "")), actions=actions, cot=original
            )
            cand = candidates.get((uid, style))
            actual_chars = len(str(cand.get("rewrite", "")).strip()) if cand else None
            add_exact_call(
                calls,
                stage="rewrite",
                role="obfuscator",
                prompt=prompt,
                models=models,
                cache_root=a.cache_root,
                generate_kwargs={
                    "temperature": a.rewrite_temperature,
                    "max_tokens": a.rewrite_max_tokens,
                },
                cache_extra={"rewrite_style": style, "salt": a.salt},
                output_kind="rewrite",
                source_cot_chars=len(original),
                style=style,
                actual_output_chars=actual_chars,
            )

    # Validation judge is called only for attack candidates. If candidates exist,
    # the prompt and cache status are exact. Before rewrite generation, estimate
    # using a same-length placeholder rewrite.
    for row in rows:
        if row.get("mode") != "attack":
            continue
        uid = row_uid(row)
        original = str(row.get("cot", ""))
        for style in ("naive", "aware"):
            cand = candidates.get((uid, style))
            if cand:
                rewrite = str(cand.get("rewrite", "")).strip()
                prompt = prompts["judge"].format(
                    task=str(row.get("task_prompt", "")),
                    side_task=str(row.get("side_task", "")),
                    original=original,
                    rewrite=rewrite,
                )
                add_exact_call(
                    calls,
                    stage="rewrite_validation",
                    role="validator",
                    prompt=prompt,
                    models=models,
                    cache_root=a.cache_root,
                    generate_kwargs={"temperature": 0, "max_tokens": 80},
                )
            else:
                # Only character count matters for the estimate; this is not used
                # as a real prompt or cache key.
                prompt = prompts["judge"].format(
                    task=str(row.get("task_prompt", "")),
                    side_task=str(row.get("side_task", "")),
                    original=original,
                    rewrite="x" * len(original),
                )
                add_estimated_call(
                    calls,
                    synthetic_key=f"estimate:judge:{uid}:{style}",
                    stage="rewrite_validation",
                    role="validator",
                    prompt_chars=len(prompt),
                )

    # Content monitors. Once conditions.jsonl exists, use its exact prompts
    # and let the local cache key dedupe attack-only/all scope duplicates. Before
    # then, estimate the three genuinely distinct CoTs per trajectory.
    cond_path = Path(a.conditions)
    if cond_path.exists():
        content_rows = read_jsonl(cond_path)
        for row in content_rows:
            body = monitor_input(
                row, include_cot=True, max_action_chars=a.monitor_action_chars
            )
            prompt = prompts["content"].format(input=body)
            add_exact_call(
                calls,
                stage="content_mid",
                role="monitor_mid",
                prompt=prompt,
                models=models,
                cache_root=a.cache_root,
                generate_kwargs={"temperature": 0, "max_tokens": 80},
            )
    else:
        for uid, row in by_uid.items():
            original = str(row.get("cot", ""))
            variants: list[tuple[str, str | None, bool]] = [("clean", original, True)]
            for style in ("naive", "aware"):
                cand = candidates.get((uid, style))
                variants.append(
                    (
                        style,
                        str(cand.get("rewrite", "")).strip() if cand else None,
                        bool(cand),
                    )
                )
            for condition, cot, exact_cot in variants:
                if cot is None:
                    cot = "x" * len(original)
                shadow = dict(row)
                shadow["cot"] = cot
                body = monitor_input(
                    shadow, include_cot=True, max_action_chars=a.monitor_action_chars
                )
                prompt = prompts["content"].format(input=body)
                if exact_cot:
                    add_exact_call(
                        calls,
                        stage="content_mid",
                        role="monitor_mid",
                        prompt=prompt,
                        models=models,
                        cache_root=a.cache_root,
                        generate_kwargs={"temperature": 0, "max_tokens": 80},
                    )
                else:
                    add_estimated_call(
                        calls,
                        synthetic_key=f"estimate:content_mid:{uid}:{condition}",
                        stage="content_mid",
                        role="monitor_mid",
                        prompt_chars=len(prompt),
                    )
    info = {
        "rows": len(rows),
        "honest": sum(r.get("mode") == "honest" for r in rows),
        "attack": sum(r.get("mode") == "attack" for r in rows),
        "candidates": len(candidates),
        "conditions_exists": cond_path.exists(),
    }
    return calls, info


def money(x: float) -> str:
    return f"${x:,.2f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trajectories", default="data/raw/trajectories.jsonl")
    ap.add_argument("--candidates", default="data/raw/rewrite_candidates.jsonl")
    ap.add_argument("--conditions", default="data/raw/conditions.jsonl")
    ap.add_argument("--models", default="config/models.yaml")
    ap.add_argument("--cache-root", default="cache/llm")
    ap.add_argument("--actions-prompt", default="config/prompts/actions_only.txt")
    ap.add_argument("--content-prompt", default="config/prompts/content_monitor.txt")
    ap.add_argument("--naive-prompt", default="config/prompts/obfuscate_naive.txt")
    ap.add_argument("--aware-prompt", default="config/prompts/obfuscate_aware.txt")
    ap.add_argument("--judge-prompt", default="config/prompts/rewrite_leak_judge.txt")
    ap.add_argument("--monitor-action-chars", type=int, default=120_000)
    ap.add_argument("--rewrite-action-chars", type=int, default=24_000)
    ap.add_argument("--rewrite-temperature", type=float, default=0.4)
    ap.add_argument("--rewrite-max-tokens", type=int, default=16_000)
    ap.add_argument(
        "--salt", default="v1", help="must match the obfuscator salt you plan to use"
    )

    # $ / 1M tokens, pinned providers from config/models.yaml.
    ap.add_argument("--qwen-in", type=float, default=0.30)
    ap.add_argument("--qwen-out", type=float, default=2.40)
    ap.add_argument("--glm-in", type=float, default=0.37)
    ap.add_argument("--glm-out", type=float, default=1.25)
    ap.add_argument(
        "--validator-in",
        type=float,
        required=True,
        help="Parasail validator price per million input tokens",
    )
    ap.add_argument(
        "--validator-out",
        type=float,
        required=True,
        help="Parasail validator price per million output tokens",
    )
    a = ap.parse_args()

    calls, info = build_calls(a)
    prices = {
        "monitor_mid": (a.qwen_in, a.qwen_out),
        "obfuscator": (a.glm_in, a.glm_out),
        "validator": (a.validator_in, a.validator_out),
    }

    stages = [
        "actions_only",
        "rewrite",
        "rewrite_validation",
        "content_mid",
    ]
    print("Pipeline cost preflight (NO NETWORK / NO MODEL CALLS)\n")
    print(
        f"kept trajectories: {info['rows']}  "
        f"(honest {info['honest']}, attack {info['attack']})"
    )
    print(
        f"rewrite candidates present: {info['candidates']}   "
        f"conditions.jsonl present: {info['conditions_exists']}"
    )
    print(f"local paid-call cache: {a.cache_root}\n")

    print("Pinned-provider prices used ($/1M input, $/1M output):")
    print(f"  Qwen3.5-27B / NovitaAI        {a.qwen_in:g}, {a.qwen_out:g}")
    print(f"  GLM-5.3-FlashX / Z.ai        {a.glm_in:g}, {a.glm_out:g}")
    print(f"  Mistral Small / Parasail     {a.validator_in:g}, {a.validator_out:g}\n")

    header = (
        f"{'stage':22s} {'calls':>7s} {'cached':>7s} {'remain':>7s} "
        f"{'low':>10s} {'base':>10s} {'high':>10s} {'est.':>6s}"
    )
    print(header)
    print("-" * len(header))
    totals = {s.name: 0.0 for s in SCENARIOS}
    total_calls = total_cached = total_est = 0
    for stage in stages:
        group = [c for c in calls.values() if c.stage == stage]
        cached = sum(c.cached for c in group)
        estimated = sum(not c.exact for c in group)
        vals = {s.name: sum(cost(c, s, prices) for c in group) for s in SCENARIOS}
        for k, v in vals.items():
            totals[k] += v
        total_calls += len(group)
        total_cached += cached
        total_est += estimated
        print(
            f"{stage:22s} {len(group):7d} {cached:7d} {len(group)-cached:7d} "
            f"{money(vals['low']):>10s} {money(vals['base']):>10s} {money(vals['high']):>10s} "
            f"{estimated:6d}"
        )

    print("-" * len(header))
    print(
        f"{'TOTAL REMAINING':22s} {total_calls:7d} {total_cached:7d} {total_calls-total_cached:7d} "
        f"{money(totals['low']):>10s} {money(totals['base']):>10s} {money(totals['high']):>10s} "
        f"{total_est:6d}"
    )

    print("\nScenario assumptions:")
    for s in SCENARIOS:
        print(
            f"  {s.name:4s}: {s.chars_per_token:g} chars/input-token; "
            f"short JSON output {s.short_output_tokens} tokens; "
            f"naive rewrite {s.naive_output_ratio:.0%}, aware rewrite {s.aware_output_ratio:.0%} of source length"
        )
    if total_est:
        print(
            "\n'est.' counts calls whose exact future prompt does not exist yet (usually because "
            "rewrites/conditions have not been generated). Rerun this preflight after each stage; "
            "the estimate becomes tighter and local cache hits reduce remaining cost to $0."
        )
    print(
        "\nThe high column uses the monitor max_tokens=80 as its short-output assumption. "
        "It is intended as a planning bound, not an OpenRouter quote."
    )


if __name__ == "__main__":
    main()
