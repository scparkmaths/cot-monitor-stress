"""Connectivity check: each role's model answers through its pinned provider."""

import asyncio
import sys

import yaml
from dotenv import load_dotenv
from inspect_ai.model import get_model

load_dotenv()
ROLES = sys.argv[1:] or [
    "agent_strong",
    "agent_baseline",
    "obfuscator",
    "monitor_mid",
    "validator",
]


async def ping(role, entry):
    # No reasoning_enabled here: some models (gpt-oss, GLM) reject "disable reasoning".
    model = get_model(
        f"openrouter/{entry['slug']}",
        provider={"order": [entry["provider"]], "allow_fallbacks": False},
    )
    try:
        out = await model.generate("Reply with exactly: OK")
        return role, "PASS", out.completion.strip()[:60]
    except Exception as e:
        # Inspect puts the request dump first; the provider's reason is at the end.
        return role, "FAIL", "..." + str(e)[-800:]


async def main():
    models = yaml.safe_load(open("config/models.yaml"))["openrouter"]
    results = await asyncio.gather(*(ping(r, models[r]) for r in ROLES))
    for role, status, detail in results:
        print(f"{role:15s} {status}  {detail}\n")
    return 0 if all(s == "PASS" for _, s, _ in results) else 1


sys.exit(asyncio.run(main()))
