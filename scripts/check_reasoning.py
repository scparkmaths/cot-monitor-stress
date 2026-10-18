"""Gate check: does an OpenRouter model, pinned to one upstream provider,
(a) return its raw reasoning text and (b) make a tool call?

Usage:
  uv run python scripts/check_reasoning.py --role agent_strong --provider <UpstreamName>
  uv run python scripts/check_reasoning.py --role agent_baseline        # uses provider from models.yaml
"""

import argparse
import asyncio
import sys

import yaml
from dotenv import load_dotenv
from inspect_ai.model import ContentReasoning, get_model
from inspect_ai.tool import tool

load_dotenv()

MIN_REASONING_CHARS = 200
PROMPT = (
    "Find files larger than 10MB under /var/log. Call the list_files tool on that "
    "directory, then say which files qualify. Think it through before acting."
)


@tool
def list_files():
    async def execute(path: str) -> str:
        """List the files in a directory with their sizes in bytes.

        Args:
            path: Directory to list.
        """
        return "syslog 52428800\nauth.log 1048576\nkern.log 20971520\n"

    return execute


def is_placeholder(v) -> bool:
    return not v or (isinstance(v, str) and v.startswith("["))


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--role", required=True, help="key under openrouter: in config/models.yaml"
    )
    ap.add_argument(
        "--provider", help="upstream provider to test (overrides models.yaml)"
    )
    args = ap.parse_args()

    entry = yaml.safe_load(open("config/models.yaml"))["openrouter"][args.role]
    slug = entry["slug"]
    provider = args.provider or entry["provider"]
    if is_placeholder(slug):
        print(f"[{args.role}] slug is still a placeholder: {slug}")
        return 2
    if is_placeholder(provider):
        print(f"[{args.role}] no provider given: pass --provider or fill models.yaml")
        return 2

    print(f"testing openrouter/{slug} via upstream '{provider}' (fallbacks off)")
    model = get_model(
        f"openrouter/{slug}",
        reasoning_enabled=True,
        provider={"order": [provider], "allow_fallbacks": False},
    )
    out = await model.generate(PROMPT, tools=[list_files()])
    msg = out.message

    content = msg.content if isinstance(msg.content, list) else []
    blocks = [c for c in content if isinstance(c, ContentReasoning)]
    redacted = [b for b in blocks if getattr(b, "redacted", False)]
    reasoning = "\n".join(
        b.reasoning for b in blocks if not getattr(b, "redacted", False)
    )
    tool_calls = [tc.function for tc in (msg.tool_calls or [])]

    print(f"reasoning blocks: {len(blocks)} (redacted: {len(redacted)})")
    print(f"reasoning chars:  {len(reasoning)}")
    print(f"tool calls:       {tool_calls}")
    print(
        "--- first 600 chars of reasoning (check it reads as raw thinking, not a summary) ---"
    )
    print(reasoning[:600] or "<none>")
    print("---")

    ok_reasoning = len(reasoning) >= MIN_REASONING_CHARS
    ok_tool = "list_files" in tool_calls
    print(
        f"reasoning: {'PASS' if ok_reasoning else 'FAIL'}   tool call: {'PASS' if ok_tool else 'FAIL'}"
    )
    return 0 if (ok_reasoning and ok_tool) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
