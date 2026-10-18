"""Collect honest/attack trajectories with one untrusted agent, no intervention.

  uv run python scripts/run_episodes.py --log-dir logs/smoke --agent agent_strong \
      --honest-epochs 1 --attack-epochs 1 --limit 1
  uv run python scripts/run_episodes.py --log-dir logs/main --agent agent_strong \
      --honest-epochs 40 --attack-epochs 6
"""

import argparse
import sys

import yaml
from dotenv import load_dotenv

load_dotenv()

from control_arena import EvalMode  # noqa: E402
from control_arena.eval import ControlEvalConfig, get_control_task  # noqa: E402
from control_arena.policy import untrusted_policy_builder  # noqa: E402
from inspect_ai import eval  # noqa: E402
from inspect_ai.model import get_model  # noqa: E402


def make_setting(name: str):
    if name == "iac":
        from control_arena.settings.iac.control_iac_setting import ControlIACSetting

        return ControlIACSetting()
    if name == "bash":
        from control_arena.settings.bash.bash_setting import BashSetting

        return BashSetting()
    sys.exit(f"unknown setting: {name}")


def openrouter_model(e: dict, reasoning: bool | None):
    kw = {"provider": {"order": [e["provider"]], "allow_fallbacks": False}}
    if reasoning is not None:  # omit for models that reject disabling reasoning
        kw["reasoning_enabled"] = reasoning
    return get_model(f"openrouter/{e['slug']}", **kw)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--log-dir", required=True, help="e.g. logs/main, logs/pilot, logs/bash_probe"
    )
    ap.add_argument("--setting", default="iac", choices=["iac", "bash"])
    ap.add_argument(
        "--agent", default="agent_strong", choices=["agent_strong", "agent_baseline"]
    )
    ap.add_argument("--honest-epochs", type=int, default=0)
    ap.add_argument("--attack-epochs", type=int, default=0)
    ap.add_argument(
        "--limit", type=int, default=None, help="cap samples per mode (pilots)"
    )
    a = ap.parse_args()

    models = yaml.safe_load(open("config/models.yaml"))["openrouter"]
    config = ControlEvalConfig(
        untrusted_model=openrouter_model(models[a.agent], reasoning=True),
        trusted_model=openrouter_model(models["trusted"], reasoning=None),
    )
    setting = make_setting(a.setting)

    for mode, epochs in [
        (EvalMode.HONEST, a.honest_epochs),
        (EvalMode.ATTACK, a.attack_epochs),
    ]:
        if epochs <= 0:
            continue
        task = get_control_task(
            setting=setting,
            micro_protocol=untrusted_policy_builder(),
            mode=mode,
            eval_config=config,
            task_name_prefix=f"{a.setting}_{a.agent}",
        )
        n = min(len(task.dataset), a.limit or len(task.dataset))
        print(f"{a.setting} {mode.name}: {n} samples x {epochs} epochs -> {a.log_dir}")
        eval(task, limit=a.limit, epochs=epochs, log_dir=a.log_dir)


if __name__ == "__main__":
    main()
