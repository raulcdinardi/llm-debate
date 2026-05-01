from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys


UNIT_TARGETS = [
    "tests/unit/test_ht_sequence_env.py",
    "tests/unit/test_masking.py",
    "tests/unit/test_episode_routing.py",
    "tests/unit/test_driver.py",
    "tests/unit/test_debate_training_projection.py",
    "tests/unit/test_debate_source_parity.py",
    "tests/unit/test_base_model_judge.py",
    "tests/unit/test_debate_judge_fn.py",
]

INTEGRATION_TARGETS = [
    "tests/integration/test_vllm_lora_load.py",
    "tests/integration/test_sampling_parity.py",
    "tests/integration/test_training_replay_parity.py",
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _pytest_env() -> dict[str, str]:
    env = dict(os.environ)
    src_path = str(_repo_root() / "src")
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = src_path if not existing else src_path + os.pathsep + existing
    return env


def _run_pytest(targets: list[str]) -> int:
    cmd = [sys.executable, "-m", "pytest", "-q", *targets]
    return subprocess.run(cmd, cwd=_repo_root(), env=_pytest_env()).returncode


def _integration_prereqs_available() -> tuple[bool, list[str]]:
    missing: list[str] = []
    for name in ("torch", "transformers", "peft", "vllm"):
        if importlib.util.find_spec(name) is None:
            missing.append(name)
    for env_name in ("LLM_LOCAL_RL_BASE_MODEL", "LLM_LOCAL_RL_ADAPTER_A", "LLM_LOCAL_RL_ADAPTER_B"):
        if not os.environ.get(env_name):
            missing.append(env_name)
    return len(missing) == 0, missing


def main() -> int:
    print("== Unit tests ==")
    unit_rc = _run_pytest(UNIT_TARGETS)

    ok_integration, missing = _integration_prereqs_available()
    if not ok_integration:
        print("== Integration tests ==")
        print("SKIP integration: missing " + ", ".join(missing))
        return unit_rc

    print("== Integration tests ==")
    integration_rc = _run_pytest(INTEGRATION_TARGETS)
    return unit_rc or integration_rc


if __name__ == "__main__":
    raise SystemExit(main())
