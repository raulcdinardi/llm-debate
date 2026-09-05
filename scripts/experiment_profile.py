"""Render a frozen experiment profile; execution is explicit and uses argv, not a shell."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shlex
from string import Template
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def render(profile: dict, bindings: dict[str, str]) -> list[str]:
    missing = set(profile['bindings']) - bindings.keys()
    if missing:
        raise ValueError(f'Missing deployment bindings: {sorted(missing)}')
    return [Template(arg).substitute(bindings) for arg in profile['argv']]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('profile', type=Path)
    parser.add_argument('--bindings', type=Path, required=True, help='JSON object of deployment paths/names')
    parser.add_argument('--run', action='store_true', help='Launch training; otherwise only print the command')
    args = parser.parse_args()
    argv = render(json.loads(args.profile.read_text()), json.loads(args.bindings.read_text()))
    sys.path.insert(0, str(ROOT / 'src'))
    from run_train import parse_args
    parse_args(argv)
    command = [sys.executable, str(ROOT / 'scripts/run_train.py'), *argv]
    if not args.run:
        print(shlex.join(command))
        return 0
    env = dict(os.environ)
    env['PYTHONPATH'] = str(ROOT / 'src') + os.pathsep + env.get('PYTHONPATH', '')
    return subprocess.run(command, env=env).returncode


if __name__ == '__main__':
    raise SystemExit(main())
