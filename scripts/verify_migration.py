"""Offline preservation check; independent of Git history and GPU dependencies."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def verify() -> list[str]:
    errors = []
    legacy = json.loads((ROOT / 'docs/migration/legacy_inventory.json').read_text())
    runtime = json.loads((ROOT / 'docs/migration/runtime_inventory.json').read_text())
    for name, row in legacy['files'].items():
        path = ROOT / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
            errors.append(f'Legacy file changed: {name}')
    for name, row in runtime['files'].items():
        if not name.startswith('prompts/local_rl/'):
            continue
        path = ROOT / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
            errors.append(f'Local RL prompt changed: {name}')
    return errors


if __name__ == '__main__':
    failures = verify()
    if failures:
        raise SystemExit('\n'.join(failures))
    print('PASS: legacy files and current runtime prompt bytes preserved')
