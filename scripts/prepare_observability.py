"""Install/check the logging contract in an UNFROZEN launch source tree."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tarfile

FILES = ('observability.py', 'score_sync.py', 'live_scoring.py', 'judge_accuracy.py',
         'workspace.py', 'dashboard_sync.py')


def check(source, canonical):
    missing = []
    for name in FILES:
        a, b = source / 'src/llm_local_rl' / name, canonical / 'src/llm_local_rl' / name
        if not a.exists() or a.read_bytes() != b.read_bytes():
            missing.append(name)
    if missing:
        raise ValueError('Stale observability launch source: ' + ', '.join(missing))
    return {n: hashlib.sha256((source / 'src/llm_local_rl' / n).read_bytes()).hexdigest() for n in FILES}


def check_archive(archive, canonical):
    hashes = {}
    with tarfile.open(archive) as tar:
        for name in FILES:
            suffix = 'src/llm_local_rl/' + name
            matches = [m for m in tar.getmembers() if m.isfile()
                       and (m.name == suffix or m.name.endswith('/' + suffix))]
            if len(matches) != 1:
                raise ValueError(f'Launch archive missing or duplicates {suffix}')
            content = tar.extractfile(matches[0]).read()
            if content != (canonical / suffix).read_bytes():
                raise ValueError(f'Stale observability launch archive: {suffix}')
            hashes[name] = hashlib.sha256(content).hexdigest()
    return hashes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sources = p.add_mutually_exclusive_group(required=True)
    sources.add_argument('--source-root', type=Path)
    sources.add_argument('--source-archive', type=Path)
    p.add_argument('--install', action='store_true')
    p.add_argument('--runtime', action='store_true')
    args = p.parse_args()
    canonical = Path(__file__).resolve().parents[1]
    source = (args.source_root or args.source_archive).resolve()
    if args.install:
        if args.source_archive:
            p.error('--install is only allowed with --source-root')
        if source != canonical and any((a / 'spec.json').exists() for a in (source, source.parent)):
            raise SystemExit('Frozen experiment: do not alter its source. Stage a new revision.')
        for name in FILES:
            dest = source / 'src/llm_local_rl' / name
            if dest.resolve() != (canonical / 'src/llm_local_rl' / name).resolve():
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(canonical / 'src/llm_local_rl' / name, dest)
    hashes = check_archive(source, canonical) if args.source_archive else check(source, canonical)
    if args.runtime:
        for name in ('wandb', 'wandb_workspaces'):
            if importlib.util.find_spec(name) is None:
                raise SystemExit(f'Runtime missing {name}; include it before freezing/provisioning')
    print(json.dumps({'observability_contract': 1, 'source': str(source), 'hashes': hashes}, indent=2))


if __name__ == '__main__':
    main()
