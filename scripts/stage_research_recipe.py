"""Stage preserved research tools with verified frozen source; never launch jobs."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def stage(recipe: Path, source: Path, destination: Path) -> None:
    manifest = json.loads((recipe / 'manifest.json').read_text())
    inputs = []
    for name, row in manifest['files'].items():
        inputs.append((recipe / name, name, row['sha256']))
    for name, digest in manifest['source_files'].items():
        inputs.append((source / name, 'source/' + name, digest))
    for path, relative, digest in inputs:
        if Path(relative).is_absolute() or '..' in Path(relative).parts:
            raise ValueError(f'Invalid recipe path: {relative}')
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f'Frozen input mismatch: {path}')
    destination.mkdir(parents=True, exist_ok=False)
    for path, relative, _ in inputs:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    for name in ['outputs', 'execution', 'inputs']:
        (destination / name).mkdir(exist_ok=True)
    (destination / 'staging_receipt.json').write_text(json.dumps(manifest, indent=2) + '\n')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('recipe', choices=['late_crossplay', 'early_amplification', 'orthogonal_r1'])
    parser.add_argument('--source-directory', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    stage(ROOT / 'research/cw_mechanisms' / args.recipe, args.source_directory, args.destination)
    print(f'Staged {args.recipe}; supply verified data/model inputs and an experiment release before execution.')


if __name__ == '__main__':
    main()
