import hashlib
import json
from pathlib import Path

import pytest
from scripts.stage_research_recipe import stage

ROOT = Path(__file__).resolve().parents[2]


def test_preserved_recipe_hashes_and_python_syntax():
    for recipe in (ROOT / 'research/cw_mechanisms').iterdir():
        if not recipe.is_dir():
            continue
        manifest = json.loads((recipe / 'manifest.json').read_text())
        for name, row in manifest['files'].items():
            data = (recipe / name).read_bytes()
            assert hashlib.sha256(data).hexdigest() == row['sha256']
            if name.endswith('.py'):
                compile(data, name, 'exec')


def test_staging_rejects_wrong_source_before_writing(tmp_path):
    recipe = ROOT / 'research/cw_mechanisms/early_amplification'
    destination = tmp_path / 'staged'
    with pytest.raises(ValueError, match='Frozen input mismatch'):
        stage(recipe, tmp_path / 'missing_source', destination)
    assert not destination.exists()


def test_staging_preserves_files_and_refuses_overwrite(tmp_path):
    recipe = tmp_path / 'recipe'; recipe.mkdir()
    source = tmp_path / 'source'; source.mkdir()
    (recipe / 'spec.json').write_text('{}')
    (source / 'model.py').write_text('FROZEN = True\n')
    manifest = {'files': {'spec.json': {'sha256': hashlib.sha256(b'{}').hexdigest()}},
                'source_files': {'model.py': hashlib.sha256(b'FROZEN = True\n').hexdigest()}}
    (recipe / 'manifest.json').write_text(json.dumps(manifest))
    destination = tmp_path / 'staged'
    stage(recipe, source, destination)
    assert (destination / 'source/model.py').read_bytes() == (source / 'model.py').read_bytes()
    assert not (destination / 'execution/release.json').exists()
    with pytest.raises(FileExistsError):
        stage(recipe, source, destination)
