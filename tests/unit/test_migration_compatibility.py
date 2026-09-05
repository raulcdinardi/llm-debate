from pathlib import Path
import importlib.util

from scripts.verify_migration import verify
from llm_local_rl import prompts


def test_legacy_files_and_current_prompt_bytes_are_preserved():
    assert verify() == []


def test_runtime_prompt_loaders_use_separate_roots():
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location('legacy_prompts_probe', root / 'src/tinker_debate/prompts.py')
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)
    assert legacy._PROMPT_DIR == root / 'prompts'
    assert prompts._PROMPT_DIR == root / 'prompts/local_rl'
    for name in ['debate/system_judge.md', 'debate/r3_user_template.md']:
        assert legacy.load_prompt(name) == (root / 'prompts' / name).read_text()
        assert prompts.load_prompt(name) == (root / 'prompts/local_rl' / name).read_text()
