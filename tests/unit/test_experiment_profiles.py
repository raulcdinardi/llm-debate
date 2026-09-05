import json
from pathlib import Path
import sys
from unittest.mock import patch

import pytest
from scripts.experiment_profile import render
from scripts import run_train
from llm_local_rl.config import TrainRunConfig

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('path', sorted((ROOT / 'profiles').glob('*.json')), ids=lambda p: p.stem)
def test_profile_builds_real_config_without_starting_driver(path):
    profile = json.loads(path.read_text())
    argv = render(profile, {name: '/deployment/' + name for name in profile['bindings']})
    captured = []
    class DriverProbe:
        def __init__(self, *, config):
            assert isinstance(config, TrainRunConfig)
            captured.append(config)
        def run(self):
            pass
    with patch.object(sys, 'argv', ['run_train.py', *argv]), patch('llm_local_rl.driver.TrainingDriver', DriverProbe):
        assert run_train.main() == 0
    config = captured[0]
    assert config.rollout.num_groups * config.rollout.group_size == 128
    if path.stem.startswith('cw_fullgap') or path.stem == 'cw_softdelta_baseline':
        assert config.debate_r23_reward == 'soft_judge_raw'
        assert config.debate_r1_reward == 'judge_soft_delta_task'
        assert config.train_adapter_names == ('solution', 'debate')
    if path.stem == 'mmlu_prompt_grpo':
        assert config.debate_r23_reward == 'soft_judge_prompt_grpo'
        assert config.train_adapter_names == ('debate',)


def test_missing_bindings_fail_before_execution():
    with pytest.raises(ValueError, match='Missing deployment bindings'):
        render({'bindings': ['MODEL'], 'argv': ['${MODEL}']}, {})
