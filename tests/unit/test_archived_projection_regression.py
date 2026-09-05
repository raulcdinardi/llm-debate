"""Compare actual training tensors against independently executed frozen run sources."""
import json
from dataclasses import asdict
from pathlib import Path

import pytest
from test_debate_training_projection import _make_debate, _soft_judge_audit, _append_fourth_round
from llm_local_rl.debate_parity import assemble_split_train_examples

FIXTURES = Path(__file__).resolve().parents[1] / 'fixtures/projection_regression'
CASES = [case for path in sorted(FIXTURES.glob('*.json')) for case in json.loads(path.read_text())['cases']]


@pytest.mark.parametrize('case', CASES, ids=lambda x: x['name'])
def test_training_tensors_match_archived_sources(case):
    debates = [
        _make_debate(instance_id='same', reward_a=1., reward_b=3., judge_raw_response=_soft_judge_audit(.25, js=.2)),
        _make_debate(instance_id='same', reward_a=4., reward_b=2., token_offset=100, judge_raw_response=_soft_judge_audit(-.5, js=.9)),
    ]
    if case['kwargs']['num_rounds'] == 4:
        debates = [_append_fourth_round(d) for d in debates]
    outputs = assemble_split_train_examples(debates=debates, task_reward_fn=lambda t,d: float(t.metrics['task_reward']), **case['kwargs'])
    actual = {key: [{k:v for k,v in asdict(row).items() if k != 'metadata'} for row in rows] for key,rows in outputs.items()}
    assert actual == case['expected']
