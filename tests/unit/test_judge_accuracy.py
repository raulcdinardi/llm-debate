from llm_local_rl.judge_accuracy import PREFIX, judge_accuracy_metrics
from llm_local_rl.observability import flatten_step_metrics
from llm_local_rl.metric_averaging import trailing_rows


def pair(a, b, verdict='A'):
    return {'metrics': {'task': 'python_optimization'}, 'verdict': verdict,
            'trajectory_a': {'task_reward_metrics': {'correct': a}},
            'trajectory_b': {'task_reward_metrics': {'correct': b}}}


def measure(*pairs):
    return judge_accuracy_metrics({'sample_records': list(pairs)})


def test_gold_ties_and_accuracy():
    m = measure(pair(True, False), pair(False, True), pair(True, True), pair(False, False))
    assert m[PREFIX+'accuracy'] == .5
    assert m[PREFIX+'gold_tie_fraction'] == .5
    assert m[PREFIX+'eligible_count'] == m[PREFIX+'gold_tie_count'] == 2


def test_missing_gold_is_not_failure_and_invalid_pick_is_wrong():
    m = measure(pair(None, True), pair(0, True), pair(False, True, 'invalid'))
    assert m[PREFIX+'missing_gold_count'] == 2
    assert m[PREFIX+'eligible_count'] == 1
    assert m[PREFIX+'accuracy'] == 0
    assert m[PREFIX+'invalid_verdict_count'] == 1


def test_swap_invariance():
    assert measure(pair(True, False, 'A')) == measure(pair(False, True, 'B'))
    assert measure(pair(True, False, 'B')) == measure(pair(False, True, 'A'))


def test_all_ties_omit_accuracy_and_unrelated_tasks_ignored():
    assert PREFIX+'accuracy' not in measure(pair(True, True))
    assert PREFIX+'gold_tie_fraction' not in measure(pair(None, False))
    unrelated = pair(True, False)
    unrelated['metrics']['task'] = 'creative_writing'
    assert measure(unrelated) == {}
    assert judge_accuracy_metrics({}) == {}


def test_logging_and_trailing_missing_denominators():
    rows = [{'step': i+1, 'metrics': flatten_step_metrics({'sample_records': [p]})}
            for i, p in enumerate([pair(True, False), pair(True, True), pair(False, True)])]
    averaged = list(trailing_rows(rows))
    assert 'trailing5/'+PREFIX+'accuracy' not in averaged[1]
    assert averaged[2]['trailing5/'+PREFIX+'accuracy'] == .5


def test_accuracy_and_ties_lead_judge_section():
    from scripts.build_rollout_workspace import layout
    panels = layout(set(measure(pair(True, False), pair(False, False))))[2]['panels']
    assert [p['metric'] for p in panels[:4]] == [PREFIX+k for k in
        ('accuracy', 'gold_tie_fraction', 'gold_tie_count', 'eligible_count')]
    assert panels[0]['title'] == 'judge acc (picks pass test) · excludes ties'


def test_native_smoothing_uses_raw_metrics_and_overlay():
    from scripts.build_rollout_workspace import layout, workspace
    view = workspace('entity', 'project', 'test', layout(set(measure(pair(True, False)))))
    assert view.settings.smoothing_type == 'average'
    assert view.settings.smoothing_weight == 5
    for section in view.sections:
        for panel in section.panels:
            assert panel.smoothing_type == 'average'
            assert panel.smoothing_factor == 5
            assert panel.smoothing_show_original is True
            assert not panel.y[0].startswith('trailing5/')
