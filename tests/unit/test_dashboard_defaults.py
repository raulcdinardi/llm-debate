from types import SimpleNamespace

import pytest
from llm_local_rl.dashboard_sync import DashboardSync, metadata
from llm_local_rl.judge_accuracy import judge_accuracy_metrics, GOLD_PREFIX
from llm_local_rl.workspace import layout, workspace
from scripts.prepare_observability import check, FILES


def pair(a=1, b=0, verdict='A', gold='A'):
    return {'verdict': verdict, 'trajectory_a': {'task_reward_metrics': {'is_correct': a, 'gold_agent': gold}},
            'trajectory_b': {'task_reward_metrics': {'is_correct': b, 'gold_agent': gold}}}


def test_dataset_gold_ties_unknown_invalid_and_empty():
    rows = [pair(), pair(0, 1, 'A', 'B'), pair(1, 1), pair(0, 0), pair(None), pair(verdict='invalid'), pair(gold='B')]
    m = judge_accuracy_metrics({'sample_records': rows})
    assert m[GOLD_PREFIX + 'accuracy'] == 1/3
    assert m[GOLD_PREFIX + 'gold_tie_fraction'] == 2/5
    assert m[GOLD_PREFIX + 'missing_gold_count'] == 2
    assert m[GOLD_PREFIX + 'invalid_verdict_count'] == 1
    assert GOLD_PREFIX + 'accuracy' not in judge_accuracy_metrics({'sample_records': [pair(1, 1)]})


def test_native_layout_discovers_late_scores_and_gold():
    sections = layout({GOLD_PREFIX+'accuracy', GOLD_PREFIX+'gold_tie_fraction',
                       'llm_eval/quality/r6/overall', 'llm_eval/quality/r6/coverage', 'train/judge/loss'})
    assert [p['metric'] for p in sections[1]['panels']] == ['llm_eval/quality/r6/overall', 'llm_eval/quality/r6/coverage']
    assert all(p['x'] == 'evaluation_step' for p in sections[1]['panels'])
    assert sections[2]['panels'][0]['title'].startswith('judge acc (picks gold answer)')
    v = workspace('entity', 'project', 'name', sections, 'run')
    assert 'run' in v.runset_settings.filters
    assert all(p.smoothing_type == 'average' and p.smoothing_factor == 5 and p.smoothing_show_original for s in v.sections for p in s.panels)


def test_dashboard_resume_and_late_metric_refresh(tmp_path, monkeypatch):
    import llm_local_rl.dashboard_sync as module
    import wandb_workspaces.workspaces as ws
    calls, saved = [], {}
    def save(view):
        saved['view'] = view
        calls.append(view)
        return SimpleNamespace(url='https://wandb.ai/entity/project?nw=test')
    monkeypatch.setattr(module, 'save_verified', save)
    monkeypatch.setattr(module, 'category_view', lambda *a: SimpleNamespace(url='category'))
    monkeypatch.setattr(ws.Workspace, 'from_url', lambda url: saved['view'])
    run = SimpleNamespace(id='run', entity='entity', project='project', name='Run', summary={})
    config = metadata({'env_name':'mixed_label_pairwise'}, '/full')
    d = DashboardSync(tmp_path, run, config)
    d.observe({'rollout/mean_reward':1}); d.publish(force=True)
    d.publish(force=True); assert len(calls) == 1
    d.observe({'llm_eval/test/r2/overall':3}); d.publish(force=True)
    assert len(calls) == 2 and run.summary['dashboard_category_url'] == 'category'
    resumed = DashboardSync(tmp_path, run, config)
    assert resumed.url.endswith('nw=test') and 'llm_eval/test/r2/overall' in resumed.metrics
    with pytest.raises(ValueError, match='another run'):
        DashboardSync(tmp_path, SimpleNamespace(id='other'), config)


def test_stale_launch_source_is_rejected(tmp_path):
    a, b = tmp_path/'source', tmp_path/'canonical'
    for name in FILES:
        for root in (a, b):
            path = root/'src/llm_local_rl'/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(name)
    assert len(check(a, b)) == len(FILES)
    (a/'src/llm_local_rl/observability.py').write_text('old logger')
    with pytest.raises(ValueError, match='Stale observability'):
        check(a, b)


def test_phase0_does_not_join_full_category():
    assert metadata({'env_name':'mixed_label_pairwise'}, '/experiment/phase0')['observability_phase'] == 'phase0'


def test_training_hook_publishes_layout_and_late_metrics(tmp_path, monkeypatch):
    import sys
    from llm_local_rl.observability import RunObservability, WandbSettings
    from llm_local_rl.live_scoring import digest
    import json
    published = []
    run = SimpleNamespace(id='run', url='run-url', config={}, summary={}, name='name',
                          define_metric=lambda *a, **k: None, log=lambda *a, **k: None, finish=lambda: None)
    monkeypatch.setitem(sys.modules, 'wandb', SimpleNamespace(init=lambda **kwargs: run))
    monkeypatch.setattr(DashboardSync, 'publish', lambda self, **kw: published.append(set(self.metrics)))
    observer = RunObservability(output_dir=tmp_path, config={'env_name':'mixed_label_pairwise'},
                                settings=WandbSettings(upload_artifacts=False))
    observer.log_step({'step':2, 'mean_reward':.5})
    path = tmp_path/'observability/llm_scoring/test/events.json';path.parent.mkdir(parents=True)
    event = {'run_id':observer._score_inbox.run_id,'step':1,'metrics':{'llm_eval/test/r2/overall':3}}
    event['id']=digest(event);path.write_text(json.dumps([event]))
    observer.finish()
    assert published == [{'rollout/mean_reward','llm_eval/test/r2/overall'}]
    assert run.config['observability_category']=='mixed4'
    assert (tmp_path/'observability/wandb_finished').exists()


def test_generic_names_show_initialization_and_rich_names_survive():
    from llm_local_rl.dashboard_sync import default_name
    cfg={'model_path':'LiquidAI/LFM2.5-2.6B-Base','init_adapter_dirs':{'judge':'/checkpoints/OBQA104/judge'}}
    name=default_name(cfg,'/full','CW-full')
    assert 'LFM2.5-2.6B-Base' in name and 'judge=OBQA104/judge' in name
    assert default_name(cfg,'/full','CW · exact lineage')=='CW · exact lineage'


def test_workspace_title_handles_initializer_symbols():
    v = workspace('e', 'p', 'CW · debate+judge · init: judge=OBQA104', [])
    assert 'debate and judge' in v.name and 'judge: OBQA104' in v.name


def test_category_keeps_original_url_and_historical_filter(monkeypatch):
    import wandb
    import wandb_workspaces.workspaces as ws
    import wandb_workspaces._graphql as gql
    import llm_local_rl.dashboard_sync as module
    view = workspace('e', 'p', 'Warmup · Mixed4', layout({'rollout/mean_reward'}))
    view.runset_settings.filters='ID == "historical"'
    urls=[]
    monkeypatch.setattr(wandb, 'Api', lambda **kw: object())
    monkeypatch.setattr(gql, 'execute_graphql', lambda *a: {'project':{'allViews':{'edges':[{'node':{'name':'nw-original-v','displayName':'Warmup · Mixed4'}}]}}})
    monkeypatch.setattr(ws.Workspace, 'from_url', lambda url: urls.append(url) or view)
    monkeypatch.setattr(module, 'save_verified', lambda value:value)
    result=module.category_view('e','p','mixed4',layout({'rollout/mean_reward','llm_eval/test/r2/overall'}))
    assert urls==['https://wandb.ai/e/p?nw=original']
    assert 'historical' in result.runset_settings.filters and 'observability_category' in result.runset_settings.filters
    assert 'observability_phase' in result.runset_settings.filters
    assert sum(len(s.panels) for s in result.sections)==2


def test_archive_check_catches_git_archive_omitting_new_modules(tmp_path):
    import io,tarfile
    from scripts.prepare_observability import check_archive
    canonical=tmp_path/'repo'
    for name in FILES:
        p=canonical/'src/llm_local_rl'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(name)
    archive=tmp_path/'source.tar.gz'
    def build(names):
        with tarfile.open(archive,'w:gz') as tar:
            for name in names:
                tar.add(canonical/'src/llm_local_rl'/name,arcname='src/llm_local_rl/'+name)
    build(FILES[:-1])
    with pytest.raises(ValueError,match='missing'):
        check_archive(archive,canonical)
    build(FILES)
    assert len(check_archive(archive,canonical))==len(FILES)


def test_driver_nested_rollout_selects_category():
    from llm_local_rl.config import TrainRunConfig, RolloutConfig
    from llm_local_rl.dashboard_sync import metadata
    for env, expected in [("python_optimization", "mbpp"), ("constrained_writing", "cw"),
                          ("mixed_label_pairwise", "mixed4")]:
        config = TrainRunConfig(model_path="model", output_dir="full", mmlu_pro_data_path="tasks.jsonl", rollout=RolloutConfig(env_name=env))
        assert metadata(config.to_dict(), "full")["observability_category"] == expected


def test_failed_publication_has_one_persisted_final_retry(tmp_path, monkeypatch):
    import json
    import llm_local_rl.dashboard_sync as module
    run = SimpleNamespace(id='run', entity='entity', project='project', name='Run', summary={})
    cfg = metadata({'env_name': 'mixed_label_pairwise'}, '/full')
    calls = []
    def unavailable(view):
        calls.append(view)
        raise ConnectionError('transient service outage')
    monkeypatch.setattr(module, 'save_verified', unavailable)
    d = DashboardSync(tmp_path, run, cfg)
    d.observe({'rollout/mean_reward': 1})
    for attempt in range(3):
        d.last_attempt = -100
        with pytest.raises(ConnectionError):
            d.publish()
    resumed = DashboardSync(tmp_path, run, cfg)
    resumed.publish()
    assert len(calls) == 3
    with pytest.raises(ConnectionError):
        resumed.publish(force=True)
    resumed.publish(force=True)
    assert len(calls) == 4
    assert json.loads((tmp_path/'dashboard_state.json').read_text())['needs_intervention']
