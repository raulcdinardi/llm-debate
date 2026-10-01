"""Publish organized views off the training thread; keep receipts for resumption."""
import json
import threading
import time
from pathlib import Path

CONTRACT_VERSION = 1
CATEGORY_NAMES = {
    'mixed4': 'Warmup · Mixed4', 'warmup': 'Warmup · OpenBookQA and MMLU-Pro',
    'cw': 'CW · baseline, debate, consultancy and pairwise',
    'mbpp': 'MBPP · training and evaluation', 'sft': 'Judge SFT · training and CW validation', 'training': 'Training · organized',
}


def metadata(config, output_dir):
    env = str(config.get('env_name', '')).lower()
    category = ('mixed4' if env == 'mixed_label_pairwise' else
                'mbpp' if 'python' in env or 'mbpp' in env else
                'cw' if 'writing' in env or 'story' in env else
                'warmup' if any(x in env for x in ('qa', 'mmlu', 'openbook')) else 'training')
    category = config.get('observability_category', 'sft' if config.get('mode') == 'sft' else category)
    if category not in CATEGORY_NAMES:
        raise ValueError(f'Unknown dashboard category: {category}')
    phase = 'phase0' if any(word in str(output_dir).lower() for word in ('phase0', 'smoke', 'preflight', 'disposable')) else 'full'
    init = config.get('init_adapter_dirs')
    model = str(config.get('model_path', config.get('model', 'model unspecified')))
    return {'observability_contract': CONTRACT_VERSION, 'observability_category': category,
            'observability_phase': phase,
            'observability_initialization': json.dumps(init, sort_keys=True) if init else f'fresh adapters · {model}'}


def default_name(config, output_dir, explicit=None):
    # Preserve intentionally descriptive names; enrich old generic launch names.
    if explicit and ' · ' in explicit:
        return explicit
    meta = metadata(config, output_dir)
    model = str(config.get('model_path', config.get('model', 'model unspecified'))).rsplit('/', 1)[-1]
    init = config.get('init_adapter_dirs')
    if isinstance(init, dict) and init:
        lineage = ', '.join(f'{role}={Path(str(path)).parent.name}/{Path(str(path)).name}' for role, path in sorted(init.items()))
    else:
        lineage = 'fresh adapters' if not init else str(init)
    return f"{explicit or meta['observability_category']} · {model} · init: {lineage} · {meta['observability_phase']}"


def inventory(view):
    def key(x): return x if isinstance(x, str) else x.name
    return [(s.name, [(p.title, key(p.x), [key(y) for y in p.y]) for p in s.panels]) for s in view.sections]


def save_verified(view):
    from wandb_workspaces.workspaces import Workspace
    expected = inventory(view)
    view = view.save()
    for attempt in range(3):
        try:
            fresh = Workspace.from_url(view.url)
            break
        except ValueError as exc:
            if 'not found' not in str(exc) or attempt == 2:
                raise
            time.sleep(2)
    if inventory(fresh) != expected:
        raise ValueError('Saved dashboard panel order differs from requested order')
    if any(p.smoothing_type != 'average' or p.smoothing_factor != 5 or not p.smoothing_show_original
           for s in fresh.sections for p in s.panels):
        raise ValueError('Saved dashboard smoothing differs from requested display')
    from wandb_workspaces.expr import expr_to_filters
    if expr_to_filters(fresh.runset_settings.filters).model_dump() != expr_to_filters(view.runset_settings.filters).model_dump():
        raise ValueError('Saved dashboard run filter differs')
    return fresh


def category_view(entity, project, category, sections):
    """Keep historical run filters and admit future full runs by metadata."""
    import wandb
    from wandb_workspaces._graphql import execute_graphql
    from wandb_workspaces.workspaces import Workspace
    from llm_local_rl.workspace import workspace
    name = CATEGORY_NAMES[category]
    query = '''query Views($entity: String, $project: String) {project(name:$project,entityName:$entity) {allViews(viewType:"project-view") {edges {node {name displayName}}}}}'''
    data = execute_graphql(wandb.Api(timeout=30), query, {'entity': entity, 'project': project})
    matches = [e['node'] for e in data['project']['allViews']['edges'] if e['node']['displayName'] == name]
    clause = f"Config('observability_category') == '{category}' and Config('observability_phase') == 'full'"
    if matches:
        query_name = matches[0]['name']
        if query_name.startswith('nw-') and query_name.endswith('-v'):
            query_name = query_name[3:-2]
        view = Workspace.from_url(f"https://wandb.ai/{entity}/{project}?nw={query_name}")
        # Existing category panels include historical axes and gold definitions.
        by_name = {s.name: s for s in view.sections}
        additions = workspace(entity, project, name, sections)
        for section in additions.sections:
            if section.name not in by_name:
                view.sections.append(section)
                continue
            target = by_name[section.name]
            present = {(str(p.x), tuple(map(str, p.y))) for p in target.panels}
            for panel in section.panels:
                identity = str(panel.x), tuple(map(str, panel.y))
                if identity not in present:
                    target.panels.append(panel)
        old = view.runset_settings.filters
        if 'observability_category' not in str(old):
            view.runset_settings.filters = f'({old}) or ({clause})' if old else clause
    else:
        view = workspace(entity, project, name, sections)
        view.runset_settings.filters = clause
    from llm_local_rl.workspace import panel_order
    for index, section in enumerate(view.sections):
        if section.name[:2].isdigit():
            rank = int(section.name[:2]) - 1
            # Historical custom outcomes may not be in the generic four-metric list.
            if rank > 0:
                section.panels.sort(key=lambda p: panel_order(rank, p.y[0] if isinstance(p.y[0], str) else p.y[0].name))
        section.is_open = index < 3
        for panel in section.panels:
            panel.smoothing_type = 'average'; panel.smoothing_factor = 5; panel.smoothing_show_original = True
    view.settings.max_runs = 100
    view.settings.sort_panels_alphabetically = False
    return save_verified(view)


class DashboardSync:
    def __init__(self, state_dir, run, config, *, training_x='train_step'):
        self.state = Path(state_dir)
        self.run = run
        self.config = config
        self.training_x = training_x
        self.path = self.state / 'dashboard_state.json'
        self.lock = threading.Lock()
        saved = json.loads(self.path.read_text()) if self.path.exists() else {}
        if saved and saved['run_id'] != run.id:
            raise ValueError('Dashboard receipt belongs to another run')
        self.metrics = set(saved.get('metrics', []))
        self.published = set()
        self.url = saved.get('url')
        self.category_url = saved.get('category_url')
        self.attempts = 0
        self.last_attempt = 0

    def observe(self, metrics):
        with self.lock:
            self.metrics.update(k for k in metrics if k.startswith(('rollout/', 'reward_component/', 'train/', 'judge_eval/', 'llm_eval/')))

    def publish(self, force=False):
        with self.lock:
            metrics = set(self.metrics)
        if not metrics or metrics == self.published or self.attempts >= 3:
            return
        if not force and time.monotonic() - self.last_attempt < 60:
            return
        from llm_local_rl.workspace import layout, workspace
        from wandb_workspaces.workspaces import Workspace
        self.last_attempt = time.monotonic()
        self.attempts += 1
        configs = [json.loads(p.read_text()) for p in sorted((self.state / "llm_scoring").glob("*/config.json"))]
        sections = layout(metrics, configs, training_x=self.training_x)
        (self.state / 'dashboard.json').write_text(json.dumps(sections, indent=2))
        desired = workspace(self.run.entity, self.run.project, f'{self.run.name} · organized', sections, self.run.id)
        if self.url:
            view = Workspace.from_url(self.url)
            view.sections, view.settings, view.runset_settings = desired.sections, desired.settings, desired.runset_settings
        else:
            view = desired
        view = save_verified(view)
        self.url = view.url
        # Save the per-run URL before a category failure so a retry reuses it.
        self._receipt(metrics)
        category = self.config['observability_category']
        if self.config['observability_phase'] == 'full':
            self.category_url = category_view(self.run.entity, self.run.project, category, sections).url
        self.run.summary['dashboard_url'] = self.url
        if self.category_url:
            self.run.summary['dashboard_category_url'] = self.category_url
        self.published = metrics
        self.attempts = 0
        self._receipt(metrics)
        print(f'Organized W&B dashboard: {self.url}', flush=True)

    def _receipt(self, metrics):
        tmp = self.path.with_suffix('.tmp')
        tmp.write_text(json.dumps({'run_id': self.run.id, 'url': self.url, 'category_url': self.category_url,
                                   'metrics': sorted(metrics)}, indent=2))
        tmp.replace(self.path)
        (self.state / 'dashboard_url.txt').write_text(self.url + '\n')
