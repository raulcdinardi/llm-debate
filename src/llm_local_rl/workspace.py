"""Create a predictable W&B view with outcomes and evaluation before diagnostics."""
import argparse
import json
import re
import unicodedata
from pathlib import Path

from llm_local_rl.judge_accuracy import PREFIX, PANELS, GOLD_PREFIX, GOLD_PANELS


def natural_key(value):
    return tuple((1, int(part)) if part.isdigit() else (0, part)
                 for part in re.split(r"(\d+)", value))


def panel_order(section, metric):
    """Rank scientific summaries before diagnostics; pair adapters by metric."""
    leaf = metric.rsplit("/", 1)[-1]
    if section == 0:
        order = ["mean_correct", "mean_reward", "mean_speedup", "mean_parse_success"]
        return (order.index(leaf) if leaf in order else len(order), natural_key(metric))
    if section == 1:
        # Every round's scores precede coverage/counts, even with many evaluators.
        health = {"coverage": 1, "scored": 2, "failed": 3}
        headline = {"overall": 0, "quality": 1, "correctness": 2, "evidence": 3, "clarity": 4,
                    "cites_correctness": 5, "tests_or_edge_cases": 6}
        return (health.get(leaf, 0), headline.get(leaf, 10), natural_key(leaf), natural_key(metric))
    if section == 2:
        if metric.startswith((PREFIX, GOLD_PREFIX)):
            order = [key for key, _, _ in PANELS] + ["correct_count"]
            return (-1, order.index(leaf), natural_key(metric))
        families = [
            ("accuracy",),
            ("order_invariant", "order_disagreement", "coherence"),
            ("invalid_rate", "valid_rate"),
            ("soft_score_mean_abs", "soft_score_near_zero", "soft_score_mean", "soft_score_std"),
            ("display_a_probability_mean", "display_b_probability_mean", "order_bias"),
            ("a_win_rate", "b_win_rate", "win_rate"),
        ]
    elif section == 3:
        order = ["loss", "approx_kl", "clipfrac", "grad_norm", "learning_rate", "num_optimizer_steps"]
        return (order.index(leaf) if leaf in order else len(order), natural_key(metric))
    else:
        families = [
            ("nonfinite", "violations", "dropped", "parse_success", "closed_fence", "checks_"),
            ("on_policy", "logprob", "parity"),
            ("entropy", "max_token_rate", "eos_rate"),
            ("length", "debate_rounds"),
            ("reward_", "group_reward", "zero_variance"),
            ("advantage",),
            ("ratio", "clip", "kl"),
            ("grad", "adam", "update", "weight_decay"),
            ("elapsed", "wall", "time", "throughput", "benchmark", "reference_ns"),
            ("memory", "vram", "batch", "tokens", "examples"),
        ]
    family, priority = next(((i, j) for i, terms in enumerate(families)
                             for j, term in enumerate(terms) if term in leaf), (len(families), 0))
    # Keep forward/reverse versions and corresponding adapters adjacent.
    paired = leaf.replace("forward_", "").replace("reverse_", "")
    return (family, priority, natural_key(paired), natural_key(metric))


def order_panels(sections):
    for index, section in enumerate(sections):
        section["panels"].sort(key=lambda p: panel_order(index, p["metric"]))
    return sections


def layout(metrics, configs=(), training_x="train_step"):
    sections = [
        {"name": "01 · Outcomes", "panels": []},
        {"name": "02 · LLM evaluation", "panels": []},
        {"name": "03 · Judge behavior", "panels": []},
        {"name": "04 · Optimization", "panels": []},
        {"name": "05 · Rollout and runtime diagnostics", "panels": []},
    ]
    used = set()
    def panel(section, metric, title, axis=training_x, bounds=None):
        if axis != "evaluation_step" and metric not in metrics:
            return
        sections[section]["panels"].append({"metric": metric, "title": title, "x": axis, "bounds": bounds})
        used.add(metric)
    for metric, title in [
        ("reward_component/mean_correct", "Task correctness · fraction"),
        ("rollout/mean_reward", "Environment reward · mean"),
        ("rollout/mean_parse_success", "Valid output format · fraction"),
        ("reward_component/mean_speedup", "Program speedup · mean"),
    ]:
        panel(0, metric, title)
    for config in configs:
        for channel in config["rounds"]:
            prefix = f"llm_eval/{config['name']}/{channel}"
            name = config.get("title", config["name"].replace("_", " "))
            for field, bounds in config["fields"].items():
                label = config.get("field_titles", {}).get(field, field.replace("_", " ").capitalize())
                panel(1, prefix + "/" + field, f"{label} · {channel.upper()} · {name}", "evaluation_step", bounds)
            for field, label, bounds in [
                ("coverage", "Scoring coverage · fraction of eligible sampled turns", [0, 1]),
                ("scored", "Arguments scored", None), ("failed", "Arguments that could not be scored", None),
            ]:
                panel(1, prefix + "/" + field, f"{label} · {channel.upper()} · {name}", "evaluation_step", bounds)
    for key, title, bounds in PANELS:
        panel(2, PREFIX + key, title, bounds=bounds)
    for key, title, bounds in GOLD_PANELS:
        panel(2, GOLD_PREFIX + key, title, bounds=bounds)
    used.add(GOLD_PREFIX + "correct_count")
    # Numerator stays available in history without another redundant panel.
    used.add(PREFIX + "correct_count")
    priority = [
        ("reward_component/train_judge_order_invariant_rate", "Same winner after swapping A/B · fraction"),
        ("reward_component/train_judge_forward_display_a_probability_mean", "Judge probability assigned to displayed A"),
        ("reward_component/train_judge_invalid_rate", "Invalid judge decisions · fraction"),
        ("reward_component/train_judge_soft_score_mean_abs", "Judge confidence · mean absolute soft score"),
    ]
    for key, title in priority:
        panel(2, key, title)
    for metric in sorted(metrics):
        if metric in used or metric in {"train_step", "evaluation_step", "judge_eval_step"}:
            continue
        if any(k in metric for k in ("/reward_audit/", "/histogram", "/samples", "/selected_layer/")):
            continue
        if metric.startswith("train/") and "/judge" not in metric and any(k in metric for k in ("/supervised_", "/judge_")):
            continue
        if metric.startswith("llm_eval/"):
            parts = metric.split("/")
            leaf = parts[-1]
            bounds = [0, 1] if leaf == "coverage" else None
            title = f"{leaf.replace('_', ' ').capitalize()} · {parts[-2].upper()} · {parts[1]}"
            panel(1, metric, title, "evaluation_step", bounds)
            continue
        if "train_judge" in metric or metric.startswith("judge_eval/") or (metric.startswith(("train/judge/", "train/judge_shadow/")) and any(k in metric for k in ("supervised_", "coherence"))):
            section = 2
        elif metric.startswith("train/") and metric.rsplit("/", 1)[-1] in {"loss", "approx_kl", "clipfrac", "learning_rate", "grad_norm", "num_optimizer_steps"}:
            section = 3
        else:
            section = 4
        title = metric.replace("reward_component/", "").replace("_", " ").replace("/", " · ")
        if section == 2 and re.search(r"_p(?:01|05|50|95|99|999)$", metric):
            section = 4
        if metric.endswith("/supervised_label_accuracy"):
            title = "judge acc (picks gold answer) · both displayed orders · training prompts"
        panel(section, metric, title)
    return order_panels(sections)


def workspace(entity, project, name, sections, run_id=None, metric_view="native"):
    if metric_view not in ("native", "raw", "trailing5"):
        raise ValueError("metric_view must be native, raw or trailing5")
    trailing = metric_view == "trailing5"
    import wandb_workspaces.workspaces as ws
    import wandb_workspaces.reports.v2 as wr
    # The workspace SDK also rejects mathematical symbols such as + and =.
    title = name.replace("+", " and ").replace("=", ": ")
    title = ''.join(c if unicodedata.category(c) not in {"So", "Sk", "Sm", "Sc", "Cs"} else ' ' for c in title)
    return ws.Workspace(entity=entity, project=project, name=title, auto_generate_panels=False,
        settings=ws.WorkspaceSettings(sort_panels_alphabetically=False, smoothing_type="average" if metric_view == "native" else "none", smoothing_weight=5 if metric_view == "native" else 0),
        runset_settings=ws.RunsetSettings(pinned_runs=[run_id] if run_id else [], filters=f'ID == "{run_id}"' if run_id else []),
        sections=[ws.Section(name=section["name"], is_open=index < 3,
            layout_settings=ws.SectionLayoutSettings(columns=2),
            panels=[wr.LinePlot(title=p["title"] + (" · trailing 5" if trailing else ""),
                x="averaging_step" if trailing else p["x"],
                y=["trailing5/" + p["metric"] if trailing else p["metric"]],
                title_x="Training step", range_y=tuple(p["bounds"] or (None, None)),
                smoothing_type="average" if metric_view == "native" else "none",
                smoothing_factor=5 if metric_view == "native" else 0,
                smoothing_show_original=True, aggregate=False) for p in section["panels"]])
            for index, section in enumerate(sections) if section["panels"]])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--records", type=Path, required=True)
    p.add_argument("--scoring-config", type=Path, action="append", default=[])
    p.add_argument("--training-x", choices=["train_step", "Step"], default="train_step",
                   help="Use Step only for historical runs logged before custom axes")
    p.add_argument("--entity", required=True)
    p.add_argument("--project", required=True)
    p.add_argument("--run-id")
    p.add_argument("--name", default="Rollouts and LLM evaluation")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--metric-view", choices=["native", "raw", "trailing5"], default="native",
                   help="Default: native running average 5 with faint raw overlay; trailing5 is a legacy explicit export")
    p.add_argument("--publish", action="store_true", help="Save a new named W&B view")
    args = p.parse_args()
    from llm_local_rl.observability import flatten_step_metrics
    metrics = set()
    with args.records.open() as handle:
        for line in handle:
            if not line.endswith("\n"):
                break
            metrics.update(flatten_step_metrics(json.loads(line)))
    sections = layout(metrics, [json.loads(c.read_text()) for c in args.scoring_config], args.training_x)
    args.output.write_text(json.dumps(sections, indent=2))
    if args.publish:
        view = workspace(args.entity, args.project, args.name, sections, args.run_id, args.metric_view).save()
        args.output.with_suffix(".url.txt").write_text(view.url + "\n")
        print(view.url)
    else:
        print(f"Saved ordered dashboard specification: {args.output}")


if __name__ == "__main__":
    main()
