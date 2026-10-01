"""Add a Raw / Trailing 5 view pair for one completed W&B run.

Read the actual run history without sampling; preserve raw metrics and view
order. A completed-run guard prevents a second logger writing alongside training.
"""
import argparse
import hashlib
import json
from pathlib import Path

from llm_local_rl.live_scoring import exclusive
from llm_local_rl.metric_averaging import trailing_rows


def key(value):
    return value if isinstance(value, str) else value.name


def source_rows(run, view):
    axes = {}
    for section in view.sections:
        for panel in section.panels:
            if len(panel.y) != 1:
                raise ValueError("Expected one raw metric per panel")
            axis = key(panel.x)
            axis = "_step" if axis == "Step" else axis
            axes.setdefault(axis, set()).add(key(panel.y[0]))
    by_step = {}
    # Request each axis separately; keys=[all_metrics] would drop sparse rows.
    for axis, metrics in axes.items():
        for row in run.scan_history(page_size=1000):
            if row.get(axis) is None or row.get("rollout/phase0_observability_probe") == 1:
                continue
            step = row[axis]
            if isinstance(step, bool) or not isinstance(step, (float, int)) or not float(step).is_integer():
                raise ValueError(f"Invalid source step: {step}")
            point = by_step.setdefault(int(step), {})
            for metric in metrics:
                if metric in row and row[metric] is not None:
                    if metric in point and point[metric] != row[metric]:
                        raise ValueError(f"Conflicting measurements at step {step}: {metric}")
                    point[metric] = row[metric]
    return [{"step": step, "metrics": values} for step, values in sorted(by_step.items()) if values]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace-url", required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    import wandb
    import wandb_workspaces.workspaces as ws
    with exclusive(args.output_dir / "publish.lock"):
        view = ws.Workspace.from_url(args.workspace_url)
        run_path = f"{view.entity}/{view.project}/{args.run_id}"
        remote = wandb.Api(timeout=30).run(run_path)
        if remote.state != "finished":
            raise SystemExit(f"Refusing concurrent writer: run is {remote.state}, expected finished")
        rows = source_rows(remote, view)
        if not rows:
            raise ValueError("No raw scalar history available")
        averaged = list(trailing_rows(rows))
        serialized = json.dumps(averaged, sort_keys=True, allow_nan=False)
        fingerprint = hashlib.sha256((run_path + serialized).encode()).hexdigest()
        receipt_path = args.output_dir / "published.json"
        receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {"fingerprint": fingerprint, "published_steps": []}
        if receipt["fingerprint"] != fingerprint:
            raise ValueError("Source data changed; use a new output directory/review the new snapshot")
        (args.output_dir / "source.json").write_text(json.dumps(rows, sort_keys=True))
        (args.output_dir / "trailing5.json").write_text(serialized)
        seen = set(receipt["published_steps"])
        pending = [r for r in averaged if r["averaging_step"] not in seen]
        if pending:
            run = wandb.init(entity=view.entity, project=view.project, id=args.run_id,
                             resume="must", dir=str(args.output_dir))
            try:
                run.define_metric("averaging_step", hidden=True)
                run.define_metric("trailing5/*", step_metric="averaging_step", step_sync=False)
                for row in pending:
                    run.log(row, commit=True)
                    receipt["published_steps"].append(row["averaging_step"])
                    temp = receipt_path.with_suffix(".tmp")
                    temp.write_text(json.dumps(receipt))
                    temp.replace(receipt_path)
            finally:
                run.finish()
        # The view selector is the native toggle. Both retain identical order.
        raw_name = view.name.removesuffix(" · Raw")
        view.name = raw_name + " · Raw"
        view.save()
        url_path = args.output_dir / "trailing_view_url.txt"
        smooth = ws.Workspace.from_url(url_path.read_text().strip()) if url_path.exists() else ws.Workspace.from_url(args.workspace_url)
        smooth.name = raw_name + " · Trailing 5"
        smooth.runset_settings.filters = f'ID == "{args.run_id}"'
        smooth.runset_settings.pinned_runs = [args.run_id]
        for section in smooth.sections:
            for panel in section.panels:
                raw = key(panel.y[0]).removeprefix("trailing5/")
                panel.y = ["trailing5/" + raw]
                panel.x = "averaging_step"
                panel.smoothing_type = "none"
                panel.title = panel.title.removesuffix(" · trailing 5") + " · trailing 5"
        if url_path.exists():
            smooth.save()
        else:
            smooth.save_as_new_view()
        url_path.write_text(smooth.url + "\n")
        checked = ws.Workspace.from_url(smooth.url)
        expected = [[key(p.y[0]) for p in s.panels] for s in smooth.sections]
        assert [[key(p.y[0]) for p in s.panels] for s in checked.sections] == expected
        assert all(key(p.x) == "averaging_step" and p.smoothing_type == "none"
                   for section in checked.sections for p in section.panels)
        fresh = wandb.Api(timeout=30).run(run_path)
        actual = {r["averaging_step"]: r for r in fresh.scan_history(page_size=1000) if "averaging_step" in r}
        for row in averaged:
            for metric, value in row.items():
                assert abs(actual[row["averaging_step"]][metric] - value) <= 1e-6 * max(1, abs(value)), metric
        result = {"raw_url": view.url, "trailing_url": checked.url, "steps": len(averaged),
                  "panels": sum(len(s.panels) for s in checked.sections), "verified": True}
        (args.output_dir / "verification.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result))


if __name__ == "__main__":
    main()
