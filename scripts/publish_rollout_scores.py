"""Drain saved scores into an ended training run; never share its active writer."""
import argparse
from pathlib import Path
from llm_local_rl.live_scoring import exclusive
from llm_local_rl.score_sync import ScoreInbox, define_axes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--entity", required=True)
    p.add_argument("--project", required=True)
    args = p.parse_args()
    state = args.output_dir / "observability"
    with exclusive(state / "wandb_writer.lock"):
        if not (state / "wandb_finished").exists():
            raise SystemExit("Training has not finished; the trainer owns live score ingestion")
        import wandb
        run_id = (state / "wandb_run_id.txt").read_text().strip()
        run = wandb.init(entity=args.entity, project=args.project, id=run_id, resume="must", dir=str(state))
        try:
            define_axes(run)
            from llm_local_rl.dashboard_sync import DashboardSync, metadata
            meta = metadata(dict(run.config), args.output_dir)
            run.config.update(meta)
            dashboard = DashboardSync(state, run, meta)
            count = ScoreInbox(state, run_id).drain(run, on_metrics=dashboard.observe)
            dashboard.publish(force=True)
            print(f"Published {count} evaluation events")
        finally:
            run.finish()


if __name__ == "__main__":
    main()
