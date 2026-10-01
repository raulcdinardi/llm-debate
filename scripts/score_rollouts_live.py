"""Run with PYTHONPATH=src python -m scripts.score_rollouts_live --help."""
import argparse
import json
import os
from pathlib import Path
import time

from llm_local_rl.live_scoring import Scorer, exclusive, validate_config


def main():
    parser = argparse.ArgumentParser(description="Parallel OpenRouter scoring, independent of training")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", help="Defaults to the output directory's saved W&B ID")
    parser.add_argument("--source", type=Path, help="Defaults to OUTPUT/step_records.jsonl")
    parser.add_argument("--follow", action="store_true", help="Keep following until --done-file exists and queued work finishes")
    parser.add_argument("--done-file", type=Path, help="Explicit producer-completion marker; without it Ctrl-C stops follow mode")
    parser.add_argument("--poll-seconds", type=float, default=5)
    parser.add_argument("--prepare-only", action="store_true", help="Validate/populate jobs without calling OpenRouter")
    parser.add_argument("--import-scores", type=Path, help="Import exact, normalized saved scores before processing")
    parser.add_argument("--export-viewer", action="store_true", help="Export canonical viewer scores when the source is final")
    args = parser.parse_args()
    if args.poll_seconds <= 0:
        parser.error("--poll-seconds must be positive")
    state = args.output_dir / "observability"
    run_id = args.run_id or (state / "wandb_run_id.txt").read_text().strip()
    config = json.loads(args.config.read_text())
    validate_config(config)
    directory = state / "llm_scoring" / config["name"]
    directory.mkdir(parents=True, exist_ok=True)
    source = args.source or args.output_dir / "step_records.jsonl"
    if not args.follow and not source.is_file():
        parser.error(f"Rollout source does not exist: {source}")
    with exclusive(directory / "worker.lock"):
        scorer = Scorer(source, directory, config, run_id)
        try:
            if args.import_scores:
                scorer.import_scores(args.import_scores)
            while True:
                scorer.ingest()
                pending_before = scorer.db.execute("SELECT count(*) FROM jobs WHERE status='pending'").fetchone()[0]
                if pending_before and not args.prepare_only and not os.environ.get("OPENROUTER_API_KEY"):
                    raise SystemExit("Set OPENROUTER_API_KEY in the scorer process; no attempts were consumed")
                worked = 0 if args.prepare_only else scorer.score_batch()
                scorer.export()
                pending = scorer.db.execute("SELECT count(*) FROM jobs WHERE status='pending'").fetchone()[0]
                if args.prepare_only or (not pending and (not args.follow or (args.done_file and args.done_file.exists()))):
                    break
                # Backoff between batches prevents a persistent provider error spinning.
                time.sleep(args.poll_seconds if not worked else config.get("retry_pause_seconds", 1))
            if args.export_viewer:
                scorer.export_viewer()
            print(json.dumps({"directory": str(directory), "status": json.loads((directory / "status.json").read_text())}))
        finally:
            scorer.db.close()


if __name__ == "__main__":
    main()
