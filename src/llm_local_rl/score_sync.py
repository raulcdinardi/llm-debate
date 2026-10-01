"""Single-owner W&B logging for late, source-step-aligned evaluation events."""
import json
from pathlib import Path


def define_axes(run):
    run.define_metric("train_step", hidden=True)
    run.define_metric("evaluation_step", hidden=True)
    for prefix in ("rollout/*", "reward_component/*", "train/*", "rollouts/*", "judge_eval/*"):
        run.define_metric(prefix, step_metric="train_step", step_sync=False)
    run.define_metric("llm_eval/*", step_metric="evaluation_step", step_sync=False)


class ScoreInbox:
    def __init__(self, state_dir, run_id):
        self.state_dir, self.run_id = Path(state_dir), run_id
        self.receipt = self.state_dir / "llm_scores_published.json"
        value = json.loads(self.receipt.read_text()) if self.receipt.exists() else {}
        if value and value["run_id"] != run_id:
            raise ValueError("Score receipt belongs to another run")
        self.seen = set(value.get("ids", []))

    def drain(self, run, on_metrics=None):
        from llm_local_rl.live_scoring import digest
        count = 0
        for path in sorted((self.state_dir / "llm_scoring").glob("*/events.json")):
            for event in json.loads(path.read_text()):
                if event["run_id"] != self.run_id:
                    raise ValueError(f"Evaluation belongs to another run: {path}")
                expected = digest({k: v for k, v in event.items() if k != "id"})
                if event["id"] != expected:
                    raise ValueError(f"Evaluation event checksum mismatch: {path}")
                if event["id"] in self.seen:
                    continue
                if not event["metrics"] or any(not k.startswith("llm_eval/") for k in event["metrics"]):
                    raise ValueError("Evaluation event has invalid metric namespace")
                # SDK history counter is automatic. Never pass an old step= here.
                run.log({"evaluation_step": int(event["step"]), **event["metrics"]}, commit=True)
                if on_metrics is not None:
                    on_metrics(event["metrics"])
                self.seen.add(event["id"])
                temporary = self.receipt.with_suffix(".tmp")
                temporary.write_text(json.dumps({"run_id": self.run_id, "ids": sorted(self.seen)}))
                temporary.replace(self.receipt)
                count += 1
        return count
