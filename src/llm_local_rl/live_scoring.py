"""CPU-only, resumable OpenRouter evaluation of committed rollout JSONL rows.

The trainer never imports an API client or waits for these jobs. SQLite commits
bind source rows, rubric and requests; W&B consumes finalized per-step events.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import time
from urllib.request import Request, urlopen


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


@contextmanager
def exclusive(path):
    """OS-owned lock: released on crash, never a stale PID/timestamp lease."""
    with Path(path).open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def validate_config(config):
    for key in ("name", "model", "provider", "system_prompt", "fields", "rounds"):
        if not config.get(key):
            raise ValueError(f"Missing scoring config: {key}")
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", config["name"]):
        raise ValueError("name must be a metric-safe identifier")
    for key, limits in config["fields"].items():
        if not re.fullmatch(r"[a-zA-Z0-9_]+", key) or len(limits) != 2 or not all(
            isinstance(v, (int, float)) and math.isfinite(v) for v in limits
        ) or limits[0] > limits[1]:
            raise ValueError(f"Invalid field bounds: {key}")
    if len(set(config["rounds"])) != len(config["rounds"]) or any(
        not re.fullmatch(r"r[1-9][0-9]*", r) for r in config["rounds"]
    ):
        raise ValueError("rounds must be unique r1, r2, ... identifiers")
    for key, default in (("concurrency", 16), ("max_attempts", 3), ("every_steps", 1)):
        if type(config.get(key, default)) is not int or config.get(key, default) < 1:
            raise ValueError(f"{key} must be a positive integer")
    if type(config.get("samples_per_step", 8)) is not int or config.get("samples_per_step", 8) < 0:
        raise ValueError("samples_per_step must be nonnegative (0 means all)")


def validate_scores(scores, fields):
    if not isinstance(scores, dict) or set(scores) != set(fields):
        raise ValueError("Score fields do not match rubric")
    result = {}
    for key, bounds in fields.items():
        value = scores[key]
        if not isinstance(value, (int, float, bool)) or not math.isfinite(value) or not bounds[0] <= value <= bounds[1]:
            raise ValueError(f"Invalid score: {key}")
        result[key] = float(value)
    return result


def openrouter(request, config):
    body = {"model": config["model"], "messages": request["messages"],
            "temperature": 0, "provider": {"only": [config["provider"]],
            "allow_fallbacks": False, "require_parameters": True}}
    if "reasoning" in config:
        body["reasoning"] = config["reasoning"]
    if "max_tokens" in config:
        body["max_tokens"] = config["max_tokens"]
    # Relace does not support response_format: validate prompt-requested JSON locally.
    req = Request("https://openrouter.ai/api/v1/chat/completions", data=canonical(body).encode(),
                  headers={"Authorization": "Bearer " + os.environ["OPENROUTER_API_KEY"],
                           "Content-Type": "application/json"})
    with urlopen(req, timeout=config.get("timeout_seconds", 120)) as response:
        return json.load(response)


class Scorer:
    def __init__(self, source, directory, config, run_id):
        validate_config(config)
        self.source, self.directory = Path(source).resolve(), Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.config, self.run_id = config, run_id
        self.db = sqlite3.connect(self.directory / "scores.sqlite")
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript("""
          CREATE TABLE IF NOT EXISTS cursor (bytes INTEGER NOT NULL);
          CREATE TABLE IF NOT EXISTS binding (value TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS steps (step INTEGER PRIMARY KEY, hash TEXT, population INTEGER);
          CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, step INTEGER, request TEXT,
            attempts INTEGER DEFAULT 0, status TEXT DEFAULT 'pending', result TEXT);
          CREATE TABLE IF NOT EXISTS attempts (job TEXT, number INTEGER, response TEXT, error TEXT,
            PRIMARY KEY(job,number));
        """)
        binding = canonical({"schema": 1, "source": str(self.source), "config": config, "run_id": run_id})
        prior = self.db.execute("SELECT value FROM binding").fetchone()
        if prior and prior[0] != binding:
            raise ValueError("Source/run/rubric changed; use a new evaluation directory")
        if not prior:
            self.db.execute("INSERT INTO binding VALUES (?)", (binding,))
            self.db.commit()
        # A killed in-flight attempt still counts toward the aggregate retry limit.
        self.db.execute("UPDATE jobs SET status='failed' WHERE status='pending' AND attempts>=?",
                        (config.get("max_attempts", 3),))
        self.db.commit()
        self.persisted_bytes = (self.db.execute("SELECT bytes FROM cursor").fetchone() or (0,))[0]
        (self.directory / "config.json").write_text(canonical(config))
        self.offset = 0
        self.source_inode = None

    def ingest(self):
        if not self.source.exists():
            return
        stat = self.source.stat()
        identity = (stat.st_dev, stat.st_ino)
        if (self.source_inode is not None and identity != self.source_inode) or stat.st_size < max(self.offset, self.persisted_bytes):
            raise ValueError("Rollout source replaced/truncated; start a new evaluation revision")
        self.source_inode = identity
        with self.source.open("rb") as handle:
            handle.seek(self.offset)
            while True:
                line = handle.readline()
                if not line or not line.endswith(b"\n"):
                    break
                record = json.loads(line)
                step = int(record["step"])
                row_hash = digest(record)
                prior = self.db.execute("SELECT hash FROM steps WHERE step=?", (step,)).fetchone()
                if prior and prior[0] != row_hash:
                    raise ValueError(f"Rollout content changed at step {step}")
                if not prior:
                    samples = record.get("sample_records", [])
                    with self.db:
                        self.db.execute("INSERT INTO steps VALUES (?,?,?)", (step, row_hash, len(samples)))
                        if step > 0 and (step - 1) % self.config.get("every_steps", 1) == 0:
                            indices = sorted(range(len(samples)), key=lambda i: digest([self.run_id, step, i, self.config.get("seed", 0)]))
                            limit = self.config.get("samples_per_step", 8)
                            if self.config.get("tasks"):
                                indices = [i for i in indices if samples[i].get("metrics", {}).get("task") in self.config["tasks"]]
                            for index in indices[:limit or None]:
                                for request in self.requests(step, index, samples[index], row_hash):
                                    self.db.execute("INSERT OR IGNORE INTO jobs(id,step,request) VALUES (?,?,?)",
                                                    (request["id"], step, canonical(request)))
                self.offset = handle.tell()
                with self.db:
                    self.db.execute("DELETE FROM cursor")
                    self.db.execute("INSERT INTO cursor VALUES (?)", (self.offset,))

    def requests(self, step, index, sample, row_hash):
        for side in ("A", "B"):
            trajectory = sample.get("trajectory_" + side.lower())
            if not isinstance(trajectory, dict):
                continue
            for channel in self.config["rounds"]:
                text = trajectory.get(channel)
                if not isinstance(text, str) or not text.strip():
                    continue
                # Score exactly one turn. Later replies and training rewards are absent.
                context = {"question": sample.get("question", ""), "target_side": side,
                           "round": channel, "target_text": text}
                if channel != "r1":
                    context["program_or_answer_a"] = sample.get("trajectory_a", {}).get("r1", "")
                    context["program_or_answer_b"] = sample.get("trajectory_b", {}).get("r1", "")
                request = {"run_id": self.run_id, "step": step, "debate_index": index,
                           "side": side, "channel": channel, "record_sha256": row_hash,
                           "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                           "messages": [{"role": "system", "content": self.config["system_prompt"]},
                                        {"role": "user", "content": canonical(context)}]}
                request["prompt_sha256"] = digest(request["messages"])
                request["id"] = digest(request)
                yield request

    def score_batch(self, client=openrouter):
        rows = self.db.execute("SELECT id,request,attempts FROM jobs WHERE status='pending' ORDER BY step,id LIMIT ?",
                               (self.config.get("concurrency", 16),)).fetchall()
        if not rows:
            return 0
        with self.db:
            for job, _, count in rows:
                self.db.execute("UPDATE jobs SET attempts=? WHERE id=?", (count + 1, job))
        with ThreadPoolExecutor(max_workers=self.config.get("concurrency", 16)) as pool:
            futures = {pool.submit(client, json.loads(request), self.config): (job, request, count + 1)
                       for job, request, count in rows}
            for future in as_completed(futures):
                job, request, count = futures[future]
                response, error, scores = None, None, None
                try:
                    response = future.result()
                    if not isinstance(response, dict):
                        raise ValueError("Expected an OpenRouter response object")
                    if response.get("model") != self.config["model"] or response.get("provider", "").casefold() != self.config.get("served_provider", self.config["provider"]).casefold():
                        raise ValueError("Served model/provider mismatch")
                    choice = response["choices"][0]
                    if choice.get("finish_reason") != "stop":
                        raise ValueError("Incomplete model response")
                    content = choice["message"]["content"].strip()
                    if content.startswith("```json\n") and content.endswith("```"):
                        content = content[8:-3].strip()
                    scores = validate_scores(json.loads(content), self.config["fields"])
                except Exception as exc:
                    # Never persist HTTP headers, credentials or arbitrary exception bodies.
                    error = type(exc).__name__
                status = "done" if scores is not None else ("failed" if count >= self.config.get("max_attempts", 3) else "pending")
                metadata = response if isinstance(response, dict) else {}
                result = json.loads(request) | {"scores": scores, "error": error,
                    "served_model": metadata.get("model"), "served_provider": metadata.get("provider"),
                    "usage": metadata.get("usage") or {}}
                result.pop("messages")
                with self.db:
                    self.db.execute("INSERT INTO attempts VALUES (?,?,?,?)", (job, count, canonical(response), error))
                    self.db.execute("UPDATE jobs SET status=?,result=? WHERE id=?", (status, canonical(result), job))
        return len(rows)

    def export(self):
        """Atomic snapshots let the single W&B writer recover without lost events."""
        events = []
        for step, population in self.db.execute("SELECT step,population FROM steps ORDER BY step"):
            rows = self.db.execute("SELECT status,result,request FROM jobs WHERE step=?", (step,)).fetchall()
            if not rows or any(status == "pending" for status, _, _ in rows):
                continue
            for channel in self.config["rounds"]:
                group = [(s, json.loads(r) if r else {}, json.loads(q)) for s, r, q in rows if json.loads(q)["channel"] == channel]
                if not group:
                    continue
                good = [r for s, r, _ in group if s == "done"]
                prefix = f"llm_eval/{self.config['name']}/{channel}"
                metrics = {prefix + "/scored": len(good), prefix + "/eligible": len(group),
                           prefix + "/failed": len(group) - len(good),
                           prefix + "/coverage": len(good) / len(group), prefix + "/population_debates": population}
                for field in self.config["fields"]:
                    if good:
                        metrics[prefix + "/" + field] = sum(r["scores"][field] for r in good) / len(good)
                event = {"step": step, "metrics": metrics, "run_id": self.run_id,
                         "config_sha256": digest(self.config)}
                event["id"] = digest(event)
                events.append(event)
        status = dict(self.db.execute("SELECT status,count(*) FROM jobs GROUP BY status"))
        costs = []
        for (raw,) in self.db.execute("SELECT response FROM attempts"):
            response = json.loads(raw)
            usage = response.get("usage") if isinstance(response, dict) else None
            cost = usage.get("cost") if isinstance(usage, dict) else None
            if isinstance(cost, (int, float)) and math.isfinite(cost):
                costs.append(cost)
        reserved_attempts = self.db.execute("SELECT COALESCE(sum(attempts),0) FROM jobs").fetchone()[0]
        unknown_cost = reserved_attempts - len(costs)
        status.update({"recorded_cost_usd": sum(costs), "attempts_without_cost": unknown_cost})
        for name, value in (("events.json", events), ("status.json", status)):
            temporary = self.directory / (name + ".tmp")
            temporary.write_text(canonical(value))
            os.replace(temporary, self.directory / name)
        return events

    def import_scores(self, path):
        """Import normalized rows only with exact prompt and source-row bindings.

        No fuzzy joins or guessed rubric equivalence. A converter for historical
        rubrics must explicitly supply the matching config and exact identities.
        """
        self.ingest()
        count = 0
        with self.db:
            for line in Path(path).read_text().splitlines():
                row = json.loads(line)
                job = self.db.execute("SELECT request,status FROM jobs WHERE id=?", (row["id"],)).fetchone()
                if job is None:
                    raise ValueError("Imported score does not match a scheduled job")
                request = json.loads(job[0])
                for key in ("run_id", "step", "debate_index", "side", "channel", "text_sha256", "record_sha256", "prompt_sha256"):
                    if row.get(key) != request[key]:
                        raise ValueError(f"Imported score identity mismatch: {key}")
                if row.get("served_model") != self.config["model"] or row.get("served_provider", "").casefold() != self.config.get("served_provider", self.config["provider"]).casefold():
                    raise ValueError("Imported evaluator differs from configured evaluator")
                row["scores"] = validate_scores(row.get("scores"), self.config["fields"])
                if row.get("error"):
                    raise ValueError("Cannot import a failed score as successful")
                if job[1] == "done":
                    prior = json.loads(self.db.execute("SELECT result FROM jobs WHERE id=?", (row["id"],)).fetchone()[0])
                    if prior["scores"] != row["scores"]:
                        raise ValueError("Imported score conflicts with saved successful score")
                    continue
                self.db.execute("UPDATE jobs SET status='done',result=? WHERE id=?", (canonical(row), row["id"]))
                count += 1
        return count

    def export_viewer(self):
        """Call after the source stops growing; binds the final file for the viewer."""
        self.ingest()
        h = hashlib.sha256()
        with self.source.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                h.update(chunk)
        manifest = {"run_id": self.run_id, "model": self.config["model"], "records_sha256": h.hexdigest(),
                    "fields": {r: list(self.config["fields"]) for r in self.config["rounds"]},
                    "rubrics": {r: self.config["system_prompt"] for r in self.config["rounds"]}}
        (self.directory / "manifest.json").write_text(canonical(manifest))
        with (self.directory / "attempts.jsonl").open("w") as handle:
            for (result,) in self.db.execute("SELECT result FROM jobs WHERE result IS NOT NULL ORDER BY step,id"):
                handle.write(result + "\n")
