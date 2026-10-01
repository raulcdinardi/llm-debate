# Rollout dashboards and live LLM scoring

Implementation: `src/llm_local_rl/{observability,score_sync,live_scoring}.py`.
Commands: `scripts/{build_rollout_workspace,score_rollouts_live,publish_rollout_scores}.py`.
Use the current research checkout `worktrees/paired-shadow-judge` in the Discord
research workspace. The older Desktop checkout does not contain this trainer.

## Dashboard order

Install the optional UI SDK with `pip install -e '.[observability]'`.
From the repository root, create a named view:

```bash
PYTHONPATH=src python -m scripts.build_rollout_workspace \
  --records /path/to/full/step_records.jsonl \
  --entity YOUR_ENTITY --project YOUR_PROJECT --run-id RUN_ID \
  --scoring-config configs/scoring/r2_tests.json \
  --output /path/to/dashboard.json --publish
```

The command writes a reviewable local layout and saves a **new named W&B view**.
Its URL is printed and saved next to the layout. It keeps the existing workspace
views. Ordering is explicit: outcomes, LLM evaluation with coverage, judge
behavior, optimization, and detailed diagnostics. Only metrics found in the
source records get training panels. LLM panels follow the supplied rubric config.
Within each section, graphs follow a semantic priority order, read left to right
and then top to bottom. Outcomes: correctness, reward, speedup, format validity.
Evaluation: headline scores across rounds, then coverage, counts and failures.
Judge: accuracy when available, order consistency, validity, confidence, position
bias, win rates, then detailed distributions. Optimization: loss, KL, clipping,
gradients, learning rate, update count; matching adapter graphs stay adjacent.
Diagnostics: failures and validity, policy checks, entropy/termination, lengths,
reward distributions, advantages, ratios/KL, gradients, runtime and resource
measurements. Unrecognized metrics go after the known groups in stable natural
order. Detailed sections start collapsed; automatic panel creation and alphabetical
sorting are disabled. Missing scores stay missing, never zero. The raw training
metric names remain compatible with existing analyses.

Use `--training-x Step` for historical runs that logged optimizer steps directly
to the SDK's history counter. New runs use `train_step`; evaluation always uses
`evaluation_step`. Do not choose Step for a new run: score/table arrivals also
advance the SDK's internal counter. Historical points do not gain a custom axis
retroactively. The dashboard can be regenerated after new metric names appear.

## Live scoring

The trainer automatically polls `OUTPUT/observability/llm_scoring/*/events.json`
every five seconds. Start the CPU-only scorer in a separate terminal/process,
sharing the output directory. It requires no GPU and does not change rewards,
training RNG, or optimizer behavior. The OpenRouter credential is read exclusively
from `OPENROUTER_API_KEY` in the scorer process; keep it off rented hosts by running
against an append-only local mirror of the rollout file when appropriate.
The mirror must preserve existing bytes and the file inode; ordinary atomic
whole-file replacement is intentionally rejected by the source watcher.

```bash
PYTHONPATH=src python -m scripts.score_rollouts_live \
  --output-dir /path/to/full --config configs/scoring/r2_tests.json \
  --follow --done-file /path/to/producer_completion_marker
```

Supply the producer's real completion marker, created only after its final JSONL
write. With no marker, follow mode stays active until interrupted. Without
`--follow`, drain the records currently available and exit. `--prepare-only`
validates and schedules without making model calls. `--source` selects a mirrored
JSONL file; `--run-id` overrides the local W&B ID file for an external CPU scorer.
For a remote mirror, copy the evaluation directory back to the trainer's expected
path, using atomic file replacement for `events.json` (do not stream a partial JSON
snapshot over the live file). Do not run another W&B writer on that host.

Config freezes the model, provider, rubric, rounds, field ranges, sample size,
seed, parallelism, and retry policy. The included tests-mention profile samples
8 debates per step, both R2 opening arguments, only `python_optimization` tasks;
32 requests run concurrently. `samples_per_step: 0` selects every eligible debate.
Selection is deterministic by run/step/index/seed, independent of text/reward.
`argument_quality.json` is a separate, new generic relevance/specificity rubric
for R2/R3. These examples are **not equivalent to the historical MBPP/CW scoring
rubrics**. Choose a versioned profile for your scientific question; do not mix
scoring versions in one curve. The tests profile measures claims, not actual
correctness. R3 is excluded to avoid counting replies echoing an opening claim.

Every request sees one target turn, the question and (after R1) both candidates.
It does not see later replies, training rewards, winner or training step. This
reader supports debate `sample_records` with `trajectory_a/b` and `r1/r2/...` text.
Absent/empty turns are ineligible, not counted as API failures. Single-turn legacy
schemas need an explicit adapter before use. Task filtering precedes sampling.

Relace is pinned with no provider fallback; requests omit `response_format`.
JSON fields, numeric ranges, completion status and served model/provider are
validated. No token cap is imposed by default; `max_tokens` can be explicitly
configured for a rubric. Transport/parse failures have a persistent attempt limit
(default 3), timeout (120 seconds), and inter-batch pause. Restart does not reset
attempt counts. Source/run/config changes require a new evaluation revision.
Incomplete final JSONL lines wait for the newline. Replaced/truncated sources
fail closed; rollback requires a new evaluation directory. Source files must be
append-only while followed; previously ingested rows are revalidated on restart.

SQLite contains exact requests, raw responses, attempts and normalized results.
`status.json` contains pending/done/failed counts, known billed cost, and the number
of attempts whose cost was unavailable. Unknown cost is not reported as free.
A per-step event is published once all selected requests for that step finish or
exhaust retries. Means use successful turns only: read coverage and sample size
alongside them. Curves are descriptive sample estimates, with no confidence band;
paired speakers are not independent observations. Retain raw results for
clustered uncertainty analysis. Failed jobs require a new evaluation revision for
more attempts. A request interrupted after the provider charges but before the
response is saved can be charged again on retry; exactly-once API billing is not
promised.

## Late scores and saved evaluations

Training and evaluation have separate explicit x-axes. A single W&B writer owns
an OS lock and serializes all scalar/table log calls. Evaluation for step 2 may
arrive after training step 100 without a backwards SDK `step=` call. The trainer
never waits for OpenRouter; on finish it drains already available scores. Scores
that finish afterward remain durable. Once the trainer has written its
`observability/wandb_finished` marker, publish the tail:

```bash
PYTHONPATH=src python -m scripts.publish_rollout_scores \
  --output-dir /path/to/full --entity YOUR_ENTITY --project YOUR_PROJECT
```

The command refuses to run while the trainer owns its writer lock or lacks a
completion marker. Publication receipts prevent ordinary duplicate replay. W&B
has no transaction shared with the local receipt: a crash between log and receipt
may replay an identical point; no exactly-once remote delivery guarantee is made.
The marker is removed on training resume. Preserve the evaluation directory and
receipt together with the training artifacts. Scoring costs are diagnostics only.

`--import-scores saved.jsonl --prepare-only` imports normalized saved results
without inference. Rows must match the scheduled job ID, run, step, sample index,
side, round, source-record/text/prompt hashes, served evaluator and rubric fields;
conflicts abort the transaction. Historical rubric-specific formats require an
explicit converter, not guessed field mappings. `--export-viewer` emits
`manifest.json` and `attempts.jsonl` for the canonical viewer's `normalized`
adapter, binding the final source-file hash. Export only after the source is final;
if it grows afterward, the viewer correctly rejects the stale hash.

Canonical viewer: `/mnt/c/Users/raulc/Desktop/llm-debate/scripts/run_viewer.py`.
Hosting/registry instructions: its `docs/rollout_viewer.md` and the
experiment-lifecycle skill's `references/rollout-viewer.md`. Live W&B evaluation
requires this new observability code in the training launch snapshot; existing
frozen running workers are not modified by installing these scripts.

APIs: [W&B ordered workspaces](https://docs.wandb.ai/models/ref/wandb_workspaces/workspaces),
[custom metric axes](https://docs.wandb.ai/ref/python/experiments/run/),
[OpenRouter provider routing](https://openrouter.ai/docs/guides/routing/provider-selection).

## Native smoothing in one dashboard

Use W&B's native **Running average**, parameter **5**, with **Show Original**
enabled. The raw series appears faintly behind the smoothed series. Change or
disable smoothing in the chart's native settings; no second page is needed.
The dashboard generator defaults to this display, using the original metrics.
W&B's running average is centered (uses points before and after), not strictly
trailing five training steps; sparse metrics use available points. See
[W&B smoothing documentation](https://docs.wandb.ai/models/app/features/panels/line-plot/smoothing).

The earlier Raw/Trailing 5 pair is superseded by this interface preference.
`publish_trailing_average.py` and `--metric-view trailing5` remain explicit legacy
exports for exact past-only analysis, not the default dashboard workflow.

### Judge accuracy and gold ties

Name judge accuracy charts `judge acc (<gold standard>)`; always say what
establishes the preferred answer. MBPP uses **judge acc (picks pass test)**:
`sample_records[].trajectory_{a,b}.task_reward_metrics.correct` supplies the
recorded boolean pass/fail outcome, and `verdict` is the training judge's pick.
Exactly one passing program is eligible. Both passed and both failed are gold
ties and are excluded from accuracy. A recorded execution failure (`correct=false`)
is a failure even when numeric check counts are absent. Missing/nonboolean
correctness is unknown, excluded, and counted separately. An invalid judge pick
on an eligible debate counts as wrong. No eligible debates means absent accuracy,
not zero. These metrics assess recorded tests, not universal program correctness.

`judge_accuracy.py` supplies the per-step numerator, eligible denominator, gold-tie
fraction/count, known-gold denominator, missing labels and invalid picks. Live
`RunObservability` logs these from complete step records under
`judge_eval/picks_pass_test/*` against `train_step`; the ordered workspace puts
accuracy and ties first in Judge behavior. Other environments require their own
explicit gold adapter; never substitute reward, LLM citation labels or speed for
correctness. For mixed environments these denominators include only MBPP pairs.

The main dashboard uses native smoothing of these raw per-step measurements,
with the raw overlay and denominator charts retained. Historical imports use
`judge_eval_step`, never W&B's current internal history step. Existing exact
trailing metrics remain stored but are unnecessary for the native interface.

### Projects containing multiple experiment families

Build category views from verified run identities: CW comparisons (including
baseline, debate, consultancy and pairwise), OpenBookQA/MMLU-Pro warmups,
Mixed4 warmups, and judge SFT. Keep one natively smoothed view per category;
do not duplicate each category into raw and averaged pages. Include existing
LLM evaluation histories with their recorded `optimizer_step` axis, and SFT
validation against `validation_epoch`. Do not chart summary-only evaluations as
if they were per-step histories. Exclude per-example reward-audit scalars and
histogram bins from default charts; retain them in run data. Keep diagnostics
collapsed, with judge accuracy/coherence and scientific outcomes first.

Use explicit run-ID filters so preflights, archive uploads and reports do not
masquerade as full training curves. Old completed runs may write records under
`real_judge`, not `full`: infer neither run category nor initialization from that
suffix alone. Verify the saved filter expression semantically because W&B
normalizes `ID == "id"` to `Metric("ID") == 'id'`.

Run-name proposals should identify Base/Instruct and the initialization for
writer, debater and judge when different. In the historical LFM CW lineage the
R1 writer is fresh, while D/J derive from OpenBookQA-GRPO104. MBPP LFM actors
instead start fresh with that judge retained. CE+JS100, label-GRPO100,
MMLU-Pro100 and Mixed4-r5@100 are distinct parents; importing one adapter is not
an all-adapter continuation. Keep proposed renames separate from approved ones.

## Automatic defaults for every new launch (2026-09-18)

`RunObservability` now publishes an ordered per-run dashboard automatically, off
its training logging path. It observes actual metric keys and saved scoring
configs, so late-arriving or additional-round LLM scores add panels. It uses
native Running average=5 with Show Original. `dashboard_url` and
`dashboard_category_url` are saved in the run summary; local receipts live in
`OUTPUT/observability/dashboard_state.json` and `dashboard_url.txt`.
Category views retain historical IDs and admit new full runs using
`observability_category` / `observability_phase` config filters. Phase0/smokes do
not join full-run comparisons. Publication failures are recorded in
`wandb_failures.jsonl`, with three attempts per outstanding publication; they do
not change optimizer behavior. Offline runs do not publish workspaces.

Before freezing ANY new experiment, including one copied from an older source
snapshot, stage the current logging files and check them:

```bash
python scripts/prepare_observability.py --source-root /path/to/new/source --install
python scripts/prepare_observability.py --source-root /path/to/new/source --runtime
```

Run this from the current research checkout. The installer refuses a frozen
experiment. Save the check's hashes in the launch manifest and repeat `--runtime`
inside the intended runtime before provisioning; dependencies must include
`wandb-workspaces>=0.4.5,<0.5`. A staged old logger is an actionable preparation
error, never evidence that the defaults are deployed. Do not retroactively edit
frozen source or experiment science. For a custom trainer using its own W&B
writer, connect `DashboardSync.observe/publish` and `ScoreInbox.drain` to that
writer; do not start a second writer. Use the actual optimizer/evaluation axes.

Live scoring remains a separate explicit, versioned scoring job. Enabling a
logging dashboard never authorizes paid model requests. The standard worker
writes the existing score inbox and the standard tail publisher now refreshes
the dashboard too. A custom scorer must export verified events to this same
interface; producing a JSONL file alone is not W&B integration.

Dataset-answer pairs with explicit `gold_agent` and consistent `is_correct`
labels now log `judge_eval/picks_gold_answer/*`. This is the final rollout
judge's verdict accuracy, distinct from `train/judge/supervised_label_accuracy`
which measures both displayed orders in the training forward pass. Both titles
state their gold standard. Gold ties are excluded from accuracy, missing or
contradictory labels are counted, and empty denominators produce no accuracy.

After packaging, also run `python scripts/prepare_observability.py
--source-archive /path/to/source.tar.gz` and retain its receipt before freeze.
Checking the directory alone is insufficient: `git archive` can omit new files
that exist locally but have not been committed. This archive check reads member
bytes without extracting files and verifies the exact logging modules consumed.
