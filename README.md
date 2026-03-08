# tinker-debate

Debate training using the Tinker API. Runs debates between two agents, judges outcomes, and trains on winners (rejection sampling).

## Setup

```bash
pip install -e .
```

## HTTP Traffic Inspection

### VCR.py (automatic recording)

All HTTP traffic is automatically recorded to `logs/<run>/http_traffic.yaml`. These YAML cassettes are human-readable and can be replayed for testing.

### mitmproxy (live inspection)

For live HTTP traffic inspection, use the `--proxy` flag:

```bash
python scripts/train.py --proxy --name proxy_smoke
```

This launches mitmweb and opens a web UI at http://127.0.0.1:8081.

#### mitmproxy CA certificate setup

mitmproxy intercepts HTTPS by presenting its own certificates. For this to work, you must add mitmproxy's CA to your system trust store:

```bash
# mitmproxy generates its CA on first run
mitmweb --help >/dev/null 2>&1

# Add to system trust store (Ubuntu/Debian)
sudo cp ~/.mitmproxy/mitmproxy-ca-cert.pem /usr/local/share/ca-certificates/mitmproxy.crt
sudo update-ca-certificates
```

Without this step, you'll get `SSL: CERTIFICATE_VERIFY_FAILED` errors because httpx (used by Tinker internally) verifies certificates against the system CA store.

## Usage

```bash
# Run 2 debates, 1 training step
python scripts/train.py --name smoke_debate

# Run 4 debates, 2 training steps
python scripts/train.py -n 4 -s 2 --name smoke_debate_n4_s2

# Dry run (debates only, no training)
python scripts/train.py --dry-run --name dryrun_debate

# Use GPQA dataset
python scripts/train.py --dataset gpqa_diamond --name gpqa_smoke

# Live HTTP inspection
python scripts/train.py --proxy --name proxy_smoke

# Adjust sampling parameters
python scripts/train.py --temperature 1.2 --max-tokens 512 --name temp12_mt512
```

## Local (Transformers) backend

This repo can also run against `tinker-local/` (a local, fail-fast subset of the Tinker SDK) without changing the training code. The local backend supports transformers only.

Assumptions:
- You run from the repo root so relative paths stay stable.
- You use `tinker-local/bin/with_local_tinker`, which sets `TINKER_BACKEND=local` per-run and keeps real `import tinker` unshadowed.

Example (local model directory):

```bash
export TINKER_LOCAL_BACKEND=transformers
export TINKER_DEBATE_BASE_MODEL=/absolute/path/to/model_dir
chmod +x tinker-local/bin/with_local_tinker
tinker-local/bin/with_local_tinker python3 scripts/train.py --dry-run -n 2 -s 1 --name local_smoke
```

To download a model snapshot into `./models/`:

```bash
python3 scripts/download_hf_model.py --repo-id Qwen/Qwen3-4B-Instruct-2507
export TINKER_DEBATE_BASE_MODEL="$(pwd)/models/Qwen__Qwen3-4B-Instruct-2507"
```

Example (LiquidAI LFM2.5 1.2B Instruct, CPU):

```bash
python3 scripts/download_hf_model.py --repo-id LiquidAI/LFM2.5-1.2B-Instruct
export TINKER_DEBATE_BASE_MODEL="$(pwd)/models/LiquidAI__LFM2.5-1.2B-Instruct"
export TINKER_LOCAL_DEVICE=cpu
tinker-local/bin/with_local_tinker python3 scripts/train.py --dry-run -n 2 -s 1 --max-tokens 16 --name lfm25_local_smoke
```

## Logs

Run directories are created as:
- `logs/<timestamp>_n<num_rollouts>_g<num_groups>_<mode>_<env>_<dataset_or_custom>_<name>/`

Each run contains:
- `run_metadata.json` - CLI args, selected env vars, and package versions
- `debate_*.json` - debate rollouts (debate mode)
- `baseline_*.json` - single-turn rollout logs
- `summary_*.json` - single-turn summary rollout logs
- `training_step_*.json` - training step logs
- `http_traffic.yaml` - VCR cassette of all HTTP requests/responses (API mode only; skipped in local mode)

## Results Search Hub

To build a searchable index over `logs_remote/` and `results/`:

```bash
python3 scripts/build_results_search_hub.py
```

This creates `results/search_hub/` with:
- `catalog/runs.csv` and `catalog/runs.jsonl`
- `by_sid/<sid>/runs` and `by_sid/<sid>/results`
- `by_env/<env>/<mode>`
- `by_judge/{real,mock,confidence,na}`
- `by_rule_family/<family>`

## Notebook Visualizer

`notebooks/run_visualizer.ipynb` uses lazy step indexing and on-demand sample loading:
- it supports baseline, debate, and summary runs
- it avoids eager loading all rollout JSON files at notebook open
- it caches per-run step indexes in `.run_visualizer_step_index.json` within each run directory

## Drive Merge Archive

A Drive snapshot was merged into this repo on 2026-03-08.

- Comparison notes: `docs/drive_snapshot_vs_local_worktree.md`
- Archived Drive decisions log: `docs/archive/drive_snapshot_decisions_20260210.md`
- Archived Drive artifact manifest: `docs/archive/drive_snapshot_logs_remote_manifest_20260308.md`
- Preserved smaller Drive checkpoint bundles: `artifacts/drive_snapshot_20260210/`
