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
python scripts/train.py --proxy
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
python scripts/train.py

# Run 4 debates, 2 training steps
python scripts/train.py -n 4 -s 2

# Dry run (debates only, no training)
python scripts/train.py --dry-run

# Use GPQA dataset
python scripts/train.py --dataset gpqa_diamond

# Live HTTP inspection
python scripts/train.py --proxy

# Adjust sampling parameters
python scripts/train.py --temperature 1.2 --max-tokens 512
```

## Local (Transformers) backend

This repo can also run against `tinker-local/` (a local, fail-fast subset of the Tinker SDK) without changing the training code. The local backend supports transformers only.

Assumptions:
- You run from the repo root so relative paths stay stable.
- You use `tinker-local/bin/with_local_tinker` to shadow `import tinker` per-run (so you can still use the real SDK in other runs).

Example (local model directory):

```bash
export TINKER_LOCAL_BACKEND=transformers
export TINKER_DEBATE_BASE_MODEL=/absolute/path/to/model_dir
chmod +x tinker-local/bin/with_local_tinker
tinker-local/bin/with_local_tinker python3 scripts/train.py --dry-run -n 2 -s 1
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

- `logs/test_runs/<timestamp>/` - default location for test runs
- `logs/gpqa_runs/<dataset>/<timestamp>/` - GPQA dataset runs
- `logs/experiments/<name>/` - named experiments (`-e <name>`)

Each run contains:
- `debate_*.json` - individual debate logs with full token/logprob data
- `training_step_*.json` - training step logs
- `http_traffic.yaml` - VCR cassette of all HTTP requests/responses
