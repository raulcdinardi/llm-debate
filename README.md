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
```

## Logs

- `logs/test_runs/<timestamp>/` - default location for test runs
- `logs/gpqa_runs/<dataset>/<timestamp>/` - GPQA dataset runs
- `logs/experiments/<name>/` - named experiments (`-e <name>`)

Each run contains:
- `debate_*.json` - individual debate logs with full token/logprob data
- `training_step_*.json` - training step logs
- `http_traffic.yaml` - VCR cassette of all HTTP requests/responses
