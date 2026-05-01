#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -gt 0 ]]; then
  exec "$@"
fi

if [[ -f /workspace/llm-local-rl/docker/vast_onstart_ht_debate.sh ]]; then
  exec bash /workspace/llm-local-rl/docker/vast_onstart_ht_debate.sh
fi

echo "entrypoint.sh could not find /workspace/llm-local-rl/docker/vast_onstart_ht_debate.sh" >&2
exit 2
