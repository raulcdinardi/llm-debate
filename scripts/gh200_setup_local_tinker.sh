#!/usr/bin/env bash
set -euo pipefail

# Assumptions:
# - Run from the repo root (so relative paths work).
# - You have a working CUDA-enabled torch install available (pip/conda/modules).
# - Network access is available for pip + HuggingFace downloads.

python3 -m venv --system-site-packages .venv-gh200
source .venv-gh200/bin/activate

python3 -m pip install --upgrade pip wheel

# Base deps for this repo's scripts (fail-fast).
python3 -m pip install python-dotenv tqdm rich vcrpy

# Runtime deps for local tinker backend (imported via PYTHONPATH wrapper).
python3 -m pip install "transformers>=4.38" "peft>=0.10" "accelerate>=0.27" "huggingface_hub>=0.20"

# System-site-packages environments often include old PIL / numpy / networkx which can break
# modern Transformers imports. Install compatible wheels into the venv to shadow system packages.
python3 -m pip install --upgrade "pillow>=10" "numpy<2,>=1.26.0" "networkx>=3"

echo
echo "Next:"
echo "  export USE_TF=0  # avoid Transformers importing system TensorFlow/Keras"
echo "  export TRANSFORMERS_NO_TF=1"
echo "  export TRANSFORMERS_NO_FLAX=1"
echo "  export TINKER_LOCAL_BACKEND=transformers"
echo "  python3 scripts/download_hf_model.py --repo-id Qwen/Qwen3-4B-Instruct-2507"
echo "  export TINKER_DEBATE_BASE_MODEL=\"$(pwd)/models/Qwen__Qwen3-4B-Instruct-2507\""
echo "  chmod +x tinker-local/bin/with_local_tinker"
echo "  tinker-local/bin/with_local_tinker python3 scripts/train.py --no-proxy --dry-run -n 2 -s 1"
