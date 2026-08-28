ARG VLLM_IMAGE=vllm/vllm-openai:v0.26.0
FROM ${VLLM_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV HF_HOME=/cache/huggingface
ENV TRANSFORMERS_CACHE=/cache/huggingface
ENV VLLM_CACHE_ROOT=/cache/vllm
ENV PYTHONPATH=/workspace/llm-local-rl/src
ENV SKIP_PYTHON_DEPS=1

WORKDIR /workspace/llm-local-rl

RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install \
      "accelerate==1.14.0" \
      "peft==0.20.0" \
      "pytest==9.1.1" \
      "safetensors==0.8.0" \
      "tokenizers==0.22.2" \
      "transformers==5.14.1" \
      "wandb>=0.19.10,<0.24"

COPY pyproject.toml README.md ./
COPY src ./src
COPY prompts ./prompts
COPY scripts ./scripts
COPY docker ./docker

RUN python3 -m pip install -e .

RUN chmod +x docker/*.sh && \
    cp docker/entrypoint.sh /usr/local/bin/entrypoint.sh && \
    chmod +x /usr/local/bin/entrypoint.sh && \
    ln -sf /usr/local/bin/entrypoint.sh /entrypoint.sh

ENTRYPOINT ["entrypoint.sh"]
