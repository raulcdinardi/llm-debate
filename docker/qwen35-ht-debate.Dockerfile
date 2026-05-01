ARG VLLM_IMAGE=vllm/vllm-openai:latest
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
      "transformers>=4.57.0" \
      "accelerate>=1.10.0" \
      "peft>=0.17.0" \
      "pytest>=8.0"

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
