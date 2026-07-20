ARG SGLANG_IMAGE=lmsysorg/sglang@sha256:a18a0642ca71a750c8b2ed99d3d0cd58761f67a52119674ab05d23dd3faef5f8
FROM ${SGLANG_IMAGE}

ARG SOURCE_COMMIT
LABEL org.opencontainers.image.source="https://github.com/raulcdinardi/llm-debate"
LABEL org.opencontainers.image.revision="${SOURCE_COMMIT}"
LABEL org.opencontainers.image.title="Constrained-writing judge-signal Phase-0"

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV HF_HOME=/workspace/model_cache/hf_home
ENV LLM_DEBATE_SOURCE_COMMIT=${SOURCE_COMMIT}

WORKDIR /workspace/llm-debate

RUN python3 -m pip install --no-cache-dir --no-deps \
      distro==1.9.0 \
      accelerate==1.14.0 \
      peft==0.19.1

COPY . .

RUN python3 -m pip install --no-cache-dir --no-deps --no-build-isolation -e . && \
    chmod +x docker/*.sh scripts/*.py && \
    python3 -c "import accelerate, distro, peft, sglang, torch, transformers" && \
    python3 -m sglang.launch_server --help >/tmp/sglang-launch-help.txt

CMD ["/bin/bash"]
