from __future__ import annotations

import argparse
import json
import os
import sys

from ._vllm_backend import get_vllm_llm


def _extract_token_logprobs(*, token_ids: list[int], token_logprobs: object) -> list[float]:
    if token_logprobs is None:
        raise RuntimeError("vLLM did not return per-token logprobs (set SamplingParams.logprobs=1).")
    if not isinstance(token_logprobs, list):
        raise TypeError(f"Unexpected vLLM logprobs type: {type(token_logprobs).__name__}")
    if len(token_logprobs) != len(token_ids):
        raise ValueError("vLLM logprobs length mismatch with generated token_ids.")

    out: list[float] = []
    for tok, lp_dict in zip(token_ids, token_logprobs):
        if not isinstance(lp_dict, dict):
            raise TypeError("Unexpected vLLM per-token logprobs element type (expected dict).")
        entry = lp_dict.get(int(tok))
        if entry is None:
            raise RuntimeError("vLLM logprobs missing generated token id.")
        if hasattr(entry, "logprob"):
            out.append(float(entry.logprob))
        elif isinstance(entry, (int, float)):
            out.append(float(entry))
        else:
            raise TypeError("Unexpected vLLM logprob entry type.")
    return out


def _handle_generate(msg: dict) -> dict:
    from vllm import SamplingParams as VSamplingParams
    from vllm.lora.request import LoRARequest

    request_id = msg["request_id"]
    prompt_token_ids = msg["prompt_token_ids"]
    temperature = float(msg["temperature"])
    max_tokens = int(msg["max_tokens"])
    stop_token_ids = msg["stop_token_ids"]
    top_k = int(msg["top_k"])
    top_p = float(msg["top_p"])
    min_p = float(msg["min_p"])
    seed = msg["seed"]
    lora_name = msg["lora_name"]
    lora_path = msg["lora_path"]

    if not isinstance(request_id, int):
        raise TypeError("request_id must be int")
    if not isinstance(prompt_token_ids, list):
        raise TypeError("prompt_token_ids must be list[list[int]]")
    if not isinstance(stop_token_ids, list):
        raise TypeError("stop_token_ids must be list[int]")
    if seed is not None and not isinstance(seed, int):
        raise TypeError("seed must be int or None")
    if not isinstance(lora_name, str):
        raise TypeError("lora_name must be str")
    if not isinstance(lora_path, str):
        raise TypeError("lora_path must be str")

    for row in prompt_token_ids:
        if not isinstance(row, list):
            raise TypeError("prompt_token_ids must be list[list[int]]")
        for x in row:
            if not isinstance(x, int):
                raise TypeError("prompt_token_ids must be list[list[int]]")
    for x in stop_token_ids:
        if not isinstance(x, int):
            raise TypeError("stop_token_ids must be list[int]")
    if not (0.0 < top_p <= 1.0):
        raise ValueError(f"top_p must be in (0,1], got {top_p}")
    if top_k < -1:
        raise ValueError(f"top_k must be -1 (no limit) or >= 0, got {top_k}")
    if not (0.0 <= min_p <= 1.0):
        raise ValueError(f"min_p must be in [0,1], got {min_p}")

    llm = get_vllm_llm(base_model=_BASE_MODEL)
    vparams = VSamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        stop_token_ids=stop_token_ids if stop_token_ids else None,
        logprobs=1,
        top_k=top_k,
        top_p=top_p,
        min_p=min_p,
        seed=seed,
    )
    lora_req = LoRARequest(lora_name, 1, lora_path)
    prompts = [{"prompt_token_ids": row} for row in prompt_token_ids]
    outputs = llm.generate(
        prompts,
        sampling_params=vparams,
        lora_request=lora_req,
        use_tqdm=False,
    )

    rows: list[dict] = []
    for out in outputs:
        if len(out.outputs) != 1:
            raise RuntimeError("Expected exactly 1 sequence per prompt in vLLM worker.")
        seq0 = out.outputs[0]
        toks = list(seq0.token_ids)
        lps = _extract_token_logprobs(token_ids=toks, token_logprobs=seq0.logprobs)
        rows.append({"token_ids": toks, "logprobs": lps})

    return {"type": "generate_result", "request_id": request_id, "outputs": rows}


_BASE_MODEL: str


def main() -> None:
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")
    os.environ.setdefault("VLLM_NO_PROGRESS_BAR", "1")
    json_stdout = sys.__stdout__
    sys.stdout = sys.stderr
    p = argparse.ArgumentParser(description="tinker-local vLLM worker (stdin/stdout JSONL).")
    p.add_argument("--model", required=True, help="Base model name/path for vLLM.")
    args = p.parse_args()

    global _BASE_MODEL
    _BASE_MODEL = args.model

    # Fail-fast: initialize the LLM on startup so errors are immediate.
    _ = get_vllm_llm(base_model=_BASE_MODEL)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        msg = json.loads(line)
        if not isinstance(msg, dict):
            raise TypeError("Worker protocol expects dict JSON objects per line.")

        msg_type = msg["type"]
        if msg_type == "generate":
            out = _handle_generate(msg)
        else:
            raise ValueError(f"Unknown message type: {msg_type!r}")

        json_stdout.write(json.dumps(out) + "\n")
        json_stdout.flush()


if __name__ == "__main__":
    main()
