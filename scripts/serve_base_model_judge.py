#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Lock
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList

from llm_local_rl.base_model_judge import extract_strict_verdict


class PresencePenaltyLogitsProcessor(LogitsProcessor):
    def __init__(self, penalty: float, *, ignored_token_id: int | None = None) -> None:
        self.penalty = float(penalty)
        self.ignored_token_id = ignored_token_id

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.penalty == 0.0:
            return scores
        for batch_index in range(input_ids.shape[0]):
            seen = torch.unique(input_ids[batch_index])
            if self.ignored_token_id is not None:
                seen = seen[seen != self.ignored_token_id]
            scores[batch_index, seen] -= self.penalty
        return scores


@dataclass(frozen=True)
class ServerConfig:
    host: str
    port: int
    model_id: str
    quantization: str
    microbatch_size: int
    max_new_tokens: int
    temperature: float
    top_p: float
    top_k: int
    min_p: float
    presence_penalty: float
    repetition_penalty: float
    seed: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batched Hugging Face base-model debate judge server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8123)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--quantization", choices=("none", "4bit", "8bit"), default="none")
    parser.add_argument("--microbatch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--presence-penalty", type=float, default=0.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--metadata-path")
    return parser.parse_args()


def load_model_and_tokenizer(*, model_id: str, quantization: str):
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    local_only = os.environ.get("HF_HUB_OFFLINE") == "1"
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, use_fast=True, token=token, local_files_only=local_only, trust_remote_code=True
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    kwargs: dict[str, Any] = {
        "token": token,
        "torch_dtype": torch.bfloat16,
        "local_files_only": local_only,
        "trust_remote_code": True,
    }
    if quantization == "4bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        kwargs["device_map"] = "auto"
    elif quantization == "8bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    adapter_path = os.environ.get("JUDGE_ADAPTER_PATH")
    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)
    if quantization == "none":
        model.to("cuda")
    model.eval()
    return model, tokenizer


def _trim_completion(token_ids: list[int], *, eos_token_id: int | None) -> list[int]:
    if eos_token_id is None or eos_token_id not in token_ids:
        return token_ids
    return token_ids[: token_ids.index(eos_token_id) + 1]


def generate_batch(*, model, tokenizer, prompt_texts: list[str], config: ServerConfig) -> list[dict]:
    if not prompt_texts:
        return []
    encoded = tokenizer(prompt_texts, padding=True, add_special_tokens=False, return_tensors="pt")
    encoded = {name: value.to(model.device) for name, value in encoded.items()}
    prompt_width = encoded["input_ids"].shape[1]
    torch.manual_seed(config.seed)
    processors = LogitsProcessorList()
    if config.presence_penalty:
        processors.append(
            PresencePenaltyLogitsProcessor(
                config.presence_penalty, ignored_token_id=tokenizer.pad_token_id
            )
        )
    generation_kwargs: dict[str, Any] = {
        "do_sample": config.temperature > 0,
        "top_p": config.top_p,
        "top_k": config.top_k,
        "min_p": config.min_p,
        "repetition_penalty": config.repetition_penalty,
        "logits_processor": processors,
        "max_new_tokens": config.max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if config.temperature > 0:
        generation_kwargs["temperature"] = config.temperature
    with torch.no_grad():
        output = model.generate(
            **encoded,
            **generation_kwargs,
        )
    results = []
    for row in output:
        completion_ids = _trim_completion(
            row[prompt_width:].tolist(), eos_token_id=tokenizer.eos_token_id
        )
        raw_text = tokenizer.decode(completion_ids, skip_special_tokens=True)
        results.append(
            {
                "raw_text": raw_text,
                "parsed_verdict": extract_strict_verdict(raw_text),
                "completion_token_count": len(completion_ids),
                "cap_hit": len(completion_ids) == config.max_new_tokens,
            }
        )
    return results


def serve(*, model, tokenizer, config: ServerConfig) -> None:
    lock = Lock()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path != "/health":
                self.send_error(404)
                return
            self._reply({"ok": True, "model_id": config.model_id})

        def do_POST(self) -> None:
            if self.path != "/judge":
                self.send_error(404)
                return
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            is_single = "prompt_text" in payload
            prompt_texts = [str(payload["prompt_text"])] if is_single else [str(x) for x in payload["prompt_texts"]]
            results: list[dict] = []
            with lock:
                for start in range(0, len(prompt_texts), config.microbatch_size):
                    results.extend(
                        generate_batch(
                            model=model,
                            tokenizer=tokenizer,
                            prompt_texts=prompt_texts[start : start + config.microbatch_size],
                            config=config,
                        )
                    )
            self._reply(results[0] if is_single else {"results": results})

        def _reply(self, value: dict) -> None:
            data = json.dumps(value).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, format: str, *args) -> None:
            return

    ThreadingHTTPServer((config.host, config.port), Handler).serve_forever()


def main() -> None:
    args = parse_args()
    if args.microbatch_size <= 0:
        raise ValueError("--microbatch-size must be positive")
    config = ServerConfig(**{name: getattr(args, name) for name in ServerConfig.__dataclass_fields__})
    model, tokenizer = load_model_and_tokenizer(model_id=config.model_id, quantization=config.quantization)
    if args.metadata_path:
        metadata_path = os.path.abspath(args.metadata_path)
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
        with open(metadata_path, "w", encoding="utf-8") as handle:
            json.dump(asdict(config), handle, indent=2)
    print(json.dumps({"event": "judge_server_ready", **asdict(config)}), flush=True)
    serve(model=model, tokenizer=tokenizer, config=config)


if __name__ == "__main__":
    main()
