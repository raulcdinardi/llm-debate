#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json

from dotenv import load_dotenv

import tinker


def _qwen_chat_prefix(*, user: str, system: str | None) -> str:
    # Prefix that ends right before the assistant completion tokens.
    prefix = ""
    if system is not None:
        prefix += f"<|im_start|>system\n{system}\n<|im_end|>\n"
    prefix += f"<|im_start|>user\n{user}\n<|im_end|>\n<|im_start|>assistant\n"
    return prefix


def _load_responses(args: argparse.Namespace) -> list[str]:
    if args.responses_json is not None:
        data = json.loads(args.responses_json)
        assert isinstance(data, list)
        assert all(isinstance(x, str) for x in data)
        return list(data)

    if args.responses_file is not None:
        raw = open(args.responses_file, "r").read()
        data = json.loads(raw)
        assert isinstance(data, list)
        assert all(isinstance(x, str) for x in data)
        return list(data)

    return list(args.response)


async def _token_logprobs(
    *,
    sampling_client: tinker.SamplingClient,
    full_tokens: list[int],
    sampling_params: tinker.SamplingParams,
) -> list[float]:
    resp = await sampling_client.sample_async(
        prompt=tinker.ModelInput.from_ints(full_tokens),
        num_samples=1,
        sampling_params=sampling_params,
        include_prompt_logprobs=True,
    )
    logprobs = resp.prompt_logprobs
    assert logprobs is not None, "Expected prompt_logprobs but got None"
    assert len(logprobs) == len(full_tokens)
    assert logprobs[0] is None
    out: list[float] = []
    for lp in logprobs[1:]:
        assert lp is not None
        out.append(float(lp))
    return out


async def _run(args: argparse.Namespace) -> dict:
    responses = _load_responses(args)
    assert len(responses) >= 1, "Need at least one response"
    assert len(responses) <= 10, "Too many responses (max 10) to keep output manageable"
    model = args.model
    score_base = model in ("both", "base")
    score_ft = model in ("both", "finetuned")
    if score_ft:
        assert args.ft_model_path is not None, "--ft-model-path is required when scoring finetuned"

    service = tinker.ServiceClient()
    training_client = await service.create_lora_training_client_async(base_model=args.base_model)
    tokenizer = training_client.get_tokenizer()

    prefix_text = _qwen_chat_prefix(user=args.user_message, system=args.system_message)
    prefix_tokens = tokenizer.encode(prefix_text, add_special_tokens=False)
    assert len(prefix_tokens) >= 1

    base_client = service.create_sampling_client(base_model=args.base_model) if score_base else None
    ft_client = service.create_sampling_client(model_path=args.ft_model_path) if score_ft else None

    # We only use the generation call to force the server to run a prefill and return prompt_logprobs.
    logprob_params = tinker.SamplingParams(max_tokens=1, temperature=0.0, seed=int(args.seed))

    async def _score_one(model_name: str, client: tinker.SamplingClient, response: str) -> dict:
        response_tokens = tokenizer.encode(response, add_special_tokens=False)
        assert len(response_tokens) >= 1, "Empty response after tokenization"
        full_tokens = prefix_tokens + response_tokens

        # prompt_logprobs are for tokens[1:] (tokens[0] is None); we keep indexing simple by dropping it.
        lps_no0 = await _token_logprobs(
            sampling_client=client,
            full_tokens=full_tokens,
            sampling_params=logprob_params,
        )
        start = len(prefix_tokens) - 1
        resp_lps = lps_no0[start : start + len(response_tokens)]
        assert len(resp_lps) == len(response_tokens)

        tok_info: list[dict] = []
        for tok_id, lp in zip(response_tokens, resp_lps, strict=True):
            tok_info.append(
                {
                    "token_id": int(tok_id),
                    "token_str": tokenizer.decode([int(tok_id)], skip_special_tokens=False),
                    "logprob": float(lp),
                }
            )

        return {
            "model": model_name,
            "response": response,
            "response_tokens": [int(t) for t in response_tokens],
            "token_logprobs": tok_info,
        }

    async def _score_pair(response: str) -> dict:
        if score_base and score_ft:
            base, ft = await asyncio.gather(
                _score_one("base", base_client, response),
                _score_one("finetuned", ft_client, response),
            )
        elif score_base:
            base = await _score_one("base", base_client, response)
            ft = None
        else:
            base = None
            ft = await _score_one("finetuned", ft_client, response)
        return {"response": response, "base": base, "finetuned": ft}

    scored: list[dict] = []
    for i in range(0, len(responses), int(args.concurrency)):
        batch = responses[i : i + int(args.concurrency)]
        scored.extend(await asyncio.gather(*[_score_pair(r) for r in batch]))

    return {
        "base_model": args.base_model,
        "ft_model_path": args.ft_model_path,
        "model": args.model,
        "chat_template": "qwen_im_start",
        "user_message": args.user_message,
        "system_message": args.system_message,
        "prefix_text": prefix_text,
        "seed": int(args.seed),
        "results": scored,
    }


def main() -> None:
    load_dotenv()

    ap = argparse.ArgumentParser(
        description=(
            "Compute per-token logprobs of a provided assistant response under base vs finetuned models, using "
            "Qwen-style chat formatting."
        )
    )
    ap.add_argument("--base-model", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--ft-model-path", type=str, default=None)
    ap.add_argument("--user-message", type=str, required=True)
    ap.add_argument("--system-message", type=str, default=None)
    ap.add_argument(
        "--model",
        type=str,
        choices=["both", "base", "finetuned"],
        default="both",
        help="Which model(s) to score.",
    )
    ap.add_argument("--seed", type=int, default=0)

    resp_group = ap.add_mutually_exclusive_group(required=False)
    resp_group.add_argument("--responses-json", type=str, default=None, help="JSON list of responses (strings).")
    resp_group.add_argument("--responses-file", type=str, default=None, help="Path to a JSON list of responses.")
    ap.add_argument(
        "--response",
        action="append",
        default=[],
        help="May be repeated. Ignored if --responses-json/--responses-file is provided.",
    )

    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--out", type=str, default=None, help="Optional path to write JSON output.")
    args = ap.parse_args()

    result = asyncio.run(_run(args))
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.out is not None:
        with open(args.out, "w") as f:
            f.write(payload + "\n")
    print(payload)


if __name__ == "__main__":
    main()
