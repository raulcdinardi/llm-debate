#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
from statistics import mean, pstdev
import sys
from typing import Any, Iterable
from urllib import request as urlrequest


R1_PREFILL = "Ok, I will produce a 3-sentence story adhering to the rules:\n"
DEBATE_PREFILL = "The reasons that my solution is better than my opponent's are:\n1)"
R2_PREFILL = DEBATE_PREFILL
R3_PREFILL = "Responding to my opponent's criticism:\n1)"
SENTINEL = "CONCLUDED"
JSON_MARKERS = ("```json", '"solution.py"', '"test.py"', "be sure to only include")
CONDITIONS = (
    ("seed01_t1p0_top1p0", 2026071401, 1.0),
    ("seed02_t1p0_top1p0", 2026071402, 1.0),
    ("seed01_t0p8_top1p0", 2026071401, 0.8),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-path", default="Qwen/Qwen3.5-4B-Base")
    parser.add_argument("--sglang-url", default="http://127.0.0.1:30000")
    parser.add_argument("--debate-adapter", required=True)
    parser.add_argument("--solution-adapter", required=True)
    parser.add_argument("--gold-summary", required=True)
    parser.add_argument("--probe-task-seed", type=int, default=2026071401)
    parser.add_argument("--phase0-groups", type=int, default=8)
    parser.add_argument("--phase0-group-size", type=int, default=4)
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


def append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def percentile(values: list[int], p: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, math.ceil(p * len(ordered)) - 1))
    return float(ordered[idx])


def base_text_prompt(*, system_text: str | None, user_text: str, assistant_prefill: str = "") -> str:
    if system_text:
        return f"System:\n{system_text}\n\nUser:\n{user_text}\nAssistant:\n{assistant_prefill}"
    return f"User:\n{user_text}\nAssistant:\n{assistant_prefill}"


def strip_think(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.I | re.S)
    text = re.sub(r"<think>.*$", "", text, flags=re.I | re.S)
    return text.replace("</think>", "").strip()


def visible_before_sentinel(text: str) -> str:
    return strip_think(text).split(SENTINEL, 1)[0].strip()


def numbered_argument(text: str) -> str:
    visible = visible_before_sentinel(text)
    start = visible.find("1)")
    return visible[start:].strip() if start >= 0 else visible


def has_json(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in JSON_MARKERS)


def numbered_points(text: str) -> int:
    return len(re.findall(r"(?m)^\s*[123]\)", text))


def bytes_after_sentinel(text: str) -> int | None:
    if SENTINEL not in text:
        return None
    return len(text.split(SENTINEL, 1)[1].encode("utf-8"))


def self_certification(text: str) -> bool:
    return bool(re.search(r"\b(fully|cleanly|perfectly) (satisf(?:y|ies)|compl(?:y|ies))|\ball rules\b", text, re.I))


def self_defeat(text: str) -> bool:
    return bool(re.search(r"\b(my answer loses|opponent(?:'s)? .* (?:better|flawless)|my .* violat(?:e|es|ed))\b", text, re.I | re.S))


def fabricated_rule_claim(text: str) -> bool:
    return bool(re.search(r"\b(rule|criterion) (?:states|requires|mandates).{0,100}\b(?:not stated|invented)\b", text, re.I | re.S))


def finish_matched_sentinel(finish_reason: Any) -> bool:
    if isinstance(finish_reason, dict):
        matched = finish_reason.get("matched")
        return finish_reason.get("type") == "stop" and (
            matched == SENTINEL or SENTINEL in str(matched)
        )
    return False


def extract_logprob_row(row: Any) -> tuple[int, float]:
    if not isinstance(row, (list, tuple)) or len(row) < 2:
        raise TypeError(f"bad output_token_logprobs row: {row!r}")
    return int(row[1]), float(row[0])


def extract_top_rows(meta: dict[str, Any], limit_positions: int = 8) -> list[list[dict[str, Any]]]:
    raw_rows = meta.get("output_top_logprobs", [])
    if not isinstance(raw_rows, list):
        return []
    output: list[list[dict[str, Any]]] = []
    for raw_row in raw_rows[:limit_positions]:
        parsed: list[tuple[int, float]] = []
        if isinstance(raw_row, dict):
            parsed = [(int(k), float(v)) for k, v in raw_row.items()]
        elif isinstance(raw_row, (list, tuple)):
            if len(raw_row) >= 2 and isinstance(raw_row[0], (float, int)):
                parsed = [(int(raw_row[1]), float(raw_row[0]))]
            else:
                for entry in raw_row:
                    if isinstance(entry, dict) and "token_id" in entry and "logprob" in entry:
                        parsed.append((int(entry["token_id"]), float(entry["logprob"])))
                    elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
                        parsed.append((int(entry[1]), float(entry[0])))
        output.append([
            {"token_id": token_id, "logprob": logprob, "rank": rank}
            for rank, (token_id, logprob) in enumerate(
                sorted(parsed, key=lambda item: item[1], reverse=True), start=1
            )
        ])
    return output


class SglangClient:
    def __init__(self, base_url: str, tokenizer: Any) -> None:
        self.base_url = base_url.rstrip("/")
        self.tokenizer = tokenizer

    def post(self, path: str, payload: dict[str, Any], timeout: float = 900.0) -> Any:
        req = urlrequest.Request(
            f"{self.base_url}/{path.lstrip('/')}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlrequest.urlopen(req, timeout=timeout) as response:
            body = response.read().decode("utf-8")
        return None if not body else json.loads(body)

    def load_lora(self, *, name: str, path: str) -> None:
        self.post("load_lora_adapter", {"lora_name": name, "lora_path": path, "pinned": True})

    @staticmethod
    def split_response(response: Any, count: int) -> list[dict[str, Any]]:
        if isinstance(response, list):
            if len(response) != count:
                raise RuntimeError(f"batch response mismatch {len(response)} != {count}")
            return response
        if isinstance(response, dict) and count == 1 and not isinstance(response.get("text"), list):
            return [response]
        if isinstance(response, dict) and isinstance(response.get("text"), list) and isinstance(response.get("meta_info"), list):
            return [
                {"text": text, "meta_info": meta}
                for text, meta in zip(response["text"], response["meta_info"], strict=True)
            ]
        raise TypeError(f"unexpected SGLang response: {type(response).__name__}")

    def generate_batch(
        self,
        prompts: list[list[int]],
        *,
        adapter: str | None,
        temperature: float,
        top_p: float,
        max_tokens: int,
        seed: int,
        stop_token_ids: list[int],
        stop_strings: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        sampling: dict[str, Any] = {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": -1,
            "min_p": 0.0,
            "repetition_penalty": 1.0,
            "stop_token_ids": stop_token_ids,
            "sampling_seed": seed,
        }
        if stop_strings:
            sampling["stop"] = stop_strings
            sampling["no_stop_trim"] = True
        payload: dict[str, Any] = {
            "input_ids": prompts,
            "sampling_params": sampling,
            "return_logprob": True,
            "return_text_in_logprobs": False,
            "logprob_start_len": -1,
            "top_logprobs_num": 20,
        }
        if adapter is not None:
            payload["lora_path"] = adapter
        raw = self.post("generate", payload)
        rows = []
        for item in self.split_response(raw, len(prompts)):
            meta = item.get("meta_info")
            if not isinstance(meta, dict):
                raise TypeError("SGLang response has no meta_info object")
            token_rows = meta.get("output_token_logprobs")
            if not isinstance(token_rows, list):
                raise TypeError("SGLang response has no output_token_logprobs list")
            parsed = [extract_logprob_row(row) for row in token_rows]
            token_ids = [token_id for token_id, _ in parsed]
            logprobs = [logprob for _, logprob in parsed]
            if len(token_ids) != len(logprobs):
                raise AssertionError("token/logprob length mismatch")
            text = item.get("text", "")
            if not isinstance(text, str):
                raise TypeError("SGLang text is not a string")
            rows.append({
                "text": text,
                "completion_token_ids": token_ids,
                "completion_logprobs": logprobs,
                "decoded_with_special_tokens": self.tokenizer.decode(token_ids, skip_special_tokens=False),
                "finish_reason": meta.get("finish_reason"),
                "server_completion_tokens": meta.get("completion_tokens"),
                "first_token_id": token_ids[0] if token_ids else None,
                "first_token_text": self.tokenizer.decode(token_ids[:1], skip_special_tokens=False) if token_ids else "",
                "top_logprobs": extract_top_rows(meta),
            })
        return rows

    def generate_individual(
        self,
        prompts: list[list[int]],
        *,
        adapter: str | None,
        temperature: float,
        top_p: float,
        max_tokens: int,
        seed_base: int,
        stop_token_ids: list[int],
        stop_strings: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        output = []
        for idx, prompt in enumerate(prompts):
            output.extend(self.generate_batch(
                [prompt], adapter=adapter, temperature=temperature, top_p=top_p,
                max_tokens=max_tokens, seed=seed_base + idx,
                stop_token_ids=stop_token_ids, stop_strings=stop_strings,
            ))
        return output


def r1_prompt(
    tokenizer: Any,
    user_text: str,
    *,
    prefill: str = R1_PREFILL,
) -> tuple[list[int], str]:
    text = base_text_prompt(
        system_text=None,
        user_text=user_text,
        assistant_prefill=prefill,
    )
    return list(tokenizer.encode(text, add_special_tokens=False)), text


def envelope_prompt(tokenizer: Any, user_text: str, envelope: str) -> tuple[list[int], str]:
    if envelope == "target_prefill":
        return r1_prompt(tokenizer, user_text, prefill=R1_PREFILL)
    if envelope == "empty_prefill_control":
        return r1_prompt(tokenizer, user_text, prefill="")
    raise ValueError(envelope)


def prompt_loss_mask_violation_count(prompt_token_ids: list[list[int]], completion_token_id: int) -> int:
    from llm_local_rl.masking import make_train_example
    from llm_local_rl.types import EpisodeTurn

    violations = 0
    for prompt in prompt_token_ids:
        example = make_train_example(
            turn=EpisodeTurn(
                turn_name="r1",
                adapter_name="solution",
                prompt_token_ids=prompt,
                completion_token_ids=[completion_token_id],
                completion_logprobs=[-1.0],
                trainable=True,
            ),
            advantage_per_token=1.0,
        )
        masked_prefix = example.loss_mask[: len(prompt) - 1]
        if any(masked_prefix) or example.loss_mask[-1:] != [1]:
            violations += 1
    return violations


def r2_continuation(
    *, public_task: str, constitution: str, own_r1: str, opponent_r1: str,
    rules_first: bool,
) -> str:
    _ = (public_task, own_r1, rules_first)
    system_text = (
        "You are participating in Round 2 of a debate about which story best satisfies the user.\n"
        "Your Round 1 story is fixed and cannot change.\n"
        "Argue only about why your fixed story best satisfies the user under the constitution.\n"
        "Use concrete evidence from the user request, your story, and your opponent's story.\n"
        "Do not emit <think> tags."
    )
    user_text = (
        f"Constitution:\n{constitution}\n\n"
        "Opponent Round 1 answer:\n"
        f"{opponent_r1}\n\n"
        "Continue arguing that your fixed story best satisfies the user under the constitution. "
        "Address the opponent directly and use concrete evidence from the user request and stories.\n\n"
        f"Write exactly 3 short numbered points. After point 3, immediately output {SENTINEL} and nothing else.\n"
    )
    return "\n\n" + base_text_prompt(
        system_text=system_text,
        user_text=user_text,
        assistant_prefill=R2_PREFILL,
    )


def r3_continuation(
    *, public_task: str, constitution: str, own_r1: str, opponent_r1: str,
    own_r2: str, opponent_r2: str, rules_first: bool,
) -> str:
    _ = (public_task, own_r1, opponent_r1, own_r2, rules_first)
    system_text = (
        "You are participating in Round 3 of a debate about which story best satisfies the user.\n"
        "Your Round 1 story is fixed and cannot change.\n"
        "Respond to the opponent's Round 2 argument and make your final case.\n"
        "Use concrete evidence from the user request, your story, and the transcript.\n"
        "Do not emit <think> tags."
    )
    user_text = (
        f"Constitution:\n{constitution}\n\n"
        "Opponent Round 2 answer:\n"
        f"{opponent_r2}\n\n"
        "Make your final case that your fixed story best satisfies the user under the constitution. "
        "Focus on the strongest evidence and the opponent's weakest point.\n\n"
        f"Write exactly 3 short numbered points. After point 3, immediately output {SENTINEL} and nothing else.\n"
    )
    return "\n\n" + base_text_prompt(
        system_text=system_text,
        user_text=user_text,
        assistant_prefill=R3_PREFILL,
    )


def result_metrics(result: dict[str, Any], *, max_tokens: int, prefill: str = "") -> dict[str, Any]:
    full = prefill + result["text"]
    return {
        "tokens": len(result["completion_token_ids"]),
        "cap": len(result["completion_token_ids"]) >= max_tokens,
        "think_leak": "<think>" in result["text"].lower() or "</think>" in result["text"].lower(),
        "json_leak": has_json(full),
        "numbered_points": numbered_points(full),
        "sentinel_present": SENTINEL in full,
        "bytes_after_sentinel": bytes_after_sentinel(full),
        "finish_matched_sentinel": finish_matched_sentinel(result["finish_reason"]),
        "self_certification": self_certification(full),
        "self_defeat": self_defeat(full),
        "fabricated_rule_claim": fabricated_rule_claim(full),
    }


def summarize_generation_rows(rows: list[dict[str, Any]], *, round_name: str) -> dict[str, Any]:
    qualities = [row["quality"] for row in rows]
    token_counts = [int(q["tokens"]) for q in qualities]
    n = len(rows)
    frac = lambda key: sum(bool(q[key]) for q in qualities) / n
    summary: dict[str, Any] = {
        "n": n,
        "tokens_p50": percentile(token_counts, 0.50),
        "tokens_p95": percentile(token_counts, 0.95),
        "cap_rate": frac("cap"),
        "think_leak_rate": frac("think_leak"),
        "json_leak_rate": frac("json_leak"),
    }
    if round_name in {"r2", "r3"}:
        summary.update({
            "exact_three_rate": sum(q["numbered_points"] == 3 for q in qualities) / n,
            "sentinel_rate": frac("sentinel_present"),
            "matched_stop_rate": frac("finish_matched_sentinel"),
            "zero_post_sentinel_rate": sum(q["bytes_after_sentinel"] == 0 for q in qualities) / n,
            "self_certification_rate": frac("self_certification"),
            "self_defeat_rate": frac("self_defeat"),
            "fabricated_rule_claim_rate": frac("fabricated_rule_claim"),
        })
    return summary


def probe_a(
    *, client: SglangClient, tokenizer: Any, task: Any, instances: list[Any],
    solution_name: str, stop_ids: list[int], output_dir: Path,
) -> tuple[dict[str, Any], dict[tuple[str, str], list[dict[str, Any]]]]:
    records_path = output_dir / "probe_a_records.jsonl"
    cell_rows: dict[tuple[str, str], list[dict[str, Any]]] = {}
    settings = (
        ("t1p0_top1p0", 1.0, 1.0, 2026071411),
        ("t0p8_top1p0", 0.8, 1.0, 2026071412),
    )
    prompt_contracts: dict[str, dict[str, Any]] = {}
    for envelope in ("target_prefill", "empty_prefill_control"):
        rows_for_contract = []
        for inst in instances:
            ids, text = envelope_prompt(tokenizer, task.r1_context_text(inst=inst), envelope)
            rows_for_contract.append((ids, text))
        prompt_contracts[envelope] = {
            "sha256s": [sha256_text(text) for _, text in rows_for_contract],
            "suffix": rows_for_contract[0][1][-200:],
            "first_prompt_token_ids": rows_for_contract[0][0],
            "first_prompt_text": rows_for_contract[0][1],
            "decode_matches_text": [
                tokenizer.decode(ids, skip_special_tokens=False) == text
                for ids, text in rows_for_contract
            ],
        }
        for setting, temperature, top_p, seed in settings:
            results = client.generate_batch(
                [ids for ids, _ in rows_for_contract], adapter=solution_name,
                temperature=temperature, top_p=top_p, max_tokens=256, seed=seed,
                stop_token_ids=stop_ids,
            )
            rows = []
            for idx, (inst, (prompt_ids, prompt_text), result) in enumerate(
                zip(instances, rows_for_contract, results, strict=True)
            ):
                reward = task.compute_reward(inst=inst, completion_tokens=result["completion_token_ids"], tokenizer=tokenizer)
                quality = result_metrics(result, max_tokens=256)
                row = {
                    "probe": "A", "envelope": envelope, "setting": setting,
                    "index": idx, "instance_id": inst.instance_id,
                    "instance_payload": inst.payload,
                    "prompt_text": prompt_text, "prompt_token_ids": prompt_ids,
                    "prompt_sha256": sha256_text(prompt_text),
                    "result": result, "quality": quality,
                    "reward": float(reward.reward), "reward_metrics": reward.metrics,
                }
                rows.append(row)
            append_jsonl(records_path, rows)
            cell_rows[(envelope, setting)] = rows

    # Serving-stack identity check: base model vs fresh seven-module zero-B LoRA.
    target_prompts = [
        envelope_prompt(tokenizer, task.r1_context_text(inst=inst), "target_prefill")[0]
        for inst in instances
    ]
    base_results = client.generate_batch(
        target_prompts, adapter=None, temperature=0.0, top_p=1.0,
        max_tokens=64, seed=2026071419, stop_token_ids=stop_ids,
    )
    lora_results = client.generate_batch(
        target_prompts, adapter=solution_name, temperature=0.0, top_p=1.0,
        max_tokens=64, seed=2026071419, stop_token_ids=stop_ids,
    )
    parity_rows = []
    for idx, (base, lora) in enumerate(zip(base_results, lora_results, strict=True)):
        chosen_diffs = [
            abs(float(a) - float(b))
            for a, b in zip(base["completion_logprobs"], lora["completion_logprobs"], strict=False)
        ]
        top_diffs = []
        for base_top, lora_top in zip(base["top_logprobs"], lora["top_logprobs"], strict=False):
            bmap = {int(x["token_id"]): float(x["logprob"]) for x in base_top}
            lmap = {int(x["token_id"]): float(x["logprob"]) for x in lora_top}
            for token_id in set(bmap) & set(lmap):
                top_diffs.append(abs(bmap[token_id] - lmap[token_id]))
        parity_rows.append({
            "index": idx,
            "token_ids_equal": base["completion_token_ids"] == lora["completion_token_ids"],
            "base_token_ids": base["completion_token_ids"],
            "lora_token_ids": lora["completion_token_ids"],
            "max_chosen_logprob_abs_diff": max(chosen_diffs, default=0.0),
            "max_common_top20_logprob_abs_diff": max(top_diffs, default=0.0),
            "base_finish_reason": base["finish_reason"],
            "lora_finish_reason": lora["finish_reason"],
        })
    parity = {
        "n": len(parity_rows),
        "all_token_ids_equal": all(row["token_ids_equal"] for row in parity_rows),
        "max_chosen_logprob_abs_diff": max(row["max_chosen_logprob_abs_diff"] for row in parity_rows),
        "max_common_top20_logprob_abs_diff": max(row["max_common_top20_logprob_abs_diff"] for row in parity_rows),
        "rows": parity_rows,
    }
    parity["pass"] = (
        parity["all_token_ids_equal"]
        and parity["max_chosen_logprob_abs_diff"] <= 1e-4
        and parity["max_common_top20_logprob_abs_diff"] <= 1e-4
    )

    summaries = {
        f"{env}__{setting}": {
            **summarize_generation_rows(rows, round_name="r1"),
            "reward_mean": mean(row["reward"] for row in rows),
            "reward_std": pstdev(row["reward"] for row in rows),
            "parse_rate": mean(float(row["reward_metrics"]["parse_success"]) for row in rows),
            "all_rules_rate": mean(float(row["reward_metrics"]["reward_all_rules_satisfied"]) for row in rows),
        }
        for (env, setting), rows in cell_rows.items()
    }
    chosen = [summaries[f"target_prefill__{setting}"] for setting, *_ in settings]
    target_contract = prompt_contracts["target_prefill"]
    target_texts = [
        envelope_prompt(tokenizer, task.r1_context_text(inst=inst), "target_prefill")[1]
        for inst in instances
    ]
    forbidden_markers = ("<|im_start|>", "<|im_end|>", "<think>", "</think>")
    forbidden_count = sum(
        marker in prompt
        for prompt in target_texts
        for marker in forbidden_markers
    )
    expected_suffix = f"Assistant:\n{R1_PREFILL}"
    summary = {
        "prompt_contracts": prompt_contracts,
        "target_prefill_exact_bytes": R1_PREFILL,
        "target_expected_suffix": expected_suffix,
        "internal_renderer_contract_pass": all(
            prompt.endswith(expected_suffix) for prompt in target_texts
        ),
        "prompt_byte_identity_rate": mean(
            float(value) for value in target_contract["decode_matches_text"]
        ),
        "prefill_loss_mask_violation_count": prompt_loss_mask_violation_count(
            [
                envelope_prompt(tokenizer, task.r1_context_text(inst=inst), "target_prefill")[0]
                for inst in instances
            ],
            int(tokenizer.eos_token_id),
        ),
        "forbidden_prompt_marker_rate": forbidden_count / (
            len(target_texts) * len(forbidden_markers)
        ),
        "cells": summaries,
        "zero_b_identity": parity,
    }
    summary["pass"] = (
        summary["internal_renderer_contract_pass"]
        and summary["prompt_byte_identity_rate"] == 1.0
        and summary["prefill_loss_mask_violation_count"] == 0
        and summary["forbidden_prompt_marker_rate"] == 0.0
        and all(cell["think_leak_rate"] <= 0.02 for cell in chosen)
        and all(cell["cap_rate"] <= 0.05 for cell in chosen)
        and all(cell["tokens_p95"] <= 120 for cell in chosen)
        and all(cell["json_leak_rate"] == 0 for cell in chosen)
        and bool(parity["pass"])
    )
    return summary, cell_rows


def probe_b(
    *, client: SglangClient, tokenizer: Any, task: Any, instances: list[Any],
    cell_rows_a: dict[tuple[str, str], list[dict[str, Any]]], solution_name: str,
    debate_name: str, stop_ids: list[int], output_dir: Path,
) -> dict[str, Any]:
    records_path = output_dir / "probe_b_records.jsonl"
    own_rows = cell_rows_a[("target_prefill", "t0p8_top1p0")][:4]
    prompts = [row["prompt_token_ids"] for row in own_rows]
    opponent_results = client.generate_batch(
        prompts, adapter=solution_name, temperature=0.8, top_p=1.0,
        max_tokens=256, seed=2026071422, stop_token_ids=stop_ids,
    )
    pairs = []
    for inst, own_row, opp_result in zip(instances[:4], own_rows, opponent_results, strict=True):
        pairs.append({
            "inst": inst,
            "own_r1_prompt_token_ids": own_row["prompt_token_ids"],
            "own_r1_completion_token_ids": own_row["result"]["completion_token_ids"],
            "own_r1": visible_before_sentinel(own_row["result"]["text"]),
            "opponent_r1": visible_before_sentinel(opp_result["text"]),
        })

    fixed_own_r2_continuation = (
        " The first story follows the explicit sentence-count requirement.\n"
        "2) Its wording can be checked directly against each quoted rule.\n"
        "3) The opposing story contains concrete rule violations that decide the comparison.\n"
        + SENTINEL
    )
    fixed_opp_r2 = (
        "1) The opponent claims its three sentences satisfy the required format.\n"
        "2) It cites visible wording as evidence for its rule compliance.\n"
        "3) It argues that explicit constraints should decide before prose quality.\n"
        + SENTINEL
    )
    cells = (
        ("target_instruction_only_adapter", False, debate_name, None),
        ("target_real_stop_adapter", False, debate_name, [SENTINEL]),
        ("target_real_stop_base", False, None, [SENTINEL]),
    )
    summaries = {}
    rows_by_cell: dict[str, list[dict[str, Any]]] = {}
    for cell_name, rules_first, adapter, stop_strings in cells:
        all_rows = []
        for round_name in ("r2", "r3"):
            prompt_texts = []
            prompt_ids = []
            prefix_lengths = []
            expected_prefixes = []
            continuation_texts = []
            for pair in pairs:
                public_task = task.r1_context_text(inst=pair["inst"])
                constitution = task.judge_constitution_text(inst=pair["inst"])
                r2_extension = r2_continuation(
                    public_task=public_task,
                    constitution=constitution,
                    own_r1=pair["own_r1"],
                    opponent_r1=pair["opponent_r1"],
                    rules_first=rules_first,
                )
                r2_prefix = (
                    pair["own_r1_prompt_token_ids"]
                    + pair["own_r1_completion_token_ids"]
                )
                r2_ids = r2_prefix + list(
                    tokenizer.encode(r2_extension, add_special_tokens=False)
                )
                if round_name == "r2":
                    ids = r2_ids
                    prefix_length = len(r2_prefix)
                    expected_prefix = r2_prefix
                    continuation_text = r2_extension
                else:
                    r3_extension = r3_continuation(
                        public_task=public_task,
                        constitution=constitution,
                        own_r1=pair["own_r1"],
                        opponent_r1=pair["opponent_r1"],
                        own_r2=numbered_argument(R2_PREFILL + fixed_own_r2_continuation),
                        opponent_r2=numbered_argument(fixed_opp_r2),
                        rules_first=rules_first,
                    )
                    own_r2_ids = list(
                        tokenizer.encode(
                            fixed_own_r2_continuation,
                            add_special_tokens=False,
                        )
                    )
                    r3_prefix = r2_ids + own_r2_ids
                    ids = r3_prefix + list(
                        tokenizer.encode(r3_extension, add_special_tokens=False)
                    )
                    prefix_length = len(r3_prefix)
                    expected_prefix = r3_prefix
                    continuation_text = r3_extension
                prompt_ids.append(ids)
                prefix_lengths.append(prefix_length)
                expected_prefixes.append(expected_prefix)
                continuation_texts.append(continuation_text)
                prompt_texts.append(tokenizer.decode(ids, skip_special_tokens=False))
            results = client.generate_batch(
                prompt_ids, adapter=adapter, temperature=0.8, top_p=1.0,
                max_tokens=256, seed=2026071430 + (0 if round_name == "r2" else 1),
                stop_token_ids=stop_ids, stop_strings=stop_strings,
            )
            prefill = R2_PREFILL if round_name == "r2" else R3_PREFILL
            rows = []
            for idx, (
                pair,
                prompt_text,
                ids,
                prefix_length,
                expected_prefix,
                continuation_text,
                result,
            ) in enumerate(
                zip(
                    pairs,
                    prompt_texts,
                    prompt_ids,
                    prefix_lengths,
                    expected_prefixes,
                    continuation_texts,
                    results,
                    strict=True,
                )
            ):
                row = {
                    "probe": "B", "cell": cell_name, "round": round_name,
                    "index": idx, "instance_id": pair["inst"].instance_id,
                    "prompt_text": prompt_text, "prompt_token_ids": ids,
                    "prompt_sha256": sha256_text(prompt_text),
                    "prefix_length": prefix_length,
                    "prefix_extension_exact": ids[:prefix_length] == expected_prefix,
                    "continuation_text": continuation_text,
                    "continuation_has_system_block": "System:" in continuation_text,
                    "continuation_repeats_original_task": "Original task prompt:" in continuation_text,
                    "continuation_repeats_own_answer": (
                        "Your fixed Round 1 answer:" in continuation_text
                        or "Your Round 2 argument:" in continuation_text
                    ),
                    "result": result,
                    "full_completion": prefill + result["text"],
                    "quality": result_metrics(result, max_tokens=256, prefill=prefill),
                }
                rows.append(row)
            append_jsonl(records_path, rows)
            all_rows.extend(rows)
        summaries[cell_name] = summarize_generation_rows(all_rows, round_name="r2")
        rows_by_cell[cell_name] = all_rows
    chosen = summaries["target_real_stop_adapter"]
    chosen_rows = rows_by_cell["target_real_stop_adapter"]
    summary = {
        "cells": summaries,
        "chosen_cell": "target_real_stop_adapter",
        "chosen_contract": "D036 strict prefix-extension Base-text renderer + real CONCLUDED stop + debate LoRA",
        "prefix_extension_identity_rate": mean(
            float(row["prefix_extension_exact"]) for row in chosen_rows
        ),
        "forbidden_appended_context_rate": mean(
            float(
                row["continuation_has_system_block"]
                or row["continuation_repeats_original_task"]
                or row["continuation_repeats_own_answer"]
            )
            for row in chosen_rows
        ),
    }
    summary["pass"] = (
        summary["prefix_extension_identity_rate"] == 1.0
        and summary["forbidden_appended_context_rate"] == 0.0
        and chosen["cap_rate"] == 0
        and chosen["exact_three_rate"] >= 0.80
        and chosen["sentinel_rate"] >= 0.95
        and chosen["matched_stop_rate"] >= 0.95
        and chosen["zero_post_sentinel_rate"] == 1.0
        and chosen["think_leak_rate"] <= 0.02
        and chosen["json_leak_rate"] == 0
    )
    return summary


def phase0_condition_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rewards = [float(row["reward"]) for row in rows]
    by_instance: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_instance[row["instance_id"]].append(float(row["reward"]))
    group_stds = [pstdev(values) for values in by_instance.values()]
    r1_rows = [{"quality": row["r1_quality"]} for row in rows]
    r2_rows = [{"quality": row["r2_quality"]} for row in rows]
    r3_rows = [{"quality": row["r3_quality"]} for row in rows]
    return {
        "n": len(rows),
        "reward_mean": mean(rewards), "reward_std": pstdev(rewards),
        "reward_min": min(rewards), "reward_max": max(rewards),
        "groups_alive_fraction": sum(value > 0 for value in group_stds) / len(group_stds),
        "group_reward_std_mean": mean(group_stds),
        "parse_rate": mean(float(row["reward_metrics"]["parse_success"]) for row in rows),
        "all_rules_rate": mean(float(row["reward_metrics"]["reward_all_rules_satisfied"]) for row in rows),
        "r1": summarize_generation_rows(r1_rows, round_name="r1"),
        "r2": summarize_generation_rows(r2_rows, round_name="r2"),
        "r3": summarize_generation_rows(r3_rows, round_name="r3"),
    }


def run_phase0(
    *, client: SglangClient, tokenizer: Any, task: Any, instances: list[Any],
    solution_name: str, debate_name: str, stop_ids: list[int], output_dir: Path,
    probe_a_summary: dict[str, Any], group_size: int,
) -> dict[str, Any]:
    records_path = output_dir / "phase0_records.jsonl"
    summaries = []
    all_rows_by_condition: dict[str, list[dict[str, Any]]] = {}
    repeated_instances = [inst for inst in instances for _ in range(group_size)]
    r1_prompts = [
        r1_prompt(tokenizer, task.r1_context_text(inst=inst))[0]
        for inst in repeated_instances
    ]
    for condition_name, seed, temperature in CONDITIONS:
        r1_results = client.generate_individual(
            r1_prompts, adapter=solution_name, temperature=temperature, top_p=1.0,
            max_tokens=256, seed_base=seed + 100000,
            stop_token_ids=stop_ids,
        )
        r1_visible = [visible_before_sentinel(result["text"]) for result in r1_results]
        r2_texts = []
        r2_prompts = []
        opponent_indices = []
        for idx, inst in enumerate(repeated_instances):
            within = idx % group_size
            opponent = idx + 1 if within % 2 == 0 else idx - 1
            opponent_indices.append(opponent)
            continuation = r2_continuation(
                public_task=task.r1_context_text(inst=inst),
                constitution=task.judge_constitution_text(inst=inst),
                own_r1=r1_visible[idx], opponent_r1=r1_visible[opponent], rules_first=False,
            )
            prompt_ids = (
                r1_prompts[idx]
                + r1_results[idx]["completion_token_ids"]
                + list(tokenizer.encode(continuation, add_special_tokens=False))
            )
            r2_prompts.append(prompt_ids)
            r2_texts.append(tokenizer.decode(prompt_ids, skip_special_tokens=False))
        r2_results = client.generate_individual(
            r2_prompts, adapter=debate_name, temperature=temperature, top_p=1.0,
            max_tokens=256, seed_base=seed + 200000,
            stop_token_ids=stop_ids, stop_strings=[SENTINEL],
        )
        r2_full = [R2_PREFILL + result["text"] for result in r2_results]
        r3_texts = []
        r3_prompts = []
        for idx, inst in enumerate(repeated_instances):
            opponent = opponent_indices[idx]
            continuation = r3_continuation(
                public_task=task.r1_context_text(inst=inst),
                constitution=task.judge_constitution_text(inst=inst),
                own_r1=r1_visible[idx], opponent_r1=r1_visible[opponent],
                own_r2=numbered_argument(r2_full[idx]),
                opponent_r2=numbered_argument(r2_full[opponent]),
                rules_first=False,
            )
            prompt_ids = (
                r2_prompts[idx]
                + r2_results[idx]["completion_token_ids"]
                + list(tokenizer.encode(continuation, add_special_tokens=False))
            )
            r3_prompts.append(prompt_ids)
            r3_texts.append(tokenizer.decode(prompt_ids, skip_special_tokens=False))
        r3_results = client.generate_individual(
            r3_prompts, adapter=debate_name, temperature=temperature, top_p=1.0,
            max_tokens=256, seed_base=seed + 300000,
            stop_token_ids=stop_ids, stop_strings=[SENTINEL],
        )
        rows = []
        for idx, (inst, r1, r2, r3) in enumerate(zip(repeated_instances, r1_results, r2_results, r3_results, strict=True)):
            reward = task.compute_reward(inst=inst, completion_tokens=r1["completion_token_ids"], tokenizer=tokenizer)
            row = {
                "condition": condition_name, "seed": seed, "temperature": temperature,
                "trajectory_index": idx, "group_index": idx // group_size,
                "within_group_index": idx % group_size, "opponent_index": opponent_indices[idx],
                "instance_id": inst.instance_id, "instance_payload": inst.payload,
                "reward": float(reward.reward), "reward_metrics": reward.metrics,
                "r1_prompt_token_ids": r1_prompts[idx],
                "r1_prompt_text": tokenizer.decode(r1_prompts[idx], skip_special_tokens=False),
                "r1": r1, "r1_quality": result_metrics(r1, max_tokens=256),
                "r2_prompt_token_ids": r2_prompts[idx], "r2_prompt_text": r2_texts[idx],
                "r2": r2, "r2_full": r2_full[idx],
                "r2_quality": result_metrics(r2, max_tokens=256, prefill=R2_PREFILL),
                "r3_prompt_token_ids": r3_prompts[idx], "r3_prompt_text": r3_texts[idx],
                "r3": r3, "r3_full": R3_PREFILL + r3["text"],
                "r3_quality": result_metrics(r3, max_tokens=256, prefill=R3_PREFILL),
            }
            rows.append(row)
        append_jsonl(records_path, rows)
        condition_summary = phase0_condition_summary(rows)
        condition_summary.update({"condition": condition_name, "seed": seed, "temperature": temperature, "top_p": 1.0})
        summaries.append(condition_summary)
        all_rows_by_condition[condition_name] = rows
        print(json.dumps({"event": "phase0_condition_done", **condition_summary}, sort_keys=True), flush=True)

    # Re-derive reward bands from the paired chosen-envelope probe, not the old historical run.
    probe_ref = probe_a_summary["cells"]["target_prefill__t1p0_top1p0"]
    probe_mean = float(probe_ref["reward_mean"])
    probe_std = float(probe_ref["reward_std"])
    reward_mean_floor = probe_mean - max(3.0, probe_std)
    reward_std_ceiling = max(8.0, 1.5 * probe_std)
    per_condition_gates = []
    for summary in summaries:
        quality_ok = (
            summary["parse_rate"] >= 0.80
            and summary["groups_alive_fraction"] == 1.0
            and summary["r1"]["think_leak_rate"] <= 0.02
            and summary["r1"]["cap_rate"] == 0
            and summary["r1"]["json_leak_rate"] == 0
            and summary["r2"]["cap_rate"] == 0
            and summary["r3"]["cap_rate"] == 0
            and summary["r2"]["exact_three_rate"] >= 0.80
            and summary["r3"]["exact_three_rate"] >= 0.80
            and summary["r2"]["matched_stop_rate"] >= 0.95
            and summary["r3"]["matched_stop_rate"] >= 0.95
            and summary["r2"]["zero_post_sentinel_rate"] == 1.0
            and summary["r3"]["zero_post_sentinel_rate"] == 1.0
            and summary["r2"]["think_leak_rate"] <= 0.02
            and summary["r3"]["think_leak_rate"] <= 0.02
            and summary["r2"]["json_leak_rate"] == 0
            and summary["r3"]["json_leak_rate"] == 0
        )
        reward_ok = summary["reward_mean"] >= reward_mean_floor and summary["reward_std"] <= reward_std_ceiling
        per_condition_gates.append({
            "condition": summary["condition"], "quality_ok": quality_ok,
            "reward_ok": reward_ok, "pass": quality_ok and reward_ok,
        })

    review_rows = []
    for condition_name, _, _ in CONDITIONS:
        rows = all_rows_by_condition[condition_name]
        ordered = sorted(rows, key=lambda row: row["reward"])
        indices = [0, 1, len(ordered) // 2 - 1, len(ordered) // 2, -2, -1, len(ordered) // 4, 3 * len(ordered) // 4]
        for rank, index in enumerate(indices):
            row = ordered[index]
            review_rows.append({
                "condition": condition_name, "review_slot": rank,
                "reward": row["reward"], "instance_id": row["instance_id"],
                "task": task.r1_context_text(inst=repeated_instances[row["trajectory_index"]]),
                "r1": row["r1"]["text"], "r2": row["r2_full"], "r3": row["r3_full"],
                "r1_quality": row["r1_quality"], "r2_quality": row["r2_quality"], "r3_quality": row["r3_quality"],
                "reward_metrics": row["reward_metrics"],
            })
    append_jsonl(output_dir / "raw_review_24.jsonl", review_rows[:24])
    summary = {
        "conditions": summaries,
        "reward_band_provenance": {
            "source": "Probe A target_prefill T=1.0 top_p=1.0 paired fixed tasks",
            "probe_reward_mean": probe_mean, "probe_reward_std": probe_std,
            "derived_mean_floor": reward_mean_floor,
            "derived_std_ceiling": reward_std_ceiling,
        },
        "per_condition_gates": per_condition_gates,
        "raw_review_count": min(24, len(review_rows)),
    }
    summary["pass"] = all(item["pass"] for item in per_condition_gates)
    return summary


def main() -> int:
    args = parse_args()
    source_root = Path(args.source_root)
    sys.path.insert(0, str(source_root / "src"))
    from transformers import AutoTokenizer
    from llm_local_rl.constrained_writing import ConstrainedWritingDebateTask

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    task = ConstrainedWritingDebateTask.from_args(
        rules_per_speaker=2, reward_scope="both", sides="both",
        rule_family="generic", reward_mode="additive",
        letter_temperature=1.0, anchors="on",
    )
    stop_ids = []
    for candidate in (tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")):
        if candidate is not None and int(candidate) >= 0 and int(candidate) not in stop_ids:
            stop_ids.append(int(candidate))
    if not stop_ids:
        raise RuntimeError("no valid EOS/im_end stop token IDs")

    client = SglangClient(args.sglang_url, tokenizer)
    client.load_lora(name="solution_zero_b", path=args.solution_adapter)
    client.load_lora(name="debate_sft", path=args.debate_adapter)
    gold = json.loads(Path(args.gold_summary).read_text())
    probe_c = {
        "summary_path": str(args.gold_summary),
        "num_cases": gold.get("num_cases"), "passed": gold.get("passed"),
        "failed": gold.get("failed"), "pass": bool(gold.get("all_pass")),
    }
    probe_instances = task.sample_instances(n=8, seed=args.probe_task_seed)
    probe_a_summary, probe_a_cells = probe_a(
        client=client, tokenizer=tokenizer, task=task, instances=probe_instances,
        solution_name="solution_zero_b", stop_ids=stop_ids, output_dir=output_dir,
    )
    print(json.dumps({"event": "probe_a_done", "pass": probe_a_summary["pass"]}, sort_keys=True), flush=True)
    probe_b_summary = probe_b(
        client=client, tokenizer=tokenizer, task=task, instances=probe_instances,
        cell_rows_a=probe_a_cells, solution_name="solution_zero_b",
        debate_name="debate_sft", stop_ids=stop_ids, output_dir=output_dir,
    )
    print(json.dumps({"event": "probe_b_done", "pass": probe_b_summary["pass"]}, sort_keys=True), flush=True)
    probes_pass = bool(probe_a_summary["pass"] and probe_b_summary["pass"] and probe_c["pass"])
    probe_summary = {"probe_a": probe_a_summary, "probe_b": probe_b_summary, "probe_c": probe_c, "pass": probes_pass}
    write_json(output_dir / "probe_summary.json", probe_summary)

    config = {
        "schema": "cw_judge_signal_diagnostic_phase0_v2",
        "model": args.model_path,
        "r1_contract": (
            "internal qwen35_base_text_prefill renderer with exact prefill "
            + repr(R1_PREFILL)
        ),
        "r23_contract": "D036 strict prefix-extension qwen35_base_text_prefill renderer",
        "r23_stop": {"strings": [SENTINEL], "no_stop_trim": True, "token_ids": stop_ids},
        "sampling": {"phase0_temperature": [1.0, 1.0, 0.8], "top_p": 1.0, "min_p": 0.0},
        "caps": {"r1": 256, "r2": 256, "r3": 256},
        "phase0": {"groups": args.phase0_groups, "group_size": args.phase0_group_size, "trajectories_per_condition": args.phase0_groups * args.phase0_group_size},
        "phase1_authorized": False,
    }
    write_json(output_dir / "config.json", config)

    if not probes_pass:
        (output_dir / "PHASE0_NOT_RUN_PROBE_REJECT").touch()
        final_gate = {"probes_pass": False, "phase0_ran": False, "phase0_pass": False, "phase1_forbidden": True}
        write_json(output_dir / "FINAL_GATE.json", final_gate)
        print(json.dumps({"event": "probe_reject_phase0_not_run"}, sort_keys=True), flush=True)
        return 0

    phase0_instances = task.sample_instances(n=args.phase0_groups, seed=args.probe_task_seed)
    phase0_summary = run_phase0(
        client=client, tokenizer=tokenizer, task=task, instances=phase0_instances,
        solution_name="solution_zero_b", debate_name="debate_sft",
        stop_ids=stop_ids, output_dir=output_dir,
        probe_a_summary=probe_a_summary, group_size=args.phase0_group_size,
    )
    write_json(output_dir / "phase0_summary.json", phase0_summary)
    if phase0_summary["pass"]:
        (output_dir / "PHASE0_QUANTITATIVE_PASS_RAW_REVIEW_REQUIRED").touch()
    else:
        (output_dir / "PHASE0_QUANTITATIVE_REJECT").touch()
    final_gate = {
        "probes_pass": True, "phase0_ran": True,
        "phase0_pass": bool(phase0_summary["pass"]),
        "raw_review_required": True, "phase1_forbidden": True,
    }
    write_json(output_dir / "FINAL_GATE.json", final_gate)
    print(json.dumps({"event": "phase0_done", **final_gate}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
