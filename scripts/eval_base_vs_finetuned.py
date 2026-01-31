from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import random
from dataclasses import dataclass

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from tinker_debate.datasets import load_cnn_dailymail
from tinker_debate.summary.rewards import RewardConfig


@dataclass(frozen=True)
class EvalRecord:
    instance_id: str
    reward_base: float
    reward_ft: float
    delta_reward: float
    completion_len_base: int
    completion_len_ft: int
    delta_completion_len: int


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _paired_p_value(deltas: list[float]) -> tuple[float, float]:
    n = len(deltas)
    mean = sum(deltas) / n
    var = sum((x - mean) ** 2 for x in deltas) / (n - 1)
    std = math.sqrt(var)
    t = mean / (std / math.sqrt(n))
    p = 2.0 * (1.0 - _normal_cdf(abs(t)))
    return t, p


def _load_run_metadata(log_dir: str) -> dict:
    path = os.path.join(log_dir, "run_metadata.json")
    with open(path, "r") as f:
        return json.load(f)


def _load_prompt_rows(log_dir: str, max_prompts: int | None, seed: int | None) -> list[dict]:
    paths = sorted(glob.glob(os.path.join(log_dir, "summary_*.json")))
    assert paths, f"No summary logs found in {log_dir}"
    if seed is not None:
        random.seed(seed)
    random.shuffle(paths)
    if max_prompts is not None:
        paths = paths[:max_prompts]
    rows: list[dict] = []
    for path in paths:
        with open(path, "r") as f:
            data = json.load(f)
        rows.append(
            {
                "instance_id": str(data["instance_id"]),
                "prompt_tokens": data["prompt_tokens"],
            }
        )
    return rows


def _build_instance_map(seed: int | None, n_samples: int) -> dict[str, dict]:
    dataset = load_cnn_dailymail(n_samples=n_samples, seed=seed)
    return {str(row["id"]): row for row in dataset}


def _generate_completion_tokens(
    *,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt_tokens: list[int],
    prompt_len_limit: int,
    temperature: float,
    max_new_tokens: int,
    min_p: float,
    top_p: float,
    top_k: int,
    seed: int,
    stop_token_id: int,
) -> list[int]:
    torch.manual_seed(seed)
    prompt_tokens = prompt_tokens[-prompt_len_limit:]
    input_ids = torch.tensor([prompt_tokens], device=model.device)
    attention_mask = torch.ones_like(input_ids)
    gen_config = GenerationConfig(
        do_sample=True,
        temperature=float(temperature),
        max_new_tokens=int(max_new_tokens),
        min_p=float(min_p),
        top_p=float(top_p),
        top_k=int(top_k),
        pad_token_id=int(tokenizer.pad_token_id),
        eos_token_id=int(stop_token_id),
        return_dict_in_generate=True,
        output_scores=False,
    )
    out = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        generation_config=gen_config,
    )
    seq = out.sequences[0].tolist()
    return seq[len(prompt_tokens):]


def _compute_reward(
    *,
    reward_config: RewardConfig,
    tokenizer: AutoTokenizer,
    article: str,
    highlights: str,
    completion_tokens: list[int],
) -> float:
    summary_text = tokenizer.decode(completion_tokens, skip_special_tokens=True).strip()
    total, _scores = reward_config.compute(summary_text, article, highlights)
    return float(total)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--lora-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-prompts", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    metadata = _load_run_metadata(args.log_dir)
    run_args = metadata["args"]
    seed = int(args.seed)
    n_samples = 1000
    temperature = float(run_args["temperature"])
    max_new_tokens = int(run_args["max_tokens"])
    prompt_len_limit = 2000
    min_p = float(run_args["min_p"])
    top_p = 1.0
    top_k = 0

    prompt_rows = _load_prompt_rows(args.log_dir, args.max_prompts, seed)
    inst_map = _build_instance_map(int(run_args["seed"]), n_samples)

    instance_ids = {row["instance_id"] for row in prompt_rows}
    missing = instance_ids.difference(inst_map.keys())
    assert not missing, f"Missing {len(missing)} instance_ids in dataset map"

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    assert tokenizer.pad_token_id is not None
    stop_token_ids = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    assert len(stop_token_ids) == 1
    stop_token_id = int(stop_token_ids[0])

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        trust_remote_code=True,
    )
    model.eval()

    reward_config = RewardConfig.from_string("compression")

    records: list[EvalRecord] = []
    for idx, row in enumerate(prompt_rows):
        inst = inst_map[row["instance_id"]]
        prompt_tokens = row["prompt_tokens"]
        completion_tokens = _generate_completion_tokens(
            model=model,
            tokenizer=tokenizer,
            prompt_tokens=prompt_tokens,
            temperature=temperature,
            prompt_len_limit=prompt_len_limit,
            max_new_tokens=max_new_tokens,
            min_p=min_p,
            top_p=top_p,
            top_k=top_k,
            seed=seed + idx,
            stop_token_id=stop_token_id,
        )
        reward_base = _compute_reward(
            reward_config=reward_config,
            tokenizer=tokenizer,
            article=inst["article"],
            highlights=inst["highlights"],
            completion_tokens=completion_tokens,
        )
        records.append(
            EvalRecord(
                instance_id=row["instance_id"],
                reward_base=reward_base,
                reward_ft=0.0,
                delta_reward=0.0,
                completion_len_base=len(completion_tokens),
                completion_len_ft=0,
                delta_completion_len=0,
            )
        )

    lora_model = PeftModel.from_pretrained(model, args.lora_dir)
    lora_model.eval()

    for idx, row in enumerate(prompt_rows):
        inst = inst_map[row["instance_id"]]
        prompt_tokens = row["prompt_tokens"]
        completion_tokens = _generate_completion_tokens(
            model=lora_model,
            tokenizer=tokenizer,
            prompt_tokens=prompt_tokens,
            temperature=temperature,
            prompt_len_limit=prompt_len_limit,
            max_new_tokens=max_new_tokens,
            min_p=min_p,
            top_p=top_p,
            top_k=top_k,
            seed=seed + idx,
            stop_token_id=stop_token_id,
        )
        reward_ft = _compute_reward(
            reward_config=reward_config,
            tokenizer=tokenizer,
            article=inst["article"],
            highlights=inst["highlights"],
            completion_tokens=completion_tokens,
        )
        base = records[idx]
        records[idx] = EvalRecord(
            instance_id=base.instance_id,
            reward_base=base.reward_base,
            reward_ft=reward_ft,
            delta_reward=reward_ft - base.reward_base,
            completion_len_base=base.completion_len_base,
            completion_len_ft=len(completion_tokens),
            delta_completion_len=len(completion_tokens) - base.completion_len_base,
        )

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "eval_base_vs_finetuned.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "instance_id",
                "reward_base",
                "reward_ft",
                "delta_reward",
                "completion_len_base",
                "completion_len_ft",
                "delta_completion_len",
            ]
        )
        for r in records:
            writer.writerow(
                [
                    r.instance_id,
                    f"{r.reward_base:.6f}",
                    f"{r.reward_ft:.6f}",
                    f"{r.delta_reward:.6f}",
                    r.completion_len_base,
                    r.completion_len_ft,
                    r.delta_completion_len,
                ]
            )

    reward_deltas = [r.delta_reward for r in records]
    len_deltas = [float(r.delta_completion_len) for r in records]
    reward_t, reward_p = _paired_p_value(reward_deltas)
    len_t, len_p = _paired_p_value(len_deltas)

    md_path = os.path.join(args.out_dir, "eval_base_vs_finetuned.md")
    with open(md_path, "w") as f:
        f.write("# Base vs Finetuned Dry-Run Eval\n\n")
        f.write(f"Log dir: {args.log_dir}\n")
        f.write(f"Base model: {args.base_model}\n")
        f.write(f"LoRA dir: {args.lora_dir}\n")
        f.write(f"Prompts: {len(records)}\n")
        f.write(f"Temperature: {temperature}\n")
        f.write(f"Max new tokens: {max_new_tokens}\n")
        f.write(f"Min p: {min_p}\n")
        f.write(f"Top p: {top_p}\n")
        f.write(f"Top k: {top_k}\n\n")
        f.write("## Paired stats (normal approx)\n")
        f.write(f"- Reward delta: t={reward_t:.4f}, p={reward_p:.6f}\n")
        f.write(f"- Completion length delta: t={len_t:.4f}, p={len_p:.6f}\n")


if __name__ == "__main__":
    main()
