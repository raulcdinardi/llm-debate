from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

from tinker_debate.debate_env import extract_reasoning, extract_solution, extract_verdict
from tinker_debate.debate_types import DebateConfig, DebateResult, DebateTrajectory, Transition, Verdict
from tinker_debate.prompts import load_prompt
from tinker_debate.tasks.task_types import TaskInstance, TaskSpec
from tinker_debate.chat_templates import get_chat_adapter


def _split_template(template: str, placeholder: str) -> tuple[str, str]:
    if placeholder not in template:
        raise ValueError(f"Template missing placeholder {placeholder!r}")
    pre, post = template.split(placeholder, 1)
    return pre, post


JudgeFn = Callable[[str, str, str, str, str, str, str], tuple[Verdict, str]]


@dataclass(frozen=True)
class DebateRolloutOutput:
    debates: list[DebateResult]
    rollout_time_s: float
    info_lines: list[str]


@dataclass
class DebateParadigm:
    task: TaskSpec
    tokenizer: Any
    sample_token_prompts: Any
    config: DebateConfig
    judge_fn: JudgeFn | None

    async def rollout(
        self,
        *,
        step: int,
        num_groups: int,
        group_size: int,
        seed: int | None,
    ) -> DebateRolloutOutput:
        rollout_t0 = time.time()
        if group_size % 2 != 0:
            raise ValueError(f"Debate requires even group_size, got {group_size}")

        step_seed = None if seed is None else seed + step
        instances = self.task.sample_instances(n=num_groups, seed=step_seed)
        instances_repeated: list[TaskInstance] = []
        for inst in instances:
            instances_repeated.extend([inst] * group_size)

        # Pre-encode debate user templates (generic across tasks).
        r2_template = self.task.debate_r2_user_template() or load_prompt("debate/r2_token_template.md")
        r3_template = self.task.debate_r3_user_template() or load_prompt("debate/r3_token_template.md")

        r2_pre, r2_post = _split_template(r2_template, "{opponent_r1}")
        r3_pre, r3_post = _split_template(r3_template, "{opponent_r2}")

        adapter = get_chat_adapter(self.tokenizer)
        r2_prefix_tokens, r2_suffix_tokens = adapter.build_user_continuation_tokens(
            user_pre=r2_pre,
            user_post=r2_post,
        )
        r3_prefix_tokens, r3_suffix_tokens = adapter.build_user_continuation_tokens(
            user_pre=r3_pre,
            user_post=r3_post,
        )

        print(f"[debate] Round 1: sampling {len(instances_repeated)} completions...")
        base_r1_prompt_tokens = [
            self.task.build_r1_prompt_tokens(inst=inst, tokenizer=self.tokenizer) for inst in instances_repeated
        ]
        r1_results = await self.sample_token_prompts(
            prompt_tokens_list=base_r1_prompt_tokens,
            max_tokens=self.config.max_tokens_per_turn,
            temperature=self.config.temperature,
        )

        r1_tokens: list[list[int]] = []
        r1_lps: list[list[float]] = []
        r1_text: list[str] = []
        r1_sol: list[str | None] = []
        r1_raw: list[dict] = []

        for res in r1_results:
            if res is None:
                raise RuntimeError("R1 sampling failed (None result) in token-only debate.")
            _p, comp, lps, raw = res
            r1_tokens.append(comp)
            r1_lps.append(lps)
            r1_raw.append(raw)
            text = self.tokenizer.decode(comp, skip_special_tokens=True)
            r1_text.append(text)
            r1_sol.append(extract_solution(text))

        r1_task_rewards: list[float] = []
        r1_task_reward_metrics: list[dict] = []
        for inst, comp in zip(instances_repeated, r1_tokens):
            out = self.task.compute_reward(inst=inst, completion_tokens=comp, tokenizer=self.tokenizer)
            r1_task_rewards.append(float(out.reward))
            r1_task_reward_metrics.append(dict(out.metrics))

        # Pair into debates: (0,1), (2,3), ...
        n_debates = len(instances_repeated) // 2
        inst_pairs = [(instances_repeated[2 * i], instances_repeated[2 * i + 1]) for i in range(n_debates)]

        print(f"[debate] Round 2: sampling {len(instances_repeated)} completions...")
        r2_prompt_tokens_list: list[list[int]] = []
        for i in range(n_debates):
            a_idx = 2 * i
            b_idx = 2 * i + 1
            r2_a_cont = r2_prefix_tokens + r1_tokens[b_idx] + r2_suffix_tokens
            r2_b_cont = r2_prefix_tokens + r1_tokens[a_idx] + r2_suffix_tokens
            r2_prompt_tokens_list.append(base_r1_prompt_tokens[a_idx] + r1_tokens[a_idx] + r2_a_cont)
            r2_prompt_tokens_list.append(base_r1_prompt_tokens[b_idx] + r1_tokens[b_idx] + r2_b_cont)

        r2_results = await self.sample_token_prompts(
            prompt_tokens_list=r2_prompt_tokens_list,
            max_tokens=self.config.max_tokens_per_turn,
            temperature=self.config.temperature,
        )

        r2_prompt_tokens: list[list[int]] = []
        r2_tokens: list[list[int]] = []
        r2_lps: list[list[float]] = []
        r2_text: list[str] = []
        r2_raw: list[dict] = []
        for res in r2_results:
            if res is None:
                raise RuntimeError("R2 sampling failed (None result) in token-only debate.")
            p, comp, lps, raw = res
            r2_prompt_tokens.append(p)
            r2_tokens.append(comp)
            r2_lps.append(lps)
            r2_raw.append(raw)
            r2_text.append(self.tokenizer.decode(comp, skip_special_tokens=True))

        print(f"[debate] Round 3: sampling {len(instances_repeated)} completions...")
        r3_prompt_tokens_list: list[list[int]] = []
        for i in range(n_debates):
            a_idx = 2 * i
            b_idx = 2 * i + 1
            r3_a_cont = r3_prefix_tokens + r2_tokens[b_idx] + r3_suffix_tokens
            r3_b_cont = r3_prefix_tokens + r2_tokens[a_idx] + r3_suffix_tokens

            r3_prompt_tokens_list.append(r2_prompt_tokens[a_idx] + r2_tokens[a_idx] + r3_a_cont)
            r3_prompt_tokens_list.append(r2_prompt_tokens[b_idx] + r2_tokens[b_idx] + r3_b_cont)

        r3_results = await self.sample_token_prompts(
            prompt_tokens_list=r3_prompt_tokens_list,
            max_tokens=self.config.max_tokens_per_turn,
            temperature=self.config.temperature,
        )

        r3_prompt_tokens: list[list[int]] = []
        r3_tokens: list[list[int]] = []
        r3_lps: list[list[float]] = []
        r3_text: list[str] = []
        r3_raw: list[dict] = []
        for res in r3_results:
            if res is None:
                raise RuntimeError("R3 sampling failed (None result) in token-only debate.")
            p, comp, lps, raw = res
            r3_prompt_tokens.append(p)
            r3_tokens.append(comp)
            r3_lps.append(lps)
            r3_raw.append(raw)
            r3_text.append(self.tokenizer.decode(comp, skip_special_tokens=True))

        # Judge
        judge_t0 = time.time()
        verdicts: list[Verdict] = []
        judge_reasonings: list[str] = []
        judge_prompt_tokens_list: list[list[int]] = []
        judge_completion_tokens_list: list[list[int] | None] = []
        judge_completion_logprobs_list: list[list[float] | None] = []
        judge_raw_response_list: list[dict | None] = []
        judge_retry_flags: list[bool] = []

        if self.judge_fn is not None:
            for i in range(n_debates):
                a_idx = 2 * i
                b_idx = 2 * i + 1
                inst_a, _inst_b = inst_pairs[i]
                verdict, reasoning = self.judge_fn(
                    inst_a.instance_id,
                    r1_text[a_idx],
                    r1_text[b_idx],
                    r2_text[a_idx],
                    r2_text[b_idx],
                    r3_text[a_idx],
                    r3_text[b_idx],
                )
                verdicts.append(verdict)
                judge_reasonings.append(reasoning)
                judge_prompt_tokens_list.append([])
                judge_completion_tokens_list.append(None)
                judge_completion_logprobs_list.append(None)
                judge_raw_response_list.append(None)
                judge_retry_flags.append(False)
        else:
            adapter = get_chat_adapter(self.tokenizer)
            extra = (
                "\n\nIMPORTANT: Output ONLY the tags. "
                "VERDICT must be A or B (agent labels), never answer options like C/D. "
                "If tied, choose A."
            )
            for i in range(n_debates):
                a_idx = 2 * i
                b_idx = 2 * i + 1
                inst_a, _inst_b = inst_pairs[i]
                context = self.task.judge_context_text(inst=inst_a)
                system = self.config.system_judge + extra
                user = (
                    f"Question: {context}\n\n"
                    "=== AGENT A ===\n"
                    "Round 1 (Proposal):\n"
                    f"{r1_text[a_idx]}\n\n"
                    "Round 2 (Argument):\n"
                    f"{r2_text[a_idx]}\n\n"
                    "Round 3 (Response):\n"
                    f"{r3_text[a_idx]}\n\n"
                    "=== AGENT B ===\n"
                    "Round 1 (Proposal):\n"
                    f"{r1_text[b_idx]}\n\n"
                    "Round 2 (Argument):\n"
                    f"{r2_text[b_idx]}\n\n"
                    "Round 3 (Response):\n"
                    f"{r3_text[b_idx]}\n\n"
                    "Based on the debate above, which agent made a more convincing case?\n"
                    "Consider: solution correctness, argument quality, rebuttal effectiveness.\n"
                    "VERDICT must be exactly A or B (agent labels), not an answer option.\n"
                    "If tied, choose A.\n"
                )
                messages = [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ]
                judge_prompt_tokens_list.append(
                    adapter.encode_messages(messages, add_generation_prompt=True)
                )

            judge_results = await self.sample_token_prompts(
                prompt_tokens_list=judge_prompt_tokens_list,
                max_tokens=self.config.max_tokens_per_turn,
                temperature=0.3,
            )
            for r in judge_results:
                if r is None:
                    raise RuntimeError("Judge sampling failed (None result) in token-only debate.")
                _p, completion_tokens, _lps, _raw = r
                text = self.tokenizer.decode(completion_tokens, skip_special_tokens=True)
                if _raw.get("error") == "request_failed":
                    verdicts.append("INVALID")
                    judge_reasonings.append("[REQUEST FAILED]")
                else:
                    verdicts.append(extract_verdict(text))
                    judge_reasonings.append(extract_reasoning(text))
                judge_completion_tokens_list.append(completion_tokens)
                judge_completion_logprobs_list.append(_lps)
                judge_raw_response_list.append(_raw)
                judge_retry_flags.append(False)

            invalid_indices = [
                i
                for i, v in enumerate(verdicts)
                if v == "INVALID"
                and judge_raw_response_list[i] is not None
                and judge_raw_response_list[i].get("error") != "request_failed"
            ]
            for idx in invalid_indices:
                print(f"\n{'!'*60}")
                print(f"!!! JUDGE INVALID VERDICT at debate {idx} - dropping rollout")
                print(f"{'!'*60}\n")
                judge_reasonings[idx] = "[JUDGE INVALID]"

        judge_time = time.time() - judge_t0

        debates: list[DebateResult] = []
        for i in range(n_debates):
            a_idx = 2 * i
            b_idx = 2 * i + 1
            inst_a, inst_b = inst_pairs[i]
            final_verdict = verdicts[i]
            final_reasoning = judge_reasonings[i]

            traj_a = DebateTrajectory(
                agent="A",
                transitions=[
                    Transition(
                        prompt_tokens=base_r1_prompt_tokens[a_idx],
                        completion_tokens=r1_tokens[a_idx],
                        completion_logprobs=r1_lps[a_idx],
                        round_num=1,
                        metrics={"solution": r1_sol[a_idx], "instance_id": inst_a.instance_id},
                        raw_response=r1_raw[a_idx],
                    ),
                    Transition(
                        prompt_tokens=r2_prompt_tokens[a_idx],
                        completion_tokens=r2_tokens[a_idx],
                        completion_logprobs=r2_lps[a_idx],
                        round_num=2,
                        raw_response=r2_raw[a_idx],
                    ),
                    Transition(
                        prompt_tokens=r3_prompt_tokens[a_idx],
                        completion_tokens=r3_tokens[a_idx],
                        completion_logprobs=r3_lps[a_idx],
                        round_num=3,
                        raw_response=r3_raw[a_idx],
                    ),
                ],
                frozen_solution=r1_sol[a_idx],
                metrics={
                    "r1": r1_text[a_idx],
                    "r2": r2_text[a_idx],
                    "r3": r3_text[a_idx],
                    "instance_id": inst_a.instance_id,
                    "task_reward": r1_task_rewards[a_idx],
                    "task_reward_metrics": r1_task_reward_metrics[a_idx],
                },
            )
            traj_b = DebateTrajectory(
                agent="B",
                transitions=[
                    Transition(
                        prompt_tokens=base_r1_prompt_tokens[b_idx],
                        completion_tokens=r1_tokens[b_idx],
                        completion_logprobs=r1_lps[b_idx],
                        round_num=1,
                        metrics={"solution": r1_sol[b_idx], "instance_id": inst_b.instance_id},
                        raw_response=r1_raw[b_idx],
                    ),
                    Transition(
                        prompt_tokens=r2_prompt_tokens[b_idx],
                        completion_tokens=r2_tokens[b_idx],
                        completion_logprobs=r2_lps[b_idx],
                        round_num=2,
                        raw_response=r2_raw[b_idx],
                    ),
                    Transition(
                        prompt_tokens=r3_prompt_tokens[b_idx],
                        completion_tokens=r3_tokens[b_idx],
                        completion_logprobs=r3_lps[b_idx],
                        round_num=3,
                        raw_response=r3_raw[b_idx],
                    ),
                ],
                frozen_solution=r1_sol[b_idx],
                metrics={
                    "r1": r1_text[b_idx],
                    "r2": r2_text[b_idx],
                    "r3": r3_text[b_idx],
                    "instance_id": inst_b.instance_id,
                    "task_reward": r1_task_rewards[b_idx],
                    "task_reward_metrics": r1_task_reward_metrics[b_idx],
                },
            )

            debates.append(
                DebateResult(
                    question=self.task.judge_context_text(inst=inst_a),
                    ground_truth=inst_a.payload["ground_truth"] if "ground_truth" in inst_a.payload else None,
                    trajectory_a=traj_a,
                    trajectory_b=traj_b,
                    verdict=final_verdict,
                    judge_reasoning=final_reasoning,
                    metrics={
                        "judge_time_s": judge_time,
                        "token_only_rollout": True,
                        "task": self.task.name,
                        "judge_retry": judge_retry_flags[i],
                    },
                    judge_prompt_tokens=judge_prompt_tokens_list[i],
                    judge_completion_tokens=judge_completion_tokens_list[i],
                    judge_completion_logprobs=judge_completion_logprobs_list[i],
                    judge_raw_response=judge_raw_response_list[i],
                )
            )

        rollout_time = time.time() - rollout_t0
        info_lines = [f"Rollout time: {rollout_time:.1f}s (judge_time={judge_time:.1f}s)"]
        return DebateRolloutOutput(debates=debates, rollout_time_s=float(rollout_time), info_lines=info_lines)
