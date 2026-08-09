from __future__ import annotations

from dataclasses import dataclass, field
import random
import re
from typing import Callable

from llm_local_rl.behavior_policy import validate_sampling_result_contract
from llm_local_rl.base_model_judge import build_base_judge_prompt, extract_strict_verdict
from llm_local_rl.chat_templates import get_chat_adapter
from llm_local_rl.debate_parity import DebateConfig, DebateResult, DebateTrajectory, Transition, Verdict
from llm_local_rl.model_io_trace import trace_context
from llm_local_rl.prompts import load_prompt
from llm_local_rl.sglang_sampling import SglangRuntimeConfig, SglangSampler
from llm_local_rl.task_types import BaseTextDebateExtension, TaskInstance, TaskSpec
from llm_local_rl.types import RolloutSampler, SamplingRequest

JudgeAdapterMode = str
JudgeFn = Callable[[str, str, str, str, str, str, str, str], tuple[Verdict, str]]

_SOLUTION_RE = re.compile(r"<SOLUTION>(.*?)</SOLUTION>", re.IGNORECASE | re.DOTALL)
_VERDICT_RE = re.compile(r"<VERDICT>\s*([AB])\s*</VERDICT>", re.IGNORECASE)
_REASONING_RE = re.compile(r"<REASONING>(.*?)</REASONING>", re.IGNORECASE | re.DOTALL)
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_THINK_TAIL_RE = re.compile(r"<think>.*$", re.IGNORECASE | re.DOTALL)
_CLOSING_THINK_RE = re.compile(r"</think>", re.IGNORECASE)
_EOS_LITERAL = "<|endoftext|>"
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")
_NUMBERED_SENTENCE_RE = re.compile(r"^\s*\d+\)")
_BASE_STOP_SENTINEL = "CONCLUDED"


def extract_solution(text: str) -> str | None:
    match = _SOLUTION_RE.search(text)
    if match is None:
        return None
    return match.group(1).strip()


def extract_verdict(text: str) -> Verdict:
    match = _VERDICT_RE.search(text)
    if match is None:
        return "INVALID"
    return match.group(1).upper()


def extract_reasoning(text: str) -> str:
    match = _REASONING_RE.search(text)
    return match.group(1).strip() if match is not None else ""


def _strip_think_blocks(text: str) -> str:
    text = _THINK_BLOCK_RE.sub("", text)
    text = _THINK_TAIL_RE.sub("", text)
    text = _CLOSING_THINK_RE.sub("", text)
    return text.strip()


def _clean_visible_completion_text(text: str) -> str:
    return _strip_think_blocks(text).replace(_EOS_LITERAL, "").split(_BASE_STOP_SENTINEL, 1)[0].strip()


def truncate_seeded_numbered_completion_text(text: str) -> str:
    clean_text = _clean_visible_completion_text(text)
    if not clean_text:
        return clean_text
    kept_end = 0
    raw_start = 0
    boundaries = [match.end() for match in _SENTENCE_BOUNDARY_RE.finditer(clean_text)]
    boundaries.append(len(clean_text))
    for raw_end in boundaries:
        raw_segment = clean_text[raw_start:raw_end]
        if raw_segment.strip():
            right_trimmed_segment = raw_segment.rstrip()
            segment_text = right_trimmed_segment
            is_numbered = kept_end == 0 or bool(_NUMBERED_SENTENCE_RE.match(segment_text))
            if is_numbered:
                kept_end = raw_start + len(right_trimmed_segment)
            else:
                break
        raw_start = raw_end
    return clean_text[:kept_end].strip()


def _base_text_prompt(*, system_text: str | None, user_text: str, assistant_prefill: str = "") -> str:
    if system_text:
        return f"System:\n{system_text}\n\nUser:\n{user_text}\nAssistant:\n{assistant_prefill}"
    return f"User:\n{user_text}\nAssistant:\n{assistant_prefill}"


def _base_debate_system_text(round_name: str) -> str:
    return (
        f"You are participating in Round {round_name} of a debate about which answer better satisfies the original task.\n"
        "Your Round 1 answer is fixed and cannot change.\n"
        "Argue only about why your fixed answer better satisfies the task constitution and required format.\n"
        "Use concrete evidence from the task, your answer, and the opponent's answer.\n"
        "Do not emit <think> tags."
    )


def _fallback_verdict_from_invalid_judge(*, step_seed: int | None, debate_idx: int) -> Verdict:
    fallback_seed = 0 if step_seed is None else step_seed
    rng = random.Random(f"{fallback_seed}:judge_invalid_fallback:{debate_idx}")
    return rng.choice(["A", "B"])


def _split_template(template: str, placeholder: str) -> tuple[str, str]:
    if placeholder not in template:
        raise ValueError(f"Template missing placeholder {placeholder!r}")
    return template.split(placeholder, 1)


def _extract_pointwise_judge_label(text: str) -> str:
    match = re.search(r"\b(PASS|FAIL)\b", text, re.IGNORECASE)
    if match is None:
        return "INVALID"
    return match.group(1).upper()


@dataclass(frozen=True)
class DebateRolloutOutput:
    debates: list[DebateResult]
    info_lines: list[str]


@dataclass(frozen=True)
class DebateRuntimeConfig:
    num_rounds: int = 3
    num_groups: int = 1
    group_size: int = 2
    enable_thinking: bool | None = None
    debate_r1_reward: str = "task"
    debate_r23_reward: str = "constant"
    debate_r23_constant: float = 0.5
    debate_r23_mode: str = "symmetric"
    judge_adapter: str = "policy"
    round_adapter_names: tuple[str, ...] = ("solution", "debate", "debate")
    rollout_batch_size: int = 0
    request_seed_mode: str = "none"
    top_p: float = 1.0
    min_p: float = 0.0
    prompt_format: str = "chat"
    r1_assistant_prefill: str = ""
    stop_on_concluded: bool = False
    base_r2_prefill: str = "The reasons that my solution is better than my opponent's are:\n1)"
    base_r3_prefill: str = "Responding to my opponent's criticism:\n1)"
    judge_prompt_format: str = "chat"
    judge_max_tokens: int = 0
    judge_temperature: float = 0.3
    judge_top_p: float = 1.0
    judge_top_k: int = -1
    judge_min_p: float = 0.0
    judge_presence_penalty: float = 0.0
    judge_repetition_penalty: float = 1.0
    judge_seed: int | None = None
    debate_judge_server_url: str | None = None
    debate_judge_server_adapter_path: str | None = None


@dataclass
class DebateRuntime:
    task: TaskSpec
    tokenizer: object
    sampler: RolloutSampler
    debate_config: DebateConfig
    runtime_config: DebateRuntimeConfig
    adapter_layout: str
    judge_fn: JudgeFn | None = None
    judge_sampler: RolloutSampler = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.runtime_config.debate_judge_server_url is None:
            self.judge_sampler = self.sampler
            return

        adapter_paths = {}
        if self.runtime_config.debate_judge_server_adapter_path is not None:
            adapter_paths["judge"] = self.runtime_config.debate_judge_server_adapter_path
        self.judge_sampler = SglangSampler(
            runtime=SglangRuntimeConfig(base_url=self.runtime_config.debate_judge_server_url),
            adapter_paths=adapter_paths,
        )

    def _run_external_judge(
        self,
        *,
        question: str,
        constitution: str,
        r1_a: str,
        r1_b: str,
        r2_a: str,
        r2_b: str,
        r3_a: str,
        r3_b: str,
    ) -> tuple[Verdict, str]:
        if self.judge_fn is None:
            raise ValueError("External judge requested but judge_fn is not configured.")
        verdict, reasoning = self.judge_fn(
            question,
            constitution,
            r1_a,
            r1_b,
            r2_a,
            r2_b,
            r3_a,
            r3_b,
        )
        return verdict, reasoning

    def _run_external_judge_three_rounds(
        self,
        *,
        inst_pairs: list[tuple[TaskInstance, TaskInstance]],
        r1_visible_text: list[str],
        r2_visible_text: list[str],
        r3_visible_text: list[str],
    ) -> tuple[list[Verdict], list[str], list[list[int]], list[list[int]], list[list[float]], list[dict], list[bool]]:
        verdicts: list[Verdict] = []
        reasonings: list[str] = []
        prompt_tokens: list[list[int]] = []
        completion_tokens: list[list[int]] = []
        completion_logprobs: list[list[float]] = []
        raw_responses: list[dict] = []
        retry_flags: list[bool] = []
        for debate_idx, (inst_a, _inst_b) in enumerate(inst_pairs):
            a_idx = 2 * debate_idx
            b_idx = 2 * debate_idx + 1
            verdict, reasoning = self._run_external_judge(
                question=self.task.judge_context_text(inst=inst_a),
                constitution=self.task.judge_constitution_text(inst=inst_a),
                r1_a=r1_visible_text[a_idx],
                r1_b=r1_visible_text[b_idx],
                r2_a=r2_visible_text[a_idx],
                r2_b=r2_visible_text[b_idx],
                r3_a=r3_visible_text[a_idx],
                r3_b=r3_visible_text[b_idx],
            )
            verdicts.append(verdict)
            reasonings.append(reasoning)
            prompt_tokens.append([])
            completion_tokens.append([])
            completion_logprobs.append([])
            raw_responses.append({"external_judge": True, "raw_text": reasoning})
            retry_flags.append(False)
        return verdicts, reasonings, prompt_tokens, completion_tokens, completion_logprobs, raw_responses, retry_flags

    def rollout(self, *, step_seed: int | None) -> DebateRolloutOutput:
        if self.runtime_config.group_size % 2 != 0:
            raise ValueError("Debate requires even group_size.")
        instances = self.task.sample_instances(n=self.runtime_config.num_groups, seed=step_seed)
        instances_repeated: list[TaskInstance] = []
        expander = getattr(self.task, "expand_group_instances", None)
        for group_idx, inst in enumerate(instances):
            if callable(expander):
                instances_repeated.extend(
                    expander(
                        inst=inst,
                        group_size=self.runtime_config.group_size,
                        seed=None if step_seed is None else step_seed + group_idx,
                    )
                )
            else:
                instances_repeated.extend([inst] * self.runtime_config.group_size)
        if self.runtime_config.num_rounds == 1:
            return self._rollout_r1_only(instances_repeated=instances_repeated, step_seed=step_seed)
        if self.runtime_config.num_rounds == 2:
            return self._rollout_two_rounds(instances_repeated=instances_repeated, step_seed=step_seed)
        if self.runtime_config.num_rounds != 3:
            raise ValueError(f"Unsupported num_rounds={self.runtime_config.num_rounds}")
        return self._rollout_three_rounds(instances_repeated=instances_repeated, step_seed=step_seed)

    def _postprocess_visible_texts(
        self,
        *,
        instances: list[TaskInstance],
        texts: list[str],
        strip_stop_sentinel: bool = False,
    ) -> tuple[list[str], list[dict]]:
        postprocess = getattr(self.task, "postprocess_visible_text", None)
        visible_texts: list[str] = []
        metrics: list[dict] = []
        for inst, text in zip(instances, texts, strict=True):
            clean_text = _strip_think_blocks(text)
            if strip_stop_sentinel:
                clean_text = clean_text.split(_BASE_STOP_SENTINEL, 1)[0].strip()
            if callable(postprocess):
                processed = postprocess(inst=inst, text=clean_text)
                if isinstance(processed, tuple):
                    visible_text, visible_metrics = processed
                else:
                    visible_text, visible_metrics = processed, {}
                visible_texts.append(str(visible_text))
                metrics.append(dict(visible_metrics))
            else:
                visible_texts.append(clean_text)
                metrics.append({})
        return visible_texts, metrics

    def sample_pointwise_judge_rewards(self, *, debates: list[DebateResult], step_seed: int | None) -> dict[int, float]:
        adapter = get_chat_adapter(self.tokenizer)
        system = load_prompt("debate/system_pointwise_judge.md")
        requests: list[SamplingRequest] = []
        keys: list[int] = []
        for debate_result in debates:
            for traj in (debate_result.trajectory_a, debate_result.trajectory_b):
                t1 = traj.transitions[0]
                prompt_text = self.tokenizer.decode(t1.prompt_tokens, skip_special_tokens=True).strip()
                answer_text = _strip_think_blocks(self.tokenizer.decode(t1.completion_tokens, skip_special_tokens=True))
                user = (
                    f"Task prompt shown to the model:\n{prompt_text}\n\n"
                    f"Candidate answer:\n{answer_text}\n\n"
                    "Does the candidate answer satisfy the task prompt, required answer format, and explicit constraints?\n"
                    "Judge instruction-following first. If the prompt has a correctness notion, require correctness as well.\n"
                    "Output only PASS or FAIL.\n"
                )
                prompt_tokens = adapter.encode_messages(
                    [{"role": "system", "content": system}, {"role": "user", "content": user}],
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                requests.append(
                    SamplingRequest(
                        adapter_name=self._judge_adapter_name(),
                        prompt_token_ids=prompt_tokens,
                        stop_token_ids=self.task.stop_token_ids(tokenizer=self.tokenizer),
                        max_tokens=8,
                        temperature=0.0,
                        seed=step_seed,
                    )
                )
                keys.append(id(traj))
        with trace_context(round_num="pointwise_judge", adapter_name=self._judge_adapter_name()):
            results = self.judge_sampler.sample_many(requests)
        reward_map: dict[int, float] = {}
        for sample_idx, (key, result) in enumerate(zip(keys, results, strict=True)):
            label = _extract_pointwise_judge_label(_strip_think_blocks(result.text))
            if label == "PASS":
                reward_map[key] = 1.0
            elif label == "FAIL":
                reward_map[key] = 0.0
            else:
                fallback_seed = 0 if step_seed is None else step_seed
                rng = random.Random(f"{fallback_seed}:pointwise_judge_invalid:{sample_idx}")
                reward_map[key] = 1.0 if rng.choice(["PASS", "FAIL"]) == "PASS" else 0.0
        return reward_map

    def _policy_adapter_name_for_round(self, *, round_num: int) -> str:
        if self.adapter_layout == "shared":
            return "shared"
        round_idx = round_num - 1
        if round_idx < 0 or round_idx >= len(self.runtime_config.round_adapter_names):
            raise ValueError(f"Missing adapter mapping for round_num={round_num}")
        return self.runtime_config.round_adapter_names[round_idx]

    def _judge_adapter_name(self) -> str:
        if self.runtime_config.debate_judge_server_url is not None:
            if self.runtime_config.debate_judge_server_adapter_path is not None:
                return "judge"
            return "base"
        if self.runtime_config.judge_adapter == "base":
            return "base"
        if self.runtime_config.judge_adapter == "judge":
            return "judge"
        if self.runtime_config.judge_adapter == "solution":
            return "solution" if self.adapter_layout == "split" else "shared"
        if self.runtime_config.judge_adapter == "debate":
            return "debate" if self.adapter_layout == "split" else "shared"
        if self.adapter_layout == "shared":
            return "shared"
        return "debate"

    def _round_seed(self, *, step_seed: int | None, request_idx: int, round_num: int) -> int | None:
        if self.runtime_config.request_seed_mode == "none":
            return None
        if self.runtime_config.request_seed_mode != "per_request":
            raise ValueError(f"Unsupported request_seed_mode={self.runtime_config.request_seed_mode!r}")
        if step_seed is None:
            return None
        return step_seed + round_num * 100000 + request_idx

    def _use_base_text_prefill(self) -> bool:
        if self.runtime_config.prompt_format == "chat":
            return False
        if self.runtime_config.prompt_format == "qwen35_base_text_prefill":
            return True
        raise ValueError(f"Unsupported prompt_format={self.runtime_config.prompt_format!r}")

    def _max_tokens_for_round(self, *, round_num: int) -> int:
        if round_num == 1:
            if self.debate_config.max_tokens_r1 is not None:
                return int(self.debate_config.max_tokens_r1)
        elif round_num in {2, 3}:
            if self.debate_config.max_tokens_r23 is not None:
                return int(self.debate_config.max_tokens_r23)
        return int(self.debate_config.max_tokens_per_turn or 64)

    def _encode_base_text(self, text: str) -> list[int]:
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    def _base_r1_prompt_tokens(self, *, inst: TaskInstance) -> list[int]:
        context_builder = getattr(self.task, "r1_context_text", None)
        if callable(context_builder):
            user_text = str(context_builder(inst=inst))
        else:
            user_text = self.task.judge_context_text(inst=inst)
        return self._encode_base_text(
            _base_text_prompt(
                system_text=None,
                user_text=user_text,
                assistant_prefill=getattr(self.runtime_config, "r1_assistant_prefill", ""),
            )
        )

    def _task_base_text_debate_extension(
        self,
        *,
        inst: TaskInstance,
        opponent_round: int,
        opponent_answer: str,
    ) -> BaseTextDebateExtension | None:
        builder = getattr(self.task, "build_base_text_debate_extension", None)
        if not callable(builder):
            return None
        extension = builder(
            inst=inst,
            opponent_round=opponent_round,
            opponent_answer=opponent_answer,
        )
        if extension is not None and not isinstance(extension, BaseTextDebateExtension):
            raise TypeError(
                "build_base_text_debate_extension must return BaseTextDebateExtension or None"
            )
        return extension

    def _base_text_debate_prefill(
        self,
        *,
        inst: TaskInstance,
        opponent_round: int,
        opponent_answer: str,
        fallback: str,
    ) -> str:
        extension = self._task_base_text_debate_extension(
            inst=inst,
            opponent_round=opponent_round,
            opponent_answer=opponent_answer,
        )
        return extension.assistant_prefill if extension is not None else fallback

    def _base_r2_continuation_tokens(
        self,
        *,
        inst: TaskInstance,
        own_r1: str,
        opponent_r1: str,
    ) -> list[int]:
        extension = self._task_base_text_debate_extension(
            inst=inst,
            opponent_round=1,
            opponent_answer=opponent_r1,
        )
        if extension is not None:
            return self._encode_base_text(
                "\n\n"
                + _base_text_prompt(
                    system_text=None,
                    user_text=extension.user_text,
                    assistant_prefill=extension.assistant_prefill,
                )
            )
        task_template = self.task.debate_r2_user_template()
        if task_template is None:
            user_text = (
                f"Original task prompt:\n{self.task.judge_context_text(inst=inst)}\n\n"
                f"Constitution:\n{self.task.judge_constitution_text(inst=inst)}\n\n"
                f"Your fixed Round 1 answer:\n{own_r1}\n\n"
                f"Opponent Round 1 answer:\n{opponent_r1}\n\n"
                "Write exactly 3 short numbered points for your Round 2 argument.\n"
                "- Argue why your fixed answer better satisfies the constitution.\n"
                "- Point out concrete weaknesses in the opponent's answer.\n"
                "- Do not revise your Round 1 answer.\n"
                f"- After point 3, immediately output {_BASE_STOP_SENTINEL} and nothing else.\n"
            )
        else:
            user_text = (
                f"Original task prompt:\n{self.task.judge_context_text(inst=inst)}\n\n"
                f"Constitution:\n{self.task.judge_constitution_text(inst=inst)}\n\n"
                f"Your fixed Round 1 answer:\n{own_r1}\n\n"
                + task_template.replace("{opponent_r1}", opponent_r1)
                + f"\n\nWrite exactly 3 short numbered points. After point 3, immediately output {_BASE_STOP_SENTINEL} and nothing else.\n"
            )
        return self._encode_base_text(
            "\n\n"
            + _base_text_prompt(
                system_text=_base_debate_system_text("2"),
                user_text=user_text,
                assistant_prefill=self.runtime_config.base_r2_prefill,
            )
        )

    def _base_r3_continuation_tokens(
        self,
        *,
        inst: TaskInstance,
        own_r1: str,
        opponent_r1: str,
        own_r2: str,
        opponent_r2: str,
    ) -> list[int]:
        extension = self._task_base_text_debate_extension(
            inst=inst,
            opponent_round=2,
            opponent_answer=opponent_r2,
        )
        if extension is not None:
            return self._encode_base_text(
                "\n\n"
                + _base_text_prompt(
                    system_text=None,
                    user_text=extension.user_text,
                    assistant_prefill=extension.assistant_prefill,
                )
            )
        task_template = self.task.debate_r3_user_template()
        if task_template is None:
            user_text = (
                f"Original task prompt:\n{self.task.judge_context_text(inst=inst)}\n\n"
                f"Constitution:\n{self.task.judge_constitution_text(inst=inst)}\n\n"
                f"Your fixed Round 1 answer:\n{own_r1}\n\n"
                f"Opponent Round 1 answer:\n{opponent_r1}\n\n"
                f"Your Round 2 argument:\n{own_r2}\n\n"
                f"Opponent Round 2 argument:\n{opponent_r2}\n\n"
                "Write exactly 3 short numbered points for your Round 3 response.\n"
                "- Respond to the opponent's criticisms.\n"
                "- Make your final case that your fixed answer better satisfies the constitution.\n"
                "- Do not revise your Round 1 answer.\n"
                f"- After point 3, immediately output {_BASE_STOP_SENTINEL} and nothing else.\n"
            )
        else:
            user_text = (
                f"Original task prompt:\n{self.task.judge_context_text(inst=inst)}\n\n"
                f"Constitution:\n{self.task.judge_constitution_text(inst=inst)}\n\n"
                f"Your fixed Round 1 answer:\n{own_r1}\n\n"
                f"Opponent Round 1 answer:\n{opponent_r1}\n\n"
                f"Your Round 2 argument:\n{own_r2}\n\n"
                + task_template.replace("{opponent_r2}", opponent_r2)
                + f"\n\nWrite exactly 3 short numbered points. After point 3, immediately output {_BASE_STOP_SENTINEL} and nothing else.\n"
            )
        return self._encode_base_text(
            "\n\n"
            + _base_text_prompt(
                system_text=_base_debate_system_text("3"),
                user_text=user_text,
                assistant_prefill=self.runtime_config.base_r3_prefill,
            )
        )

    def _truncate_numbered_result(
        self,
        *,
        completion_tokens: list[int],
        completion_logprobs: list[float],
        text: str,
        raw: dict,
    ) -> tuple[list[int], list[float], str, dict]:
        truncated_text = truncate_seeded_numbered_completion_text(text)
        clean_text = _clean_visible_completion_text(text)
        if not truncated_text or truncated_text == clean_text:
            return completion_tokens, completion_logprobs, text, raw
        for end in range(1, len(completion_tokens) + 1):
            prefix_text = _clean_visible_completion_text(
                self.tokenizer.decode(completion_tokens[:end], skip_special_tokens=True)
            )
            if prefix_text == truncated_text:
                updated_raw = dict(raw)
                updated_raw["untruncated_completion_text"] = text
                updated_raw["truncated_completion_text"] = truncated_text
                return completion_tokens[:end], completion_logprobs[:end], truncated_text, updated_raw
        updated_raw = dict(raw)
        updated_raw["untruncated_completion_text"] = text
        updated_raw["truncated_completion_text"] = truncated_text
        return completion_tokens, completion_logprobs, truncated_text, updated_raw

    def _sample_many(self, *, prompt_tokens_list: list[list[int]], round_num: int, step_seed: int | None, stop_token_ids: list[int], max_tokens: int, temperature: float, adapter_name: str | None = None) -> list[tuple[list[int], list[float], str, dict]]:
        requests = []
        use_real_debate_stop = self.runtime_config.stop_on_concluded and round_num in {2, 3}
        for idx, prompt_tokens in enumerate(prompt_tokens_list):
            requests.append(
                SamplingRequest(
                    adapter_name=self._policy_adapter_name_for_round(round_num=round_num) if adapter_name is None else adapter_name,
                    prompt_token_ids=prompt_tokens,
                    stop_token_ids=stop_token_ids,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    seed=self._round_seed(step_seed=step_seed, request_idx=idx, round_num=round_num),
                    min_p=float(self.runtime_config.min_p),
                    top_p=float(self.runtime_config.top_p),
                    stop_strings=(_BASE_STOP_SENTINEL,) if use_real_debate_stop else (),
                    include_stop_str_in_output=use_real_debate_stop,
                )
            )
        batch_size = self.runtime_config.rollout_batch_size if self.runtime_config.rollout_batch_size > 0 else len(requests)
        out = []
        for start_idx in range(0, len(requests), batch_size):
            chunk = requests[start_idx : start_idx + batch_size]
            chunk_adapter = chunk[0].adapter_name if chunk else None
            with trace_context(
                round_num=round_num,
                adapter_name=chunk_adapter,
                rollout_batch_start=start_idx,
                rollout_batch_size=len(chunk),
            ):
                results = self.sampler.sample_many(chunk)
            for request, result in zip(chunk, results, strict=True):
                validate_sampling_result_contract(request=request, result=result)
            out.extend(
                (result.completion_token_ids, result.completion_logprobs, result.text, result.raw)
                for result in results
            )
        return out

    def _sample_judge_many(self, *, prompt_tokens_list: list[list[int]], round_num: int, step_seed: int | None, stop_token_ids: list[int], max_tokens: int, temperature: float) -> list[tuple[list[int], list[float], str, dict]]:
        requests = []
        adapter_name = self._judge_adapter_name()
        for prompt_tokens in prompt_tokens_list:
            requests.append(
                SamplingRequest(
                    adapter_name=adapter_name,
                    prompt_token_ids=prompt_tokens,
                    stop_token_ids=stop_token_ids,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    seed=self.runtime_config.judge_seed,
                    min_p=float(self.runtime_config.judge_min_p),
                    top_p=float(self.runtime_config.judge_top_p),
                    top_k=int(self.runtime_config.judge_top_k),
                    presence_penalty=float(self.runtime_config.judge_presence_penalty),
                    repetition_penalty=float(self.runtime_config.judge_repetition_penalty),
                )
            )
        batch_size = self.runtime_config.rollout_batch_size if self.runtime_config.rollout_batch_size > 0 else len(requests)
        out = []
        for start_idx in range(0, len(requests), batch_size):
            chunk = requests[start_idx : start_idx + batch_size]
            chunk_adapter = chunk[0].adapter_name if chunk else None
            with trace_context(
                round_num=round_num,
                adapter_name=chunk_adapter,
                rollout_batch_start=start_idx,
                rollout_batch_size=len(chunk),
            ):
                results = self.judge_sampler.sample_many(chunk)
            out.extend(
                (result.completion_token_ids, result.completion_logprobs, result.text, result.raw)
                for result in results
            )
        return out

    def _judge_prompts(self, *, inst_pairs: list[tuple[TaskInstance, TaskInstance]], r1_visible_text: list[str], r2_visible_text: list[str], r3_visible_text: list[str]) -> list[list[int]]:
        if self.runtime_config.judge_prompt_format == "base_model_sft":
            return [
                list(
                    self.tokenizer.encode(
                        build_base_judge_prompt(
                            question=self.task.judge_context_text(inst=inst_a),
                            constitution=self.task.judge_constitution_text(inst=inst_a),
                            r1_a=r1_visible_text[2 * debate_idx],
                            r1_b=r1_visible_text[2 * debate_idx + 1],
                            r2_a=r2_visible_text[2 * debate_idx],
                            r2_b=r2_visible_text[2 * debate_idx + 1],
                            r3_a=r3_visible_text[2 * debate_idx],
                            r3_b=r3_visible_text[2 * debate_idx + 1],
                        ),
                        add_special_tokens=False,
                    )
                )
                for debate_idx, (inst_a, _inst_b) in enumerate(inst_pairs)
            ]
        adapter = get_chat_adapter(self.tokenizer)
        prompts: list[list[int]] = []
        for debate_idx, (inst_a, _inst_b) in enumerate(inst_pairs):
            a_idx = 2 * debate_idx
            b_idx = 2 * debate_idx + 1
            system = (
                self.debate_config.system_judge
                + "\n\nIMPORTANT: Output exactly one tag: <VERDICT>...</VERDICT>. VERDICT must be A or B only. Do not emit <think> tags or any reasoning."
            )
            user = (
                f"Question: {self.task.judge_context_text(inst=inst_a)}\n\n"
                f"Constitution:\n{self.task.judge_constitution_text(inst=inst_a)}\n\n"
                "=== AGENT A ===\n"
                "Round 1 (Proposal):\n"
                f"{r1_visible_text[a_idx]}\n\n"
                "Round 2 (Argument):\n"
                f"{r2_visible_text[a_idx]}\n\n"
                "Round 3 (Response):\n"
                f"{r3_visible_text[a_idx]}\n\n"
                "=== AGENT B ===\n"
                "Round 1 (Proposal):\n"
                f"{r1_visible_text[b_idx]}\n\n"
                "Round 2 (Argument):\n"
                f"{r2_visible_text[b_idx]}\n\n"
                "Round 3 (Response):\n"
                f"{r3_visible_text[b_idx]}\n\n"
                "Based on the debate above, which agent made a more convincing case?\n"
                "Consider: solution correctness, argument quality, rebuttal effectiveness.\n"
                "Return only <VERDICT>A</VERDICT> or <VERDICT>B</VERDICT>.\n"
            )
            prompts.append(
                adapter.encode_messages(
                    [{"role": "system", "content": system}, {"role": "user", "content": user}],
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            )
        return prompts

    def _rollout_three_rounds(self, *, instances_repeated: list[TaskInstance], step_seed: int | None) -> DebateRolloutOutput:
        use_base_text_prefill = self._use_base_text_prefill()
        if use_base_text_prefill:
            r2_prefix_tokens: list[int] = []
            r2_suffix_tokens: list[int] = []
            r3_prefix_tokens: list[int] = []
            r3_suffix_tokens: list[int] = []
        else:
            r2_template = self.task.debate_r2_user_template() or load_prompt("debate/r2_token_template.md")
            r3_template = self.task.debate_r3_user_template() or load_prompt("debate/r3_token_template.md")
            r2_pre, r2_post = _split_template(r2_template, "{opponent_r1}")
            r3_pre, r3_post = _split_template(r3_template, "{opponent_r2}")
            adapter = get_chat_adapter(self.tokenizer)
            r2_prefix_tokens, r2_suffix_tokens = adapter.build_user_continuation_tokens(
                user_pre=r2_pre,
                user_post=r2_post,
                enable_thinking=self.debate_config.enable_thinking,
            )
            r3_prefix_tokens, r3_suffix_tokens = adapter.build_user_continuation_tokens(
                user_pre=r3_pre,
                user_post=r3_post,
                enable_thinking=self.debate_config.enable_thinking,
            )
        base_r1_prompt_tokens = [
            (
                self._base_r1_prompt_tokens(inst=inst)
                if use_base_text_prefill
                else self.task.build_r1_prompt_tokens(
                    inst=inst,
                    tokenizer=self.tokenizer,
                    enable_thinking=self.debate_config.enable_thinking,
                )
            )
            for inst in instances_repeated
        ]
        stop_token_ids = self.task.stop_token_ids(tokenizer=self.tokenizer)
        r1_results = self._sample_many(
            prompt_tokens_list=base_r1_prompt_tokens,
            round_num=1,
            step_seed=step_seed,
            stop_token_ids=stop_token_ids,
            max_tokens=self._max_tokens_for_round(round_num=1),
            temperature=float(self.debate_config.temperature),
        )
        r1_tokens = [comp for comp, _lps, _text, _raw in r1_results]
        r1_lps = [lps for _comp, lps, _text, _raw in r1_results]
        r1_text = [text for _comp, _lps, text, _raw in r1_results]
        r1_visible_text, r1_visible_metrics = self._postprocess_visible_texts(instances=instances_repeated, texts=r1_text)
        r1_sol = [extract_solution(text) for text in r1_visible_text]
        r1_raw = [raw for _comp, _lps, _text, raw in r1_results]
        r1_task_rewards = []
        r1_task_reward_metrics = []
        for inst, comp in zip(instances_repeated, r1_tokens, strict=True):
            out = self.task.compute_reward(inst=inst, completion_tokens=comp, tokenizer=self.tokenizer)
            r1_task_rewards.append(float(out.reward))
            r1_task_reward_metrics.append(dict(out.metrics))

        n_debates = len(instances_repeated) // 2
        inst_pairs = [(instances_repeated[2 * idx], instances_repeated[2 * idx + 1]) for idx in range(n_debates)]
        r2_prompt_tokens = []
        r2_argument_prefills: list[str] = []
        for idx in range(n_debates):
            a_idx = 2 * idx
            b_idx = 2 * idx + 1
            if use_base_text_prefill:
                r2_argument_prefills.append(
                    self._base_text_debate_prefill(
                        inst=instances_repeated[a_idx],
                        opponent_round=1,
                        opponent_answer=r1_visible_text[b_idx],
                        fallback=self.runtime_config.base_r2_prefill,
                    )
                )
                r2_prompt_tokens.append(
                    base_r1_prompt_tokens[a_idx]
                    + r1_tokens[a_idx]
                    + self._base_r2_continuation_tokens(
                        inst=instances_repeated[a_idx],
                        own_r1=r1_visible_text[a_idx],
                        opponent_r1=r1_visible_text[b_idx],
                    )
                )
                r2_argument_prefills.append(
                    self._base_text_debate_prefill(
                        inst=instances_repeated[b_idx],
                        opponent_round=1,
                        opponent_answer=r1_visible_text[a_idx],
                        fallback=self.runtime_config.base_r2_prefill,
                    )
                )
                r2_prompt_tokens.append(
                    base_r1_prompt_tokens[b_idx]
                    + r1_tokens[b_idx]
                    + self._base_r2_continuation_tokens(
                        inst=instances_repeated[b_idx],
                        own_r1=r1_visible_text[b_idx],
                        opponent_r1=r1_visible_text[a_idx],
                    )
                )
            else:
                opp_r1_a = self.tokenizer.encode(r1_visible_text[b_idx], add_special_tokens=False)
                opp_r1_b = self.tokenizer.encode(r1_visible_text[a_idx], add_special_tokens=False)
                r2_prompt_tokens.append(base_r1_prompt_tokens[a_idx] + r1_tokens[a_idx] + r2_prefix_tokens + opp_r1_a + r2_suffix_tokens)
                r2_prompt_tokens.append(base_r1_prompt_tokens[b_idx] + r1_tokens[b_idx] + r2_prefix_tokens + opp_r1_b + r2_suffix_tokens)
        r2_results = self._sample_many(
            prompt_tokens_list=r2_prompt_tokens,
            round_num=2,
            step_seed=step_seed,
            stop_token_ids=stop_token_ids,
            max_tokens=self._max_tokens_for_round(round_num=2),
            temperature=float(self.debate_config.temperature),
        )
        if use_base_text_prefill and not self.runtime_config.stop_on_concluded:
            r2_results = [
                self._truncate_numbered_result(completion_tokens=comp, completion_logprobs=lps, text=text, raw=raw)
                for comp, lps, text, raw in r2_results
            ]
        r2_tokens = [comp for comp, _lps, _text, _raw in r2_results]
        r2_lps = [lps for _comp, lps, _text, _raw in r2_results]
        r2_text = [text for _comp, _lps, text, _raw in r2_results]
        r2_visible_text, r2_visible_metrics = self._postprocess_visible_texts(
            instances=instances_repeated,
            texts=r2_text,
            strip_stop_sentinel=self.runtime_config.stop_on_concluded,
        )
        r2_argument_text = (
            [
                prefill + text
                for prefill, text in zip(r2_argument_prefills, r2_visible_text, strict=True)
            ]
            if use_base_text_prefill
            else list(r2_visible_text)
        )
        r2_raw = [raw for _comp, _lps, _text, raw in r2_results]

        r3_prompt_tokens = []
        r3_argument_prefills: list[str] = []
        for idx in range(n_debates):
            a_idx = 2 * idx
            b_idx = 2 * idx + 1
            if use_base_text_prefill:
                r3_argument_prefills.append(
                    self._base_text_debate_prefill(
                        inst=instances_repeated[a_idx],
                        opponent_round=2,
                        opponent_answer=r2_argument_text[b_idx],
                        fallback=self.runtime_config.base_r3_prefill,
                    )
                )
                r3_prompt_tokens.append(
                    r2_prompt_tokens[a_idx]
                    + r2_tokens[a_idx]
                    + self._base_r3_continuation_tokens(
                        inst=instances_repeated[a_idx],
                        own_r1=r1_visible_text[a_idx],
                        opponent_r1=r1_visible_text[b_idx],
                        own_r2=r2_argument_text[a_idx],
                        opponent_r2=r2_argument_text[b_idx],
                    )
                )
                r3_argument_prefills.append(
                    self._base_text_debate_prefill(
                        inst=instances_repeated[b_idx],
                        opponent_round=2,
                        opponent_answer=r2_argument_text[a_idx],
                        fallback=self.runtime_config.base_r3_prefill,
                    )
                )
                r3_prompt_tokens.append(
                    r2_prompt_tokens[b_idx]
                    + r2_tokens[b_idx]
                    + self._base_r3_continuation_tokens(
                        inst=instances_repeated[b_idx],
                        own_r1=r1_visible_text[b_idx],
                        opponent_r1=r1_visible_text[a_idx],
                        own_r2=r2_argument_text[b_idx],
                        opponent_r2=r2_argument_text[a_idx],
                    )
                )
            else:
                opp_r2_a = self.tokenizer.encode(r2_visible_text[b_idx], add_special_tokens=False)
                opp_r2_b = self.tokenizer.encode(r2_visible_text[a_idx], add_special_tokens=False)
                r3_prompt_tokens.append(r2_prompt_tokens[a_idx] + r2_tokens[a_idx] + r3_prefix_tokens + opp_r2_a + r3_suffix_tokens)
                r3_prompt_tokens.append(r2_prompt_tokens[b_idx] + r2_tokens[b_idx] + r3_prefix_tokens + opp_r2_b + r3_suffix_tokens)
        r3_results = self._sample_many(
            prompt_tokens_list=r3_prompt_tokens,
            round_num=3,
            step_seed=step_seed,
            stop_token_ids=stop_token_ids,
            max_tokens=self._max_tokens_for_round(round_num=3),
            temperature=float(self.debate_config.temperature),
        )
        if use_base_text_prefill and not self.runtime_config.stop_on_concluded:
            r3_results = [
                self._truncate_numbered_result(completion_tokens=comp, completion_logprobs=lps, text=text, raw=raw)
                for comp, lps, text, raw in r3_results
            ]
        r3_tokens = [comp for comp, _lps, _text, _raw in r3_results]
        r3_lps = [lps for _comp, lps, _text, _raw in r3_results]
        r3_text = [text for _comp, _lps, text, _raw in r3_results]
        r3_visible_text, r3_visible_metrics = self._postprocess_visible_texts(
            instances=instances_repeated,
            texts=r3_text,
            strip_stop_sentinel=self.runtime_config.stop_on_concluded,
        )
        r3_argument_text = (
            [
                prefill + text
                for prefill, text in zip(r3_argument_prefills, r3_visible_text, strict=True)
            ]
            if use_base_text_prefill
            else list(r3_visible_text)
        )
        r3_raw = [raw for _comp, _lps, _text, raw in r3_results]

        if self.judge_fn is not None:
            verdicts, judge_reasonings, judge_prompt_tokens, judge_completion_tokens, judge_completion_logprobs, judge_raw_responses, judge_retry_flags = self._run_external_judge_three_rounds(
                inst_pairs=inst_pairs,
                r1_visible_text=r1_visible_text,
                r2_visible_text=r2_argument_text,
                r3_visible_text=r3_argument_text,
            )
        else:
            verdicts, judge_reasonings, judge_prompt_tokens, judge_completion_tokens, judge_completion_logprobs, judge_raw_responses, judge_retry_flags = self._run_llm_judge_three_rounds(
                inst_pairs=inst_pairs,
                r1_visible_text=r1_visible_text,
                r2_visible_text=r2_argument_text,
                r3_visible_text=r3_argument_text,
                step_seed=step_seed,
            )
        debates: list[DebateResult] = []
        for idx, (inst_a, inst_b) in enumerate(inst_pairs):
            a_idx = 2 * idx
            b_idx = 2 * idx + 1
            traj_a = DebateTrajectory(
                agent="A",
                transitions=[
                    Transition(prompt_tokens=base_r1_prompt_tokens[a_idx], completion_tokens=r1_tokens[a_idx], completion_logprobs=r1_lps[a_idx], round_num=1, metrics={"solution": r1_sol[a_idx], "instance_id": inst_a.instance_id, "visible_text_metrics": r1_visible_metrics[a_idx]}, raw_response=r1_raw[a_idx]),
                    Transition(prompt_tokens=r2_prompt_tokens[a_idx], completion_tokens=r2_tokens[a_idx], completion_logprobs=r2_lps[a_idx], round_num=2, metrics={"visible_text_metrics": r2_visible_metrics[a_idx]}, raw_response=r2_raw[a_idx]),
                    Transition(prompt_tokens=r3_prompt_tokens[a_idx], completion_tokens=r3_tokens[a_idx], completion_logprobs=r3_lps[a_idx], round_num=3, metrics={"visible_text_metrics": r3_visible_metrics[a_idx]}, raw_response=r3_raw[a_idx]),
                ],
                frozen_solution=r1_sol[a_idx],
                metrics={"r1": r1_text[a_idx], "r2": r2_argument_text[a_idx], "r3": r3_argument_text[a_idx], "r2_completion_raw": r2_text[a_idx], "r3_completion_raw": r3_text[a_idx], "instance_id": inst_a.instance_id, "task_reward": r1_task_rewards[a_idx], "task_reward_metrics": r1_task_reward_metrics[a_idx]},
            )
            traj_b = DebateTrajectory(
                agent="B",
                transitions=[
                    Transition(prompt_tokens=base_r1_prompt_tokens[b_idx], completion_tokens=r1_tokens[b_idx], completion_logprobs=r1_lps[b_idx], round_num=1, metrics={"solution": r1_sol[b_idx], "instance_id": inst_b.instance_id, "visible_text_metrics": r1_visible_metrics[b_idx]}, raw_response=r1_raw[b_idx]),
                    Transition(prompt_tokens=r2_prompt_tokens[b_idx], completion_tokens=r2_tokens[b_idx], completion_logprobs=r2_lps[b_idx], round_num=2, metrics={"visible_text_metrics": r2_visible_metrics[b_idx]}, raw_response=r2_raw[b_idx]),
                    Transition(prompt_tokens=r3_prompt_tokens[b_idx], completion_tokens=r3_tokens[b_idx], completion_logprobs=r3_lps[b_idx], round_num=3, metrics={"visible_text_metrics": r3_visible_metrics[b_idx]}, raw_response=r3_raw[b_idx]),
                ],
                frozen_solution=r1_sol[b_idx],
                metrics={"r1": r1_text[b_idx], "r2": r2_argument_text[b_idx], "r3": r3_argument_text[b_idx], "r2_completion_raw": r2_text[b_idx], "r3_completion_raw": r3_text[b_idx], "instance_id": inst_b.instance_id, "task_reward": r1_task_rewards[b_idx], "task_reward_metrics": r1_task_reward_metrics[b_idx]},
            )
            debates.append(
                DebateResult(
                    question=self.task.judge_context_text(inst=inst_a),
                    ground_truth=inst_a.payload.get("ground_truth"),
                    trajectory_a=traj_a,
                    trajectory_b=traj_b,
                    verdict=verdicts[idx],
                    judge_reasoning=judge_reasonings[idx],
                    metrics={"token_only_rollout": True, "task": self.task.name, "judge_retry": judge_retry_flags[idx]},
                    judge_prompt_tokens=judge_prompt_tokens[idx],
                    judge_completion_tokens=judge_completion_tokens[idx],
                    judge_completion_logprobs=judge_completion_logprobs[idx],
                    judge_raw_response=judge_raw_responses[idx],
                )
            )
        return DebateRolloutOutput(debates=debates, info_lines=[f"Debates={len(debates)} rounds=3"])

    def _rollout_two_rounds(self, *, instances_repeated: list[TaskInstance], step_seed: int | None) -> DebateRolloutOutput:
        three_round_output = self._rollout_three_rounds(instances_repeated=instances_repeated, step_seed=step_seed)
        debates = []
        for debate in three_round_output.debates:
            debates.append(
                DebateResult(
                    question=debate.question,
                    ground_truth=debate.ground_truth,
                    trajectory_a=DebateTrajectory(agent="A", transitions=debate.trajectory_a.transitions[:2], frozen_solution=debate.trajectory_a.frozen_solution, metrics=dict(debate.trajectory_a.metrics)),
                    trajectory_b=DebateTrajectory(agent="B", transitions=debate.trajectory_b.transitions[:2], frozen_solution=debate.trajectory_b.frozen_solution, metrics=dict(debate.trajectory_b.metrics)),
                    verdict=debate.verdict,
                    judge_reasoning=debate.judge_reasoning,
                    metrics=dict(debate.metrics),
                    judge_prompt_tokens=debate.judge_prompt_tokens,
                    judge_completion_tokens=debate.judge_completion_tokens,
                    judge_completion_logprobs=debate.judge_completion_logprobs,
                    judge_raw_response=debate.judge_raw_response,
                )
            )
        return DebateRolloutOutput(debates=debates, info_lines=[f"Debates={len(debates)} rounds=2"])

    def _rollout_r1_only(self, *, instances_repeated: list[TaskInstance], step_seed: int | None) -> DebateRolloutOutput:
        use_base_text_prefill = self._use_base_text_prefill()
        base_r1_prompt_tokens = [
            (
                self._base_r1_prompt_tokens(inst=inst)
                if use_base_text_prefill
                else self.task.build_r1_prompt_tokens(
                    inst=inst,
                    tokenizer=self.tokenizer,
                    enable_thinking=self.debate_config.enable_thinking,
                )
            )
            for inst in instances_repeated
        ]
        stop_token_ids = self.task.stop_token_ids(tokenizer=self.tokenizer)
        r1_results = self._sample_many(
            prompt_tokens_list=base_r1_prompt_tokens,
            round_num=1,
            step_seed=step_seed,
            stop_token_ids=stop_token_ids,
            max_tokens=self._max_tokens_for_round(round_num=1),
            temperature=float(self.debate_config.temperature),
        )
        r1_tokens = [comp for comp, _lps, _text, _raw in r1_results]
        r1_lps = [lps for _comp, lps, _text, _raw in r1_results]
        r1_text = [text for _comp, _lps, text, _raw in r1_results]
        r1_visible_text, r1_visible_metrics = self._postprocess_visible_texts(instances=instances_repeated, texts=r1_text)
        r1_sol = [extract_solution(text) for text in r1_visible_text]
        r1_raw = [raw for _comp, _lps, _text, raw in r1_results]
        r1_task_rewards = []
        r1_task_reward_metrics = []
        for inst, comp in zip(instances_repeated, r1_tokens, strict=True):
            out = self.task.compute_reward(inst=inst, completion_tokens=comp, tokenizer=self.tokenizer)
            r1_task_rewards.append(float(out.reward))
            r1_task_reward_metrics.append(dict(out.metrics))
        n_debates = len(instances_repeated) // 2
        inst_pairs = [(instances_repeated[2 * idx], instances_repeated[2 * idx + 1]) for idx in range(n_debates)]
        if self.judge_fn is not None:
            verdicts = []
            judge_reasonings = []
            judge_prompt_tokens = []
            judge_completion_tokens = []
            judge_completion_logprobs = []
            judge_raw_responses = []
            for idx, (inst_a, _inst_b) in enumerate(inst_pairs):
                a_idx = 2 * idx
                b_idx = 2 * idx + 1
                verdict, reasoning = self._run_external_judge(
                    question=self.task.judge_context_text(inst=inst_a),
                    constitution=self.task.judge_constitution_text(inst=inst_a),
                    r1_a=r1_visible_text[a_idx],
                    r1_b=r1_visible_text[b_idx],
                    r2_a="",
                    r2_b="",
                    r3_a="",
                    r3_b="",
                )
                verdicts.append(verdict)
                judge_reasonings.append(reasoning)
                judge_prompt_tokens.append([])
                judge_completion_tokens.append([])
                judge_completion_logprobs.append([])
                judge_raw_responses.append({"external_judge": True, "raw_text": reasoning})
        else:
            judge_prompt_tokens = []
            for idx, (inst_a, _inst_b) in enumerate(inst_pairs):
                a_idx = 2 * idx
                b_idx = 2 * idx + 1
                if self.runtime_config.judge_prompt_format == "base_model_sft":
                    judge_prompt_tokens.append(
                        list(
                            self.tokenizer.encode(
                                build_base_judge_prompt(
                                    question=self.task.judge_context_text(inst=inst_a),
                                    constitution=self.task.judge_constitution_text(inst=inst_a),
                                    r1_a=r1_visible_text[a_idx],
                                    r1_b=r1_visible_text[b_idx],
                                    r2_a="",
                                    r2_b="",
                                    r3_a="",
                                    r3_b="",
                                ),
                                add_special_tokens=False,
                            )
                        )
                    )
                    continue
                adapter = get_chat_adapter(self.tokenizer)
                system = (
                    self.debate_config.system_judge
                    + "\n\nIMPORTANT: Output exactly one tag: <VERDICT>...</VERDICT>. Judge task compliance first. VERDICT must be A or B only."
                )
                user = (
                    f"Task context:\n{self.task.judge_context_text(inst=inst_a)}\n\n"
                    f"Constitution:\n{self.task.judge_constitution_text(inst=inst_a)}\n\n"
                    "=== AGENT A ===\nAnswer:\n"
                    f"{r1_visible_text[a_idx]}\n\n"
                    "=== AGENT B ===\nAnswer:\n"
                    f"{r1_visible_text[b_idx]}\n\n"
                    "Compare the two answers only.\nWhich agent gave the better answer?\nReturn only <VERDICT>A</VERDICT> or <VERDICT>B</VERDICT>.\n"
                )
                judge_prompt_tokens.append(adapter.encode_messages([{"role": "system", "content": system}, {"role": "user", "content": user}], add_generation_prompt=True, enable_thinking=False))
            judge_results = self._sample_judge_many(
                prompt_tokens_list=judge_prompt_tokens,
                round_num=99,
                step_seed=step_seed,
                stop_token_ids=stop_token_ids,
                max_tokens=(
                    self.runtime_config.judge_max_tokens
                    or self._max_tokens_for_round(round_num=1)
                ),
                temperature=float(self.runtime_config.judge_temperature),
            )
            verdicts = []
            judge_reasonings = []
            judge_completion_tokens = []
            judge_completion_logprobs = []
            judge_raw_responses = []
            for idx, (comp, lps, text, raw) in enumerate(judge_results):
                verdict = (
                    extract_strict_verdict(text)
                    if self.runtime_config.judge_prompt_format == "base_model_sft"
                    else extract_verdict(_strip_think_blocks(text))
                )
                if verdict == "INVALID" and self.runtime_config.debate_r1_reward != "judge_rejection_task":
                    verdict = _fallback_verdict_from_invalid_judge(step_seed=step_seed, debate_idx=idx)
                verdicts.append(verdict)
                judge_reasonings.append(extract_reasoning(text))
                judge_completion_tokens.append(comp)
                judge_completion_logprobs.append(lps)
                judge_raw_responses.append(raw)
        debates = []
        for idx, (inst_a, inst_b) in enumerate(inst_pairs):
            a_idx = 2 * idx
            b_idx = 2 * idx + 1
            debates.append(
                DebateResult(
                    question=self.task.judge_context_text(inst=inst_a),
                    ground_truth=inst_a.payload.get("ground_truth"),
                    trajectory_a=DebateTrajectory(agent="A", transitions=[Transition(prompt_tokens=base_r1_prompt_tokens[a_idx], completion_tokens=r1_tokens[a_idx], completion_logprobs=r1_lps[a_idx], round_num=1, metrics={"solution": r1_sol[a_idx], "instance_id": inst_a.instance_id, "visible_text_metrics": r1_visible_metrics[a_idx]}, raw_response=r1_raw[a_idx])], frozen_solution=r1_sol[a_idx], metrics={"r1": r1_text[a_idx], "instance_id": inst_a.instance_id, "task_reward": r1_task_rewards[a_idx], "task_reward_metrics": r1_task_reward_metrics[a_idx]}),
                    trajectory_b=DebateTrajectory(agent="B", transitions=[Transition(prompt_tokens=base_r1_prompt_tokens[b_idx], completion_tokens=r1_tokens[b_idx], completion_logprobs=r1_lps[b_idx], round_num=1, metrics={"solution": r1_sol[b_idx], "instance_id": inst_b.instance_id, "visible_text_metrics": r1_visible_metrics[b_idx]}, raw_response=r1_raw[b_idx])], frozen_solution=r1_sol[b_idx], metrics={"r1": r1_text[b_idx], "instance_id": inst_b.instance_id, "task_reward": r1_task_rewards[b_idx], "task_reward_metrics": r1_task_reward_metrics[b_idx]}),
                    verdict=verdicts[idx],
                    judge_reasoning=judge_reasonings[idx],
                    metrics={"token_only_rollout": True, "task": self.task.name, "judge_retry": False},
                    judge_prompt_tokens=judge_prompt_tokens[idx],
                    judge_completion_tokens=judge_completion_tokens[idx],
                    judge_completion_logprobs=judge_completion_logprobs[idx],
                    judge_raw_response=judge_raw_responses[idx],
                )
            )
        return DebateRolloutOutput(debates=debates, info_lines=[f"Debates={len(debates)} rounds=1"])

    def _run_llm_judge_three_rounds(
        self,
        *,
        inst_pairs: list[tuple[TaskInstance, TaskInstance]],
        r1_visible_text: list[str],
        r2_visible_text: list[str],
        r3_visible_text: list[str],
        step_seed: int | None,
    ) -> tuple[list[Verdict], list[str], list[list[int]], list[list[int]], list[list[float]], list[dict], list[bool]]:
        prompt_tokens = self._judge_prompts(
            inst_pairs=inst_pairs,
            r1_visible_text=r1_visible_text,
            r2_visible_text=r2_visible_text,
            r3_visible_text=r3_visible_text,
        )
        stop_token_ids = self.task.stop_token_ids(tokenizer=self.tokenizer)
        results = self._sample_judge_many(
            prompt_tokens_list=prompt_tokens,
            round_num=99,
            step_seed=step_seed,
            stop_token_ids=stop_token_ids,
            max_tokens=(
                self.runtime_config.judge_max_tokens
                or self._max_tokens_for_round(round_num=1)
            ),
            temperature=float(self.runtime_config.judge_temperature),
        )
        verdicts: list[Verdict] = []
        reasonings: list[str] = []
        completion_tokens: list[list[int]] = []
        completion_logprobs: list[list[float]] = []
        raw_responses: list[dict] = []
        retry_flags: list[bool] = []
        invalid_indices: list[int] = []
        for idx, (comp, lps, text, raw) in enumerate(results):
            verdict = (
                extract_strict_verdict(text)
                if self.runtime_config.judge_prompt_format == "base_model_sft"
                else extract_verdict(_strip_think_blocks(text))
            )
            verdicts.append(verdict)
            reasonings.append(extract_reasoning(text))
            completion_tokens.append(comp)
            completion_logprobs.append(lps)
            raw_responses.append(raw)
            retry_flags.append(False)
            if verdict == "INVALID":
                invalid_indices.append(idx)
        if invalid_indices:
            retry_results = self._sample_judge_many(
                prompt_tokens_list=[prompt_tokens[idx] for idx in invalid_indices],
                round_num=100,
                step_seed=step_seed,
                stop_token_ids=stop_token_ids,
                max_tokens=(
                    self.runtime_config.judge_max_tokens
                    or self._max_tokens_for_round(round_num=1)
                ),
                temperature=0.0,
            )
            for debate_idx, (comp, lps, text, raw) in zip(invalid_indices, retry_results, strict=True):
                retry_verdict = (
                    extract_strict_verdict(text)
                    if self.runtime_config.judge_prompt_format == "base_model_sft"
                    else extract_verdict(_strip_think_blocks(text))
                )
                if retry_verdict == "INVALID":
                    continue
                verdicts[debate_idx] = retry_verdict
                reasonings[debate_idx] = extract_reasoning(text)
                completion_tokens[debate_idx] = comp
                completion_logprobs[debate_idx] = lps
                raw_responses[debate_idx] = raw
                retry_flags[debate_idx] = True
        for debate_idx, verdict in enumerate(verdicts):
            if verdict == "INVALID":
                if self.runtime_config.debate_r1_reward == "judge_rejection_task":
                    reasonings[debate_idx] = "[JUDGE INVALID -> DEBATE DROPPED FROM TRAINING]"
                else:
                    verdicts[debate_idx] = _fallback_verdict_from_invalid_judge(
                        step_seed=step_seed,
                        debate_idx=debate_idx,
                    )
                    reasonings[debate_idx] = "[JUDGE INVALID -> RANDOM FALLBACK]"
        return verdicts, reasonings, prompt_tokens, completion_tokens, completion_logprobs, raw_responses, retry_flags
