from __future__ import annotations

from dataclasses import dataclass, field
import random
import re
from typing import Callable

from llm_local_rl.behavior_policy import validate_sampling_result_contract
from llm_local_rl.judge_harness import (
    AgentDebateText,
    CHAT_SOLUTION_TAGGED_V1,
    JudgeTranscript,
    get_judge_harness,
)
from llm_local_rl.soft_judge import (
    JUDGE_LABEL_TOKEN_CONTRACT_NONE,
    LFM25_OPENBOOKQA_SPACED_AB_V1,
    order_symmetric_soft_judge_score,
    resolve_judge_label_token_contract,
    validate_judge_prompt_label_boundary,
)
from llm_local_rl.chat_templates import get_chat_adapter
from llm_local_rl.debate_parity import DebateConfig, DebateResult, DebateTrajectory, Transition, Verdict
from llm_local_rl.model_io_trace import trace_context
from llm_local_rl.prompts import load_prompt
from llm_local_rl.sglang_sampling import SglangRuntimeConfig, SglangSampler
from llm_local_rl.task_types import BaseTextDebateExtension, TaskInstance, TaskSpec
from llm_local_rl.types import RolloutSampler, SamplingRequest

JudgeAdapterMode = str
JudgeFn = Callable[..., tuple[Verdict, str]]

_SOLUTION_RE = re.compile(r"<SOLUTION>(.*?)</SOLUTION>", re.IGNORECASE | re.DOTALL)
_VERDICT_RE = re.compile(r"<VERDICT>\s*([AB])\s*</VERDICT>", re.IGNORECASE)
_REASONING_RE = re.compile(r"<REASONING>(.*?)</REASONING>", re.IGNORECASE | re.DOTALL)
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_THINK_TAIL_RE = re.compile(r"<think>.*$", re.IGNORECASE | re.DOTALL)
_CLOSING_THINK_RE = re.compile(r"</think>", re.IGNORECASE)
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


def _fallback_verdict_from_order_disagreement(*, step_seed: int | None, debate_idx: int) -> Verdict:
    fallback_seed = 0 if step_seed is None else step_seed
    rng = random.Random(f"{fallback_seed}:judge_order_disagreement:{debate_idx}")
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
    min_num_rounds: int = 0
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
    judge_harness_id: str = CHAT_SOLUTION_TAGGED_V1
    judge_max_tokens: int = 0
    judge_temperature: float = 1.0
    judge_top_p: float = 1.0
    judge_top_k: int = -1
    judge_min_p: float = 0.0
    judge_presence_penalty: float = 0.0
    judge_repetition_penalty: float = 1.0
    judge_seed: int | None = None
    judge_bidirectional: bool = False
    judge_constrain_single_token: bool = False
    judge_score_mode: str = "hard_verdict"
    judge_label_token_contract: str = JUDGE_LABEL_TOKEN_CONTRACT_NONE
    debate_judge_server_url: str | None = None
    debate_judge_server_adapter_path: str | None = None

    def __post_init__(self) -> None:
        minimum = self.min_num_rounds or self.num_rounds
        if self.num_rounds < 1 or minimum < 1 or minimum > self.num_rounds:
            raise ValueError("Round range must satisfy 1 <= min_num_rounds <= num_rounds")
        if not self.round_adapter_names:
            raise ValueError("round_adapter_names must not be empty")

    def effective_min_num_rounds(self) -> int:
        return self.min_num_rounds or self.num_rounds


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
        rounds_a: list[str] | tuple[str, ...],
        rounds_b: list[str] | tuple[str, ...],
    ) -> tuple[Verdict, str]:
        if self.judge_fn is None:
            raise ValueError("External judge requested but judge_fn is not configured.")
        if len(rounds_a) != len(rounds_b):
            raise ValueError("External judge received different A/B round counts")
        if len(rounds_a) < 3:
            padding = ("",) * (3 - len(rounds_a))
            rounds_a = (*rounds_a, *padding)
            rounds_b = (*rounds_b, *padding)
        interleaved = tuple(
            text
            for pair in zip(rounds_a, rounds_b, strict=True)
            for text in pair
        )
        verdict, reasoning = self.judge_fn(question, constitution, *interleaved)
        return verdict, reasoning

    def _run_external_judge_transcripts(
        self,
        *,
        transcripts: list[JudgeTranscript],
    ) -> tuple[list[Verdict], list[str], list[list[int]], list[list[int]], list[list[float]], list[dict], list[bool]]:
        verdicts: list[Verdict] = []
        reasonings: list[str] = []
        prompt_tokens: list[list[int]] = []
        completion_tokens: list[list[int]] = []
        completion_logprobs: list[list[float]] = []
        raw_responses: list[dict] = []
        retry_flags: list[bool] = []
        for transcript in transcripts:
            verdict, reasoning = self._run_external_judge(
                question=transcript.question,
                constitution=transcript.constitution,
                rounds_a=transcript.agent_a.rounds,
                rounds_b=transcript.agent_b.rounds,
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
        harness = get_judge_harness(self.runtime_config.judge_harness_id)
        if self.judge_fn is None and self.runtime_config.effective_min_num_rounds() < harness.required_rounds:
            raise ValueError(
                f"Judge harness {harness.harness_id!r} requires at least "
                f"{harness.required_rounds} rounds; configured minimum is "
                f"{self.runtime_config.effective_min_num_rounds()}"
            )
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
        return self._rollout_variable_rounds(instances_repeated=instances_repeated, step_seed=step_seed)

    def _postprocess_visible_texts(
        self,
        *,
        instances: list[TaskInstance],
        texts: list[str],
        strip_stop_sentinel: bool = False,
        preserve_sampled_text: bool = False,
    ) -> tuple[list[str], list[dict]]:
        postprocess = getattr(self.task, "postprocess_visible_text", None)
        visible_texts: list[str] = []
        metrics: list[dict] = []
        for inst, text in zip(instances, texts, strict=True):
            clean_text = text if preserve_sampled_text else _strip_think_blocks(text)
            if strip_stop_sentinel:
                clean_text = clean_text.split(_BASE_STOP_SENTINEL, 1)[0].strip()
            if callable(postprocess) and not preserve_sampled_text:
                processed = postprocess(inst=inst, text=clean_text)
                if isinstance(processed, tuple):
                    visible_text, visible_metrics = processed
                else:
                    visible_text, visible_metrics = processed, {}
                visible_texts.append(str(visible_text))
                metrics.append(dict(visible_metrics))
            else:
                visible_texts.append(clean_text)
                metrics.append({"semantic_post_truncation_applied": False} if preserve_sampled_text else {})
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
        if round_idx < 0:
            raise ValueError(f"Invalid round_num={round_num}")
        return self.runtime_config.round_adapter_names[
            min(round_idx, len(self.runtime_config.round_adapter_names) - 1)
        ]

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
        elif round_num == 2:
            if self.debate_config.max_tokens_r2 is not None:
                return int(self.debate_config.max_tokens_r2)
            if self.debate_config.max_tokens_r23 is not None:
                return int(self.debate_config.max_tokens_r23)
        elif round_num >= 3:
            if self.debate_config.max_tokens_r3 is not None:
                return int(self.debate_config.max_tokens_r3)
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
                    system_text=extension.system_text,
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
                    system_text=extension.system_text,
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
                "- Reinforce why your fixed answer better satisfies the constitution.\n"
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

    def _base_response_continuation_tokens(
        self,
        *,
        inst: TaskInstance,
        round_num: int,
        opponent_response: str,
    ) -> list[int]:
        if round_num < 3:
            raise ValueError("Repeated response continuation starts at round 3")
        extension = self._task_base_text_debate_extension(
            inst=inst,
            opponent_round=round_num - 1,
            opponent_answer=opponent_response,
        )
        if extension is not None:
            system_text = extension.system_text
            user_text = extension.user_text
            assistant_prefill = extension.assistant_prefill
        else:
            task_template = self.task.debate_r3_user_template()
            if task_template is None:
                user_text = (
                    f"Constitution:\n{self.task.judge_constitution_text(inst=inst)}\n\n"
                    f"Opponent Round {round_num - 1} response:\n{opponent_response}\n\n"
                    f"Write exactly 3 short numbered points for your Round {round_num} response.\n"
                    "- Respond to the opponent's latest criticisms.\n"
                    "- Reinforce why your fixed Round 1 answer better satisfies the constitution.\n"
                    "- Do not revise your Round 1 answer.\n"
                    f"- After point 3, immediately output {_BASE_STOP_SENTINEL} and nothing else.\n"
                )
            else:
                user_text = task_template.format(
                    round_num=round_num,
                    opponent_round=round_num - 1,
                    opponent_response=opponent_response,
                )
                user_text += (
                    f"\n\nWrite exactly 3 short numbered points. After point 3, immediately output "
                    f"{_BASE_STOP_SENTINEL} and nothing else.\n"
                )
            system_text = _base_debate_system_text(str(round_num))
            assistant_prefill = self.runtime_config.base_r3_prefill
        return self._encode_base_text(
            "\n\n"
            + _base_text_prompt(
                system_text=system_text,
                user_text=user_text,
                assistant_prefill=assistant_prefill,
            )
        )

    def _chat_continuation_parts(self, *, round_num: int) -> tuple[list[int], list[int]]:
        if round_num == 2:
            template = self.task.debate_r2_user_template() or load_prompt("debate/r2_token_template.md")
            marker = "{opponent_r1}"
        elif round_num >= 3:
            template = self.task.debate_r3_user_template() or load_prompt("debate/r3_token_template.md")
            marker = "{opponent_response}"
            template = template.format(
                round_num=round_num,
                opponent_round=round_num - 1,
                opponent_response=marker,
            )
        else:
            raise ValueError("Continuation rounds start at round 2")
        pre, post = _split_template(template, marker)
        return get_chat_adapter(self.tokenizer).build_user_continuation_tokens(
            user_pre=pre,
            user_post=post,
            enable_thinking=self.debate_config.enable_thinking,
        )

    def _target_round_counts(self, *, n_debates: int, step_seed: int | None) -> list[int]:
        minimum = self.runtime_config.effective_min_num_rounds()
        maximum = self.runtime_config.num_rounds
        if minimum == maximum:
            return [maximum] * n_debates
        rng = random.Random(f"{step_seed}:debate_round_counts")
        return [rng.randint(minimum, maximum) for _ in range(n_debates)]

    def _sample_many(self, *, prompt_tokens_list: list[list[int]], round_num: int, step_seed: int | None, stop_token_ids: list[int], max_tokens: int, temperature: float, adapter_name: str | None = None) -> list[tuple[list[int], list[float], str, dict]]:
        requests = []
        use_real_debate_stop = self.runtime_config.stop_on_concluded and round_num >= 2
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
        allowed_token_ids = self._judge_allowed_token_ids()
        candidate_logprob_token_ids = (
            allowed_token_ids
            if self.runtime_config.judge_score_mode == "order_sym_soft_logit"
            else ()
        )
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
                    allowed_token_ids=allowed_token_ids,
                    candidate_logprob_token_ids=candidate_logprob_token_ids,
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
            for result in results:
                raw = dict(result.raw)
                if result.candidate_logprobs:
                    raw["candidate_logprobs"] = result.candidate_logprobs
                out.append(
                    (result.completion_token_ids, result.completion_logprobs, result.text, raw)
                )
        return out

    def _judge_allowed_token_ids(self) -> tuple[int, ...]:
        if not self.runtime_config.judge_constrain_single_token:
            return ()
        if self.runtime_config.judge_label_token_contract != JUDGE_LABEL_TOKEN_CONTRACT_NONE:
            contract = resolve_judge_label_token_contract(
                tokenizer=self.tokenizer,
                contract_name=self.runtime_config.judge_label_token_contract,
            )
            return contract.allowed_token_ids
        harness = get_judge_harness(self.runtime_config.judge_harness_id)
        token_ids: list[int] = []
        for candidate in ("A", " A", "B", " B"):
            encoded = list(self.tokenizer.encode(candidate, add_special_tokens=False))
            if len(encoded) != 1:
                continue
            if harness.parse_verdict(self.tokenizer.decode(encoded)) not in ("A", "B"):
                continue
            if encoded[0] not in token_ids:
                token_ids.append(encoded[0])
        if len(token_ids) < 2:
            raise ValueError(
                "Single-token judge constraint could not resolve valid A/B token ids"
            )
        return tuple(token_ids)

    def _judge_prompts(
        self,
        *,
        transcripts: list[JudgeTranscript],
        reverse_order: bool = False,
    ) -> list[list[int]]:
        prompts: list[list[int]] = []
        for transcript in transcripts:
            if reverse_order:
                transcript = transcript.swapped()
            prompts.append(self._encode_judge_transcript(transcript))
        return prompts

    def _encode_judge_transcript(self, transcript: JudgeTranscript) -> list[int]:
        harness = get_judge_harness(self.runtime_config.judge_harness_id)
        rendered = harness.render_checked(
            transcript=transcript,
            base_system_text=self.debate_config.system_judge,
        )
        if rendered.raw_text is not None:
            prompt_tokens = list(self.tokenizer.encode(rendered.raw_text, add_special_tokens=False))
            if self.runtime_config.judge_label_token_contract == LFM25_OPENBOOKQA_SPACED_AB_V1:
                contract = resolve_judge_label_token_contract(
                    tokenizer=self.tokenizer,
                    contract_name=self.runtime_config.judge_label_token_contract,
                )
                # Harness-authored suffix is invariant across examples. Validate
                # one real rendered boundary per distinct suffix and avoid two
                # extra full-prompt tokenizations for every later debate.
                cache_key = (contract.name, rendered.raw_text[-128:])
                validated = getattr(self, "_validated_judge_label_boundaries", set())
                if cache_key not in validated:
                    validate_judge_prompt_label_boundary(
                        tokenizer=self.tokenizer,
                        prompt_text=rendered.raw_text,
                        prompt_token_ids=prompt_tokens,
                        contract=contract,
                    )
                    validated.add(cache_key)
                    self._validated_judge_label_boundaries = validated
            return prompt_tokens
        return get_chat_adapter(self.tokenizer).encode_messages(
            list(rendered.messages),
            add_generation_prompt=True,
            enable_thinking=False,
        )

    def _judge_max_tokens(self) -> int:
        if self.runtime_config.judge_max_tokens > 0:
            return self.runtime_config.judge_max_tokens
        harness_default = get_judge_harness(
            self.runtime_config.judge_harness_id
        ).default_max_tokens
        if harness_default is not None:
            return harness_default
        return self._max_tokens_for_round(round_num=1)

    def _rollout_variable_rounds(
        self,
        *,
        instances_repeated: list[TaskInstance],
        step_seed: int | None,
    ) -> DebateRolloutOutput:
        use_base_text_prefill = self._use_base_text_prefill()
        n_agents = len(instances_repeated)
        n_debates = n_agents // 2
        inst_pairs = [
            (instances_repeated[2 * index], instances_repeated[2 * index + 1])
            for index in range(n_debates)
        ]
        target_round_counts = self._target_round_counts(
            n_debates=n_debates,
            step_seed=step_seed,
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
        fixed_r1_builder = getattr(self.task, "fixed_r1_completion_text", None)
        if callable(fixed_r1_builder):
            r1_results = []
            for inst in instances_repeated:
                text = str(fixed_r1_builder(inst=inst))
                tokens = list(self.tokenizer.encode(text, add_special_tokens=False))
                if not tokens:
                    raise ValueError("Fixed R1 completion must contain at least one token")
                r1_results.append(
                    (tokens, [0.0] * len(tokens), text, {"fixed_r1": True, "sampled": False})
                )
        else:
            r1_results = self._sample_many(
                prompt_tokens_list=base_r1_prompt_tokens,
                round_num=1,
                step_seed=step_seed,
                stop_token_ids=stop_token_ids,
                max_tokens=self._max_tokens_for_round(round_num=1),
                temperature=float(self.debate_config.temperature),
            )

        r1_tokens = [result[0] for result in r1_results]
        r1_lps = [result[1] for result in r1_results]
        r1_text = [result[2] for result in r1_results]
        r1_raw = [result[3] for result in r1_results]
        r1_visible_text, r1_visible_metrics = self._postprocess_visible_texts(
            instances=instances_repeated,
            texts=r1_text,
        )
        r1_solutions = [
            text if callable(fixed_r1_builder) else extract_solution(text)
            for text in r1_visible_text
        ]
        task_rewards: list[float] = []
        task_reward_metrics: list[dict] = []
        for inst, completion in zip(instances_repeated, r1_tokens, strict=True):
            reward = self.task.compute_reward(
                inst=inst,
                completion_tokens=completion,
                tokenizer=self.tokenizer,
            )
            task_rewards.append(float(reward.reward))
            task_reward_metrics.append(dict(reward.metrics))

        transitions: list[list[Transition]] = []
        visible_rounds: list[list[str]] = []
        trajectory_metrics: list[dict] = []
        for index, inst in enumerate(instances_repeated):
            transitions.append([
                Transition(
                    prompt_tokens=base_r1_prompt_tokens[index],
                    completion_tokens=r1_tokens[index],
                    completion_logprobs=r1_lps[index],
                    round_num=1,
                    metrics={
                        "solution": r1_solutions[index],
                        "instance_id": inst.instance_id,
                        "visible_text_metrics": r1_visible_metrics[index],
                    },
                    raw_response=r1_raw[index],
                )
            ])
            visible_rounds.append([r1_visible_text[index]])
            trajectory_metrics.append({
                "r1": r1_text[index],
                "instance_id": inst.instance_id,
                "task_reward": task_rewards[index],
                "task_reward_metrics": task_reward_metrics[index],
            })

        for round_num in range(2, self.runtime_config.num_rounds + 1):
            active_debates = [
                index
                for index, target in enumerate(target_round_counts)
                if target >= round_num
            ]
            if not active_debates:
                continue
            active_agents = [
                agent_index
                for debate_index in active_debates
                for agent_index in (2 * debate_index, 2 * debate_index + 1)
            ]
            chat_prefix: list[int] = []
            chat_suffix: list[int] = []
            if not use_base_text_prefill:
                chat_prefix, chat_suffix = self._chat_continuation_parts(round_num=round_num)
            prompt_tokens_list: list[list[int]] = []
            prefills: list[str] = []
            for agent_index in active_agents:
                opponent_index = agent_index + 1 if agent_index % 2 == 0 else agent_index - 1
                previous = transitions[agent_index][-1]
                opponent_response = visible_rounds[opponent_index][-1]
                if use_base_text_prefill:
                    fallback = (
                        self.runtime_config.base_r2_prefill
                        if round_num == 2
                        else self.runtime_config.base_r3_prefill
                    )
                    prefill = self._base_text_debate_prefill(
                        inst=instances_repeated[agent_index],
                        opponent_round=round_num - 1,
                        opponent_answer=opponent_response,
                        fallback=fallback,
                    )
                    prefills.append(prefill)
                    continuation = (
                        self._base_r2_continuation_tokens(
                            inst=instances_repeated[agent_index],
                            own_r1=visible_rounds[agent_index][0],
                            opponent_r1=visible_rounds[opponent_index][0],
                        )
                        if round_num == 2
                        else self._base_response_continuation_tokens(
                            inst=instances_repeated[agent_index],
                            round_num=round_num,
                            opponent_response=opponent_response,
                        )
                    )
                else:
                    continuation = (
                        chat_prefix
                        + list(self.tokenizer.encode(opponent_response, add_special_tokens=False))
                        + chat_suffix
                    )
                prompt_tokens_list.append(
                    previous.prompt_tokens + previous.completion_tokens + continuation
                )

            results = self._sample_many(
                prompt_tokens_list=prompt_tokens_list,
                round_num=round_num,
                step_seed=step_seed,
                stop_token_ids=stop_token_ids,
                max_tokens=self._max_tokens_for_round(round_num=round_num),
                temperature=float(self.debate_config.temperature),
            )
            raw_texts = [result[2] for result in results]
            visible_texts, visible_metrics = self._postprocess_visible_texts(
                instances=[instances_repeated[index] for index in active_agents],
                texts=raw_texts,
                strip_stop_sentinel=self.runtime_config.stop_on_concluded,
                preserve_sampled_text=(
                    use_base_text_prefill and not self.runtime_config.stop_on_concluded
                ),
            )
            argument_texts = (
                [
                    prefill + text
                    for prefill, text in zip(prefills, visible_texts, strict=True)
                ]
                if use_base_text_prefill
                else visible_texts
            )
            for local_index, agent_index in enumerate(active_agents):
                completion_tokens, completion_lps, raw_text, raw_response = results[local_index]
                transitions[agent_index].append(
                    Transition(
                        prompt_tokens=prompt_tokens_list[local_index],
                        completion_tokens=completion_tokens,
                        completion_logprobs=completion_lps,
                        round_num=round_num,
                        metrics={"visible_text_metrics": visible_metrics[local_index]},
                        raw_response=raw_response,
                    )
                )
                visible_rounds[agent_index].append(argument_texts[local_index])
                trajectory_metrics[agent_index][f"r{round_num}"] = argument_texts[local_index]
                trajectory_metrics[agent_index][f"r{round_num}_completion_raw"] = raw_text

        transcripts = [
            JudgeTranscript(
                question=self.task.judge_context_text(inst=inst_a),
                constitution=self.task.judge_constitution_text(inst=inst_a),
                agent_a=AgentDebateText(rounds=visible_rounds[2 * index]),
                agent_b=AgentDebateText(rounds=visible_rounds[2 * index + 1]),
            )
            for index, (inst_a, _inst_b) in enumerate(inst_pairs)
        ]
        if self.judge_fn is not None:
            judge_outputs = self._run_external_judge_transcripts(transcripts=transcripts)
        else:
            judge_outputs = self._run_llm_judge_transcripts(
                transcripts=transcripts,
                step_seed=step_seed,
            )
        (
            verdicts,
            judge_reasonings,
            judge_prompt_tokens,
            judge_completion_tokens,
            judge_completion_logprobs,
            judge_raw_responses,
            judge_retry_flags,
        ) = judge_outputs

        debates: list[DebateResult] = []
        for index, (inst_a, inst_b) in enumerate(inst_pairs):
            a_index = 2 * index
            b_index = a_index + 1
            debates.append(
                DebateResult(
                    question=self.task.judge_context_text(inst=inst_a),
                    ground_truth=inst_a.payload.get("ground_truth"),
                    trajectory_a=DebateTrajectory(
                        agent="A",
                        transitions=transitions[a_index],
                        frozen_solution=r1_solutions[a_index],
                        metrics=trajectory_metrics[a_index],
                    ),
                    trajectory_b=DebateTrajectory(
                        agent="B",
                        transitions=transitions[b_index],
                        frozen_solution=r1_solutions[b_index],
                        metrics=trajectory_metrics[b_index],
                    ),
                    verdict=verdicts[index],
                    judge_reasoning=judge_reasonings[index],
                    metrics={
                        "token_only_rollout": True,
                        "task": self.task.name,
                        "judge_retry": judge_retry_flags[index],
                        "num_rounds": target_round_counts[index],
                    },
                    judge_prompt_tokens=judge_prompt_tokens[index],
                    judge_completion_tokens=judge_completion_tokens[index],
                    judge_completion_logprobs=judge_completion_logprobs[index],
                    judge_raw_response=judge_raw_responses[index],
                )
            )
        depth_counts = {
            depth: target_round_counts.count(depth)
            for depth in sorted(set(target_round_counts))
        }
        return DebateRolloutOutput(
            debates=debates,
            info_lines=[
                f"Debates={len(debates)} round_range="
                f"{self.runtime_config.effective_min_num_rounds()}-{self.runtime_config.num_rounds} "
                f"depth_counts={depth_counts}"
            ],
        )

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
                    rounds_a=[r1_visible_text[a_idx]],
                    rounds_b=[r1_visible_text[b_idx]],
                )
                verdicts.append(verdict)
                judge_reasonings.append(reasoning)
                judge_prompt_tokens.append([])
                judge_completion_tokens.append([])
                judge_completion_logprobs.append([])
                judge_raw_responses.append({"external_judge": True, "raw_text": reasoning})
        elif self.runtime_config.judge_bidirectional:
            transcripts = [
                JudgeTranscript(
                    question=self.task.judge_context_text(inst=inst_a),
                    constitution=self.task.judge_constitution_text(inst=inst_a),
                    agent_a=AgentDebateText(rounds=[r1_visible_text[2 * idx]]),
                    agent_b=AgentDebateText(rounds=[r1_visible_text[2 * idx + 1]]),
                )
                for idx, (inst_a, _inst_b) in enumerate(inst_pairs)
            ]
            (
                verdicts,
                judge_reasonings,
                judge_prompt_tokens,
                judge_completion_tokens,
                judge_completion_logprobs,
                judge_raw_responses,
                _judge_retry_flags,
            ) = self._run_llm_judge_transcripts(
                transcripts=transcripts,
                step_seed=step_seed,
            )
        else:
            judge_prompt_tokens = []
            for idx, (inst_a, _inst_b) in enumerate(inst_pairs):
                a_idx = 2 * idx
                b_idx = 2 * idx + 1
                judge_prompt_tokens.append(
                    self._encode_judge_transcript(
                        JudgeTranscript(
                            question=self.task.judge_context_text(inst=inst_a),
                            constitution=self.task.judge_constitution_text(inst=inst_a),
                            agent_a=AgentDebateText(
                                r1=r1_visible_text[a_idx], r2="", r3=""
                            ),
                            agent_b=AgentDebateText(
                                r1=r1_visible_text[b_idx], r2="", r3=""
                            ),
                        )
                    )
                )
            judge_results = self._sample_judge_many(
                prompt_tokens_list=judge_prompt_tokens,
                round_num=99,
                step_seed=step_seed,
                stop_token_ids=stop_token_ids,
                max_tokens=self._judge_max_tokens(),
                temperature=float(self.runtime_config.judge_temperature),
            )
            verdicts = []
            judge_reasonings = []
            judge_completion_tokens = []
            judge_completion_logprobs = []
            judge_raw_responses = []
            for idx, (comp, lps, text, raw) in enumerate(judge_results):
                verdict = get_judge_harness(
                    self.runtime_config.judge_harness_id
                ).parse_verdict(text)
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

    def _run_llm_judge_transcripts(
        self,
        *,
        transcripts: list[JudgeTranscript],
        step_seed: int | None,
    ) -> tuple[list[Verdict], list[str], list[list[int]], list[list[int]], list[list[float]], list[dict], list[bool]]:
        forward_prompt_tokens = self._judge_prompts(
            transcripts=transcripts,
        )
        reverse_prompt_tokens = (
            self._judge_prompts(
                transcripts=transcripts,
                reverse_order=True,
            )
            if self.runtime_config.judge_bidirectional
            else []
        )
        prompt_tokens = forward_prompt_tokens + reverse_prompt_tokens
        stop_token_ids = self.task.stop_token_ids(tokenizer=self.tokenizer)
        results = self._sample_judge_many(
            prompt_tokens_list=prompt_tokens,
            round_num=99,
            step_seed=step_seed,
            stop_token_ids=stop_token_ids,
            max_tokens=self._judge_max_tokens(),
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
            verdict = get_judge_harness(
                self.runtime_config.judge_harness_id
            ).parse_verdict(text)
            verdicts.append(verdict)
            reasonings.append(extract_reasoning(text))
            completion_tokens.append(comp)
            completion_logprobs.append(lps)
            raw_responses.append(raw)
            retry_flags.append(False)
            if verdict == "INVALID":
                invalid_indices.append(idx)
        if self.runtime_config.judge_score_mode == "order_sym_soft_logit":
            if not self.runtime_config.judge_bidirectional:
                raise ValueError("order_sym_soft_logit requires bidirectional judge sampling")
            contract = resolve_judge_label_token_contract(
                tokenizer=self.tokenizer,
                contract_name=self.runtime_config.judge_label_token_contract,
            )
            debate_count = len(transcripts)
            final_verdicts: list[Verdict] = []
            final_reasonings: list[str] = []
            final_raw_responses: list[dict] = []
            for debate_idx in range(debate_count):
                forward_rows = raw_responses[debate_idx].get("candidate_logprobs")
                reverse_rows = raw_responses[debate_count + debate_idx].get("candidate_logprobs")
                if not isinstance(forward_rows, list) or len(forward_rows) != 1:
                    raise ValueError("Soft judge requires exactly one forward candidate-logprob row")
                if not isinstance(reverse_rows, list) or len(reverse_rows) != 1:
                    raise ValueError("Soft judge requires exactly one reverse candidate-logprob row")
                forward_row = {int(key): float(value) for key, value in forward_rows[0].items()}
                reverse_row = {int(key): float(value) for key, value in reverse_rows[0].items()}
                soft = order_symmetric_soft_judge_score(
                    forward_candidate_logprobs=forward_row,
                    reverse_candidate_logprobs=reverse_row,
                    contract=contract,
                )
                final_verdict: Verdict = "A" if soft.score >= 0.0 else "B"
                forward_verdict = verdicts[debate_idx]
                reverse_verdict = verdicts[debate_count + debate_idx]
                reverse_mapped: Verdict = (
                    "B" if reverse_verdict == "A" else "A" if reverse_verdict == "B" else "INVALID"
                )
                hard_order_invariant = (
                    forward_verdict in ("A", "B")
                    and reverse_mapped in ("A", "B")
                    and forward_verdict == reverse_mapped
                )
                final_verdicts.append(final_verdict)
                final_reasonings.append(
                    "[ORDER-SYMMETRIZED SOFT LOGIT SCORE]\n"
                    f"s={soft.score:.9g}; z_sym={soft.z_symmetric:.9g}"
                )
                final_raw_responses.append({
                    "bidirectional_judge": True,
                    "soft_judge": True,
                    "judge_score_mode": "order_sym_soft_logit",
                    "judge_label_token_contract": contract.record(),
                    "forward": raw_responses[debate_idx],
                    "reverse": raw_responses[debate_count + debate_idx],
                    "forward_verdict": forward_verdict,
                    "reverse_verdict": reverse_verdict,
                    "reverse_mapped_verdict": reverse_mapped,
                    "order_invariant": hard_order_invariant,
                    "aggregation": "order_sym_soft_logit_no_fallback",
                    "soft_score": soft.record(),
                    "debate_reward_a": soft.score,
                    "debate_reward_b": -soft.score,
                    "zero_sum_residual": soft.score + (-soft.score),
                    "final_verdict": final_verdict,
                    "final_verdict_is_diagnostic_only": True,
                    # Exact behavior-policy turns retained in memory for optional
                    # labeled+JS judge GRPO.  The driver strips these from the
                    # durable step record after assembling training examples.
                    "_training_judge_turns": [
                        {
                            "order": "forward",
                            "verdict": forward_verdict,
                            "prompt_tokens": prompt_tokens[debate_idx],
                            "completion_tokens": completion_tokens[debate_idx],
                            "completion_logprobs": completion_logprobs[debate_idx],
                            "candidate_logprobs": forward_row,
                            "behavior_policy_allowed_token_ids": list(contract.allowed_token_ids),
                        },
                        {
                            "order": "reverse",
                            "verdict": reverse_verdict,
                            "prompt_tokens": prompt_tokens[debate_count + debate_idx],
                            "completion_tokens": completion_tokens[debate_count + debate_idx],
                            "completion_logprobs": completion_logprobs[debate_count + debate_idx],
                            "candidate_logprobs": reverse_row,
                            "behavior_policy_allowed_token_ids": list(contract.allowed_token_ids),
                        },
                    ],
                })
            return (
                final_verdicts,
                final_reasonings,
                forward_prompt_tokens,
                completion_tokens[:debate_count],
                completion_logprobs[:debate_count],
                final_raw_responses,
                [False] * debate_count,
            )
        # A deterministic retry is useful for a frozen single-order judge, but it
        # is a different behavior policy and therefore cannot supply an on-policy
        # judge-GRPO turn. Bidirectional sampling keeps the original invalid
        # completion, treats the pair as incoherent, and trains against that exact
        # sampled action instead of silently replacing it with a temperature-0
        # retry.
        if invalid_indices and not self.runtime_config.judge_bidirectional:
            retry_results = self._sample_judge_many(
                prompt_tokens_list=[prompt_tokens[idx] for idx in invalid_indices],
                round_num=100,
                step_seed=step_seed,
                stop_token_ids=stop_token_ids,
                max_tokens=self._judge_max_tokens(),
                temperature=0.0,
            )
            for debate_idx, (comp, lps, text, raw) in zip(invalid_indices, retry_results, strict=True):
                retry_verdict = get_judge_harness(
                    self.runtime_config.judge_harness_id
                ).parse_verdict(text)
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
                if self.runtime_config.judge_bidirectional:
                    reasonings[debate_idx] = "[JUDGE INVALID AFTER RETRY]"
                elif self.runtime_config.debate_r1_reward == "judge_rejection_task":
                    reasonings[debate_idx] = "[JUDGE INVALID -> DEBATE DROPPED FROM TRAINING]"
                else:
                    verdicts[debate_idx] = _fallback_verdict_from_invalid_judge(
                        step_seed=step_seed,
                        debate_idx=debate_idx,
                    )
                    reasonings[debate_idx] = "[JUDGE INVALID -> RANDOM FALLBACK]"
        if not self.runtime_config.judge_bidirectional:
            return verdicts, reasonings, prompt_tokens, completion_tokens, completion_logprobs, raw_responses, retry_flags

        debate_count = len(transcripts)
        final_verdicts: list[Verdict] = []
        final_reasonings: list[str] = []
        final_raw_responses: list[dict] = []
        final_retry_flags: list[bool] = []
        for debate_idx in range(debate_count):
            forward_verdict = verdicts[debate_idx]
            reverse_verdict = verdicts[debate_count + debate_idx]
            reverse_mapped: Verdict = (
                "B" if reverse_verdict == "A" else "A" if reverse_verdict == "B" else "INVALID"
            )
            order_invariant = (
                forward_verdict in ("A", "B")
                and reverse_mapped in ("A", "B")
                and forward_verdict == reverse_mapped
            )
            both_valid = forward_verdict in ("A", "B") and reverse_mapped in ("A", "B")
            if order_invariant:
                final_verdict: Verdict = forward_verdict
                aggregation = "order_invariant_agreement"
            else:
                final_verdict = _fallback_verdict_from_order_disagreement(
                    step_seed=step_seed, debate_idx=debate_idx,
                )
                aggregation = (
                    "seeded_random_on_order_disagreement"
                    if both_valid
                    else "seeded_random_on_non_invariant_invalid_judgment"
                )
            final_verdicts.append(final_verdict)
            final_reasonings.append(
                "[FORWARD ORDER]\n" + reasonings[debate_idx]
                + "\n[REVERSED ORDER]\n" + reasonings[debate_count + debate_idx]
            )
            final_raw_responses.append({
                "bidirectional_judge": True,
                "forward": raw_responses[debate_idx],
                "reverse": raw_responses[debate_count + debate_idx],
                "forward_verdict": forward_verdict,
                "reverse_verdict": reverse_verdict,
                "reverse_mapped_verdict": reverse_mapped,
                "order_invariant": order_invariant,
                "aggregation": aggregation,
                "final_verdict": final_verdict,
                # Kept in-memory for judge GRPO and removed from step-record JSON by
                # the driver. These are the exact sampled behavior-policy turns,
                # including any successful deterministic retry.
                "_training_judge_turns": [
                    {
                        "order": "forward",
                        "verdict": forward_verdict,
                        "prompt_tokens": prompt_tokens[debate_idx],
                        "completion_tokens": completion_tokens[debate_idx],
                        "completion_logprobs": completion_logprobs[debate_idx],
                    },
                    {
                        "order": "reverse",
                        "verdict": reverse_verdict,
                        "prompt_tokens": prompt_tokens[debate_count + debate_idx],
                        "completion_tokens": completion_tokens[debate_count + debate_idx],
                        "completion_logprobs": completion_logprobs[debate_count + debate_idx],
                    },
                ],
            })
            final_retry_flags.append(retry_flags[debate_idx] or retry_flags[debate_count + debate_idx])
        return (
            final_verdicts, final_reasonings, forward_prompt_tokens,
            completion_tokens[:debate_count], completion_logprobs[:debate_count],
            final_raw_responses, final_retry_flags,
        )
