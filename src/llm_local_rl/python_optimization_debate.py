"""MBPP debate task with the identical baseline R1 prompt and timing reward."""
from llm_local_rl.python_optimization import PythonOptimizationEnv, PREFILL
from llm_local_rl.task_types import BaseTextDebateExtension, TaskReward

class PythonOptimizationDebateTask(PythonOptimizationEnv):
    qwen35_debate_subject = "implementation"

    def __init__(self, *, config_path, prompt_format="qwen35_base_text_prefill"):
        super().__init__(config_path=config_path)
        self.prompt_format = prompt_format
    def r1_context_text(self, *, inst):
        prompt=self.build_initial_prompt(instance=inst)
        return prompt[len('User:\n'):-len('\nAssistant:\n'+PREFILL)]

    def build_r1_prompt_tokens(self, *, inst, tokenizer, enable_thinking=None):
        if getattr(self, "prompt_format", "qwen35_base_text_prefill") == "qwen35_instruct_three_points":
            from llm_local_rl.qwen35_instruct_format import ASSISTANT_HEADER
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": self.r1_context_text(inst=inst)}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
            assert rendered.endswith(ASSISTANT_HEADER)
            # The fixed code prefix makes the first generated fence a closing fence.
            # Prefix tokens are prompt tokens, never actor loss positions.
            return list(tokenizer.encode(rendered + PREFILL, add_special_tokens=False))
        return self.build_initial_prompt_token_ids(instance=inst,tokenizer=tokenizer,enable_thinking=enable_thinking)

    def judge_context_text(self, *, inst):return self.r1_context_text(inst=inst)

    def judge_constitution_text(self, *, inst):
        return ('Prefer the agent whose fixed Python implementation best satisfies the user: '
                'run faster while preserving the canonical behavior and exact function names. '
                'Consider concrete code, edge cases, and the debate evidence. Passing the supplied tests alone '
                'does not establish equivalence. Do not reward merely claiming a speedup. Output only A or B.')

    def compute_reward(self, *, inst, completion_tokens, tokenizer):
        value,metrics=self.score_completion(instance=inst,tokenizer=tokenizer,completion_token_ids=completion_tokens)
        return TaskReward(reward=value,metrics=metrics)

    def compute_rewards(self, *, instances, completion_tokens, tokenizer):
        return [TaskReward(reward=value,metrics=metrics) for value,metrics in self.score_completions(
            instances=instances,tokenizer=tokenizer,completion_token_ids=completion_tokens)]

    def debate_r2_user_template(self):return None
    def debate_r3_user_template(self):return None

    def build_base_text_debate_extension(self, *, inst, opponent_round, opponent_answer):
        assert opponent_round>=1
        first=opponent_round==1
        prefill=("The reasons that my solution is better than my opponent's are:\n1)" if first
                 else "Responding to my opponent's criticism:\n1)")
        instruction=('Explain why your fixed implementation best satisfies the optimization request.' if first
                     else "Respond to the opponent's criticism and make your final case.")
        return BaseTextDebateExtension(
            system_text=(f'You are participating in Round {opponent_round+1} of a debate about Python optimization.\n'
                         'Your Round 1 implementation is fixed and cannot change.\n'
                         'Use concrete code and edge-case evidence. Do not emit <think> tags.'),
            user_text=(f'Constitution:\n{self.judge_constitution_text(inst=inst)}\n\n'
                       f'Opponent Round {opponent_round} answer:\n{opponent_answer}\n\n{instruction}\n\n'
                       'Write exactly 3 numbered points, each at most 30 words. After point 3, immediately output '
                       'CONCLUDED and nothing else.\n'),assistant_prefill=prefill)
