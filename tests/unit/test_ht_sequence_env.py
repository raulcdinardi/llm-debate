from llm_local_rl.debate_tasks import HTSequenceDebateTask
from llm_local_rl.envs import HTSequenceEnv


class TinyTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        _ = add_special_tokens
        return [ord(ch) for ch in text]

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        _ = skip_special_tokens
        return "".join(chr(tok) for tok in token_ids)


class ChatTemplateTinyTokenizer:
    bos_token = "<bos>"
    all_special_tokens = ["<|im_end|>"]
    additional_special_tokens: list[str] = []

    _SPECIAL_TO_ID = {
        "<bos>": 1,
        "<|im_start|>": 2,
        "<|im_end|>": 3,
    }

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        _ = add_special_tokens
        out: list[int] = []
        idx = 0
        while idx < len(text):
            matched = False
            for token, token_id in self._SPECIAL_TO_ID.items():
                if text.startswith(token, idx):
                    out.append(token_id)
                    idx += len(token)
                    matched = True
                    break
            if matched:
                continue
            out.append(ord(text[idx]))
            idx += 1
        return out

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        inverse = {value: key for key, value in self._SPECIAL_TO_ID.items()}
        chars: list[str] = []
        for token_id in token_ids:
            if token_id in inverse:
                if skip_special_tokens:
                    continue
                chars.append(inverse[token_id])
            else:
                chars.append(chr(token_id))
        return "".join(chars)

    def apply_chat_template(
        self,
        messages: list[dict],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool | None = None,
    ) -> str:
        _ = (tokenize, enable_thinking)
        rendered = "".join(
            f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
            for message in messages
        )
        if add_generation_prompt:
            rendered += "<|im_start|>assistant\n"
        return rendered


def test_ht_sequence_reward_counts_h() -> None:
    env = HTSequenceEnv(sequence_len=4, reward_mode="num_h")
    reward, metrics = env.score_completion(
        instance=env.sample_instances(n=1, seed=0)[0],
        tokenizer=TinyTokenizer(),
        completion_token_ids=[ord(ch) for ch in "HHTH"],
    )
    assert reward == 3.0
    assert metrics["parsed_sequence"] == "HHTH"
    assert metrics["parse_success"] == 1.0


def test_ht_sequence_strict_format_rejects_noisy_text() -> None:
    env = HTSequenceEnv(sequence_len=4, reward_mode="num_h", strict_format=True)
    reward, metrics = env.score_completion(
        instance=env.sample_instances(n=1, seed=0)[0],
        tokenizer=TinyTokenizer(),
        completion_token_ids=[ord(ch) for ch in "H H T H"],
    )
    assert reward == 0.0
    assert metrics["parsed_sequence"] == ""
    assert metrics["parse_success"] == 0.0


def test_ht_sequence_strict_format_accepts_exact_comma_space_sequence() -> None:
    env = HTSequenceEnv(sequence_len=4, reward_mode="num_h", strict_format=True)
    reward, metrics = env.score_completion(
        instance=env.sample_instances(n=1, seed=0)[0],
        tokenizer=TinyTokenizer(),
        completion_token_ids=[ord(ch) for ch in "H, T, H, T"],
    )
    assert reward == 2.0
    assert metrics["parsed_sequence"] == "HTHT"
    assert metrics["parse_success"] == 1.0


def test_ht_sequence_reward_counts_transitions() -> None:
    env = HTSequenceEnv(sequence_len=4, reward_mode="num_transitions")
    reward, metrics = env.score_completion(
        instance=env.sample_instances(n=1, seed=0)[0],
        tokenizer=TinyTokenizer(),
        completion_token_ids=[ord(ch) for ch in "HHTT"],
    )
    assert reward == 1.0
    assert metrics["num_transitions"] == 1


def test_ht_sequence_prompt_tokens_match_old_task_path() -> None:
    tokenizer = ChatTemplateTinyTokenizer()
    env = HTSequenceEnv(sequence_len=4)
    task = HTSequenceDebateTask(sequence_len=4)
    env_instance = env.sample_instances(n=1, seed=0)[0]
    task_instance = task.sample_instances(n=1, seed=0)[0]

    env_prompt_tokens = env.build_initial_prompt_token_ids(
        instance=env_instance,
        tokenizer=tokenizer,
        enable_thinking=None,
    )
    task_prompt_tokens = task.build_r1_prompt_tokens(
        inst=task_instance,
        tokenizer=tokenizer,
        enable_thinking=None,
    )

    assert env_prompt_tokens == task_prompt_tokens
    assert env.stop_token_ids(tokenizer=tokenizer) == task.stop_token_ids(tokenizer=tokenizer)


def test_ht_sequence_debate_task_respects_strict_format() -> None:
    tokenizer = TinyTokenizer()
    task = HTSequenceDebateTask(sequence_len=4, strict_format=True)
    inst = task.sample_instances(n=1, seed=0)[0]

    out = task.compute_reward(
        inst=inst,
        tokenizer=tokenizer,
        completion_tokens=[ord(ch) for ch in "H H T H"],
    )
    assert out.reward == 0.0
    assert out.metrics["parsed_sequence"] == ""
    assert out.metrics["parse_success"] == 0.0
