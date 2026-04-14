from llm_local_rl.envs import HTSequenceEnv


class TinyTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        _ = add_special_tokens
        return [ord(ch) for ch in text]

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        _ = skip_special_tokens
        return "".join(chr(tok) for tok in token_ids)


def test_ht_sequence_reward_counts_h() -> None:
    env = HTSequenceEnv(sequence_len=4, reward_mode="num_h")
    reward, metrics = env.score_completion(
        instance=env.sample_instances(n=1, seed=0)[0],
        tokenizer=TinyTokenizer(),
        completion_token_ids=[ord(ch) for ch in "H H T H"],
    )
    assert reward == 3.0
    assert metrics["parsed_sequence"] == "HHTH"
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
