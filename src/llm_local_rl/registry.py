from __future__ import annotations

from llm_local_rl.config import TrainRunConfig
from llm_local_rl.debate_tasks import CoinDebateTask, HTSequenceDebateTask, ShortStoryDebateTask
from llm_local_rl.envs import CoinFlipEnv, HTSequenceEnv, ShortStoryEnv
from llm_local_rl.episodes import SingleTurnEpisodeBuilder


def build_environment(config: TrainRunConfig):
    if config.rollout.env_name == "ht_sequence":
        return HTSequenceEnv(
            sequence_len=config.sequence_len,
            reward_mode=config.reward_mode,
            strict_format=config.strict_ht_format,
        )
    if config.rollout.env_name == "coin_flip":
        return CoinFlipEnv()
    if config.rollout.env_name in {"short_story", "secret_word"}:
        return ShortStoryEnv()
    raise ValueError(f"Unknown env_name={config.rollout.env_name!r}")


def build_episode_builder(config: TrainRunConfig):
    if config.rollout.mode == "single_turn":
        adapter_name = "shared" if config.adapter_layout == "shared" else "solution"
        return SingleTurnEpisodeBuilder(adapter_name=adapter_name)
    raise ValueError("Episode builders are only used for single_turn mode; debate uses DebateRuntime.")


def build_debate_task(config: TrainRunConfig):
    if config.rollout.env_name == "ht_sequence":
        return HTSequenceDebateTask(
            sequence_len=config.sequence_len,
            reward_mode=config.reward_mode,
            strict_format=config.strict_ht_format,
        )
    if config.rollout.env_name == "coin_flip":
        return CoinDebateTask()
    if config.rollout.env_name in {"short_story", "secret_word"}:
        return ShortStoryDebateTask()
    raise ValueError(f"Unknown env_name={config.rollout.env_name!r} for debate mode.")
