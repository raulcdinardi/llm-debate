"""tinker_debate

Minimal debate + Tinker API training utilities.

This package is intentionally small; scripts live in /scripts.
"""

from .debate_types import (
    DebateConfig,
    DebateResult,
    DebateTrajectory,
    TrainingDatum,
    Transition,
    Verdict,
    assemble_training_data,
    compute_training_stats,
)
from .debate_env import (
    DebateRolloutClient,
    extract_reasoning,
    extract_solution,
    extract_verdict,
    mock_judge_random,
    run_debate,
    run_debate_batch,
)
from .tinker_client import TinkerDebateClient

__all__ = [
    "DebateConfig",
    "Verdict",
    "Transition",
    "DebateTrajectory",
    "DebateResult",
    "TrainingDatum",
    "assemble_training_data",
    "compute_training_stats",
    "extract_solution",
    "extract_verdict",
    "extract_reasoning",
    "DebateRolloutClient",
    "mock_judge_random",
    "run_debate",
    "run_debate_batch",
    "TinkerDebateClient",
]
