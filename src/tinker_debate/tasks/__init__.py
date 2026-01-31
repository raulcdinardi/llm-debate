from __future__ import annotations

from .coin_task import CoinTask
from .confidence_task import ConfidenceTask
from .qa_task import QATask
from .secret_word_debate_task import SecretWordDebateTask
from .summary_task import SummaryTask
from .task_types import TaskInstance, TaskReward, TaskSpec

__all__ = [
    "CoinTask",
    "ConfidenceTask",
    "QATask",
    "SecretWordDebateTask",
    "SummaryTask",
    "TaskInstance",
    "TaskReward",
    "TaskSpec",
]
