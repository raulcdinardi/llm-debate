from __future__ import annotations

from .coin_task import CoinTask
from .constrained_writing_task import ConstrainedWritingTask
from .confidence_task import ConfidenceTask
from .qa_task import QATask
from .graph_path_task import GraphPathTask
from .secret_word_debate_task import SecretWordDebateTask
from .summary_task import SummaryTask
from .task_types import TaskInstance, TaskReward, TaskSpec

__all__ = [
    "CoinTask",
    "ConstrainedWritingTask",
    "ConfidenceTask",
    "QATask",
    "GraphPathTask",
    "SecretWordDebateTask",
    "SummaryTask",
    "TaskInstance",
    "TaskReward",
    "TaskSpec",
]
