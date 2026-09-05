from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class IntegrationAssets:
    base_model_path: str
    adapter_a_path: str
    adapter_b_path: str

    @classmethod
    def from_env(cls) -> "IntegrationAssets":
        base_model_path = os.environ["LLM_LOCAL_RL_BASE_MODEL"]
        adapter_a_path = os.environ["LLM_LOCAL_RL_ADAPTER_A"]
        adapter_b_path = os.environ["LLM_LOCAL_RL_ADAPTER_B"]
        return cls(
            base_model_path=base_model_path,
            adapter_a_path=adapter_a_path,
            adapter_b_path=adapter_b_path,
        )
