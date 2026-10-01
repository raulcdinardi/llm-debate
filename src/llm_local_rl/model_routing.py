from __future__ import annotations

from typing import Any


JUDGE_ADAPTERS = frozenset({"judge", "judge_shadow"})


def partition_adapters(values: dict) -> tuple[dict, dict]:
    return (
        {name: value for name, value in values.items() if name not in JUDGE_ADAPTERS},
        {name: value for name, value in values.items() if name in JUDGE_ADAPTERS},
    )


class RoutedTrainer:
    """Keep existing objectives on their own backbone and optimizer.

    The driver still commits one rollout transaction. Components are paged in
    sequentially so the smaller judge does not reduce the actor's backward capacity.
    """

    def __init__(self, *, actor: Any, judge: Any) -> None:
        self.actor = actor
        self.judge = judge
        self._active = None
        self.sleep()

    def for_adapter(self, adapter_name: str):
        trainer = self.judge if adapter_name in JUDGE_ADAPTERS else self.actor
        if adapter_name not in trainer.config.adapter_names:
            raise ValueError(f"Adapter {adapter_name!r} is not configured on its backbone")
        return trainer

    def _activate(self, adapter_name: str):
        trainer = self.for_adapter(adapter_name)
        if self._active is not trainer:
            self.sleep()
            trainer.wake_up()
            self._active = trainer
        return trainer

    def wake_up(self) -> None:
        # Selection happens at train_batch; waking both backbones wastes VRAM.
        return None

    def sleep(self) -> None:
        self.actor.sleep()
        self.judge.sleep()
        self._active = None

    def train_batch(self, *, adapter_name: str, **kwargs):
        return self._activate(adapter_name).train_batch(adapter_name=adapter_name, **kwargs)

    def compute_logprobs(self, *, adapter_name: str, **kwargs):
        return self._activate(adapter_name).compute_logprobs(adapter_name=adapter_name, **kwargs)

    def save_adapter(self, *, adapter_name: str, **kwargs):
        return self.for_adapter(adapter_name).save_adapter(adapter_name=adapter_name, **kwargs)

    def load_reference_adapters(self, *, adapter_dirs: dict[str, str]) -> None:
        actor, judge = partition_adapters(adapter_dirs)
        if actor:
            self.actor.load_reference_adapters(adapter_dirs=actor)
        if judge:
            self.judge.load_reference_adapters(adapter_dirs=judge)

    def training_state_dict(self) -> dict:
        return {
            "schema": "routed_trainer_state_v1",
            "actor_model_path": self.actor.config.base_model_path,
            "judge_model_path": self.judge.config.base_model_path,
            "actor": self.actor.training_state_dict(),
            "judge": self.judge.training_state_dict(),
        }

    def load_training_state_dict(self, state: dict) -> None:
        if state["schema"] != "routed_trainer_state_v1":
            raise ValueError("Expected a two-backbone exact-resume state")
        for role in ("actor", "judge"):
            trainer = getattr(self, role)
            if state[role + "_model_path"] != trainer.config.base_model_path:
                raise ValueError(f"Exact-resume {role} backbone mismatch")
        self.actor.load_training_state_dict(state["actor"])
        self.judge.load_training_state_dict(state["judge"])


class RoutedSampler:
    """Use two existing samplers without mixing model-specific token IDs."""

    def __init__(self, *, actor: Any, judge: Any, trainable_adapter_names: set[str]) -> None:
        self.actor = actor
        self.judge = judge
        self.trainable_adapter_names = set(trainable_adapter_names)
        self._active = None

    def _names_for(self, sampler, names):
        return names & JUDGE_ADAPTERS if sampler is self.judge else names - JUDGE_ADAPTERS

    def _sleep_active(self, level):
        if self._active is not None:
            names = self._names_for(self._active, self.trainable_adapter_names)
            if names:
                self._active.unload_adapters(adapter_names=names)
            self._active.sleep(level=level)
            self._active = None

    def _activate(self, sampler):
        if self._active is not sampler:
            self._sleep_active(level=1)
            sampler.wake_up()
            self._active = sampler

    def sample_many(self, requests: list):
        if not requests:
            return []
        is_judge = requests[0].adapter_name in JUDGE_ADAPTERS
        if any((request.adapter_name in JUDGE_ADAPTERS) != is_judge for request in requests):
            raise ValueError("A routed sampling batch must use one backbone")
        sampler = self.judge if is_judge else self.actor
        self._activate(sampler)
        return sampler.sample_many(requests)

    def sample(self, request):
        return self.sample_many([request])[0]

    def set_adapter_paths(self, *, adapter_paths: dict[str, str]) -> None:
        actor, judge = partition_adapters(adapter_paths)
        self.actor.set_adapter_paths(adapter_paths=actor)
        self.judge.set_adapter_paths(adapter_paths=judge)

    def unload_adapters(self, *, adapter_names: set[str]) -> None:
        for sampler in (self.actor, self.judge):
            names = self._names_for(sampler, adapter_names)
            if not names:
                continue
            if sampler is self._active:
                sampler.unload_adapters(adapter_names=names)
            else:
                # Every registered mutable LoRA was evicted before sleep.
                # Do not queue by name: next-step paths may bind it to a new ID.
                if names - self.trainable_adapter_names:
                    raise ValueError("Cannot unload a frozen adapter while its engine is asleep")

    def wake_up(self) -> None:
        # Each actual request activates exactly one engine.
        return None

    def sleep(self, *, level: int = 1) -> None:
        self._sleep_active(level=level)

    def close(self) -> None:
        try:
            self.actor.close()
        finally:
            self.judge.close()
