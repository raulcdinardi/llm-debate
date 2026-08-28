from __future__ import annotations

from dataclasses import dataclass, field
import random


@dataclass
class SeededRandomJudge:
    """Stateful, reproducible A/B control judge for debate experiments."""

    seed: int
    _rng: random.Random = field(init=False, repr=False)
    _call_index: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def __call__(
        self,
        question: str,
        constitution: str,
        r1_a: str,
        r1_b: str,
        r2_a: str,
        r2_b: str,
        r3_a: str,
        r3_b: str,
    ) -> tuple[str, str]:
        _ = (question, constitution, r1_a, r1_b, r2_a, r2_b, r3_a, r3_b)
        self._call_index += 1
        verdict = self._rng.choice(("A", "B"))
        return verdict, f"seeded_random_judge seed={self.seed} call={self._call_index} verdict={verdict}"
