from __future__ import annotations
from dataclasses import dataclass, field
import random
from typing import Any
from llm_local_rl.mmlu_pro_pairwise import MMLUProPairwiseDebateTask
from llm_local_rl.task_types import TaskInstance, TaskReward

CONSTITUTIONS = {
    'helpsteer3': 'Choose the completion the user would prefer.',
    'openbookqa': 'Prefer the answer that is correct.',
}
DATASETS = tuple(CONSTITUTIONS)

@dataclass(frozen=True)
class MixedLabelPairwiseDebateTask(MMLUProPairwiseDebateTask):
    """Exact 3:1 human-preference/correctness fixed-answer warmup."""
    name: str = 'mixed_label_pairwise'
    _by_dataset: dict[str, tuple[dict[str, Any], ...]] = field(init=False, repr=False)

    def __post_init__(self):
        super().__post_init__()
        buckets = {key: [] for key in DATASETS}; seen = set()
        for row in self._rows:
            if row['dataset'] not in buckets: raise ValueError('Unknown source')
            if row['question_id'] in seen: raise ValueError('Duplicate pair identity')
            if row['correct_answer'].strip() == row['wrong_answer'].strip(): raise ValueError('Identical responses')
            seen.add(row['question_id']); buckets[row['dataset']].append(row)
        if any(not x for x in buckets.values()): raise ValueError('Both sources are required')
        object.__setattr__(self, '_by_dataset', {k: tuple(v) for k,v in buckets.items()})

    def sample_instances(self, *, n: int, seed: int | None):
        if n < 0 or n % 4: raise ValueError('Exact 75/25 mixture requires n divisible by four')
        rng = random.Random(seed); result = []
        for dataset, count in [('helpsteer3', 3*n//4), ('openbookqa', n//4)]:
            if count > len(self._by_dataset[dataset]): raise ValueError('Insufficient unique pairs per batch')
            result.extend(TaskInstance(instance_id='preference_pair_'+r['question_id'],payload=dict(r))
                          for r in rng.sample(self._by_dataset[dataset], count))
        rng.shuffle(result)
        return result

    def judge_constitution_text(self, *, inst):
        return CONSTITUTIONS[inst.payload['dataset']]

    def r1_context_text(self, *, inst):
        # The fixed response stays byte-identical; criterion is visible in policy history.
        return str(inst.payload['question'])+'\n\nEvaluation constitution:\n'+self.judge_constitution_text(inst=inst)

    def compute_reward(self, *, inst, completion_tokens, tokenizer):
        result = super().compute_reward(inst=inst, completion_tokens=completion_tokens, tokenizer=tokenizer)
        return TaskReward(reward=result.reward, metrics={**result.metrics,'dataset':inst.payload['dataset'],
            'label_semantics':'human_preference' if inst.payload['dataset']=='helpsteer3' else 'correctness'})
