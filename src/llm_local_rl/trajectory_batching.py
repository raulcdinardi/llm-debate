"""Union round loss positions on one causal history, without changing rewards."""

from dataclasses import replace

from llm_local_rl.types import TrainExample


def merge_trajectory_examples(examples: list[TrainExample]) -> TrainExample:
    """The caller must supply exactly one trajectory and one adapter's rows.

    Context can be reused only when every shorter history is an exact prefix of
    the longest. Matching substrings or re-rendered chat histories are unsafe:
    their behavior logprobs were conditioned on different tokens.
    """
    if not examples:
        raise ValueError("Cannot merge an empty trajectory")
    if len(examples) == 1:
        return examples[0]
    longest = max(examples, key=lambda row: len(row.input_ids))
    full_tokens = longest.input_ids + longest.target_ids[-1:]
    n = len(longest.input_ids)
    mask = [0] * n
    old_logprobs = [0.0] * n
    advantages = [0.0] * n
    projections = []
    round_nums = []
    for row in examples:
        if row.adapter_name != longest.adapter_name:
            raise ValueError("Cannot merge different adapters")
        tokens = row.input_ids + row.target_ids[-1:]
        if tokens != full_tokens[:len(tokens)] or row.target_ids != tokens[1:]:
            raise ValueError("Cannot merge rounds: exact token-history prefix violated")
        for index, (loss, behavior, logprob, advantage) in enumerate(zip(
            row.loss_mask, row.behavior_logprob_mask, row.old_logprobs,
            row.advantages, strict=True,
        )):
            if not (loss and behavior):
                if advantage != 0.0:
                    raise ValueError("Nonzero advantage outside sampled loss positions")
                continue
            if mask[index]:
                raise ValueError("Duplicate sampled loss position across round projections")
            mask[index] = 1
            old_logprobs[index] = logprob
            advantages[index] = advantage
        projections.append(dict(row.metadata))
        round_nums.extend(row.metadata["round_nums"] if "round_nums" in row.metadata
                          else [row.metadata["round_num"]])
    return replace(
        longest,
        loss_mask=mask,
        behavior_logprob_mask=list(mask),
        old_logprobs=old_logprobs,
        advantages=advantages,
        metadata={
            **longest.metadata,
            "reason": "same_adapter_trajectory_mask_union",
            "round_nums": sorted(round_nums),
            "rounds_merged": len(round_nums),
            "round_projections": projections,
            "normalization_unit": "agent_trajectory_per_adapter",
        },
    )
