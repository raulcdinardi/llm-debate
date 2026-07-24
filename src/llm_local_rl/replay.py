from __future__ import annotations

from dataclasses import replace

import torch

from llm_local_rl.masking import make_train_example
from llm_local_rl.types import EpisodeTurn, TrainExample


def build_replay_example(
    *,
    prompt_token_ids: list[int],
    completion_token_ids: list[int],
    completion_logprobs: list[float],
    advantage_per_token: float,
) -> TrainExample:
    turn = EpisodeTurn(
        turn_name="response",
        adapter_name="solution",
        prompt_token_ids=prompt_token_ids,
        completion_token_ids=completion_token_ids,
        completion_logprobs=completion_logprobs,
        trainable=True,
    )
    example = make_train_example(turn=turn, advantage_per_token=advantage_per_token)
    # Keep one generated token untrained to exercise the interaction between
    # explicit loss masks and zero-advantage positions.
    advantages = list(example.advantages)
    if advantages:
        advantages[-1] = 0.0
    return replace(example, advantages=advantages)


def target_logprobs_from_model(
    *,
    model,
    input_ids: list[int],
    target_ids: list[int],
    device: str,
) -> torch.Tensor:
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    target_tensor = torch.tensor([target_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_tensor, dtype=torch.long, device=device)
    outputs = model(input_ids=input_tensor, attention_mask=attention_mask)
    logits = outputs.logits.float()
    return torch.log_softmax(logits, dim=-1).gather(
        dim=-1,
        index=target_tensor.unsqueeze(-1),
    ).squeeze(-1).squeeze(0)


def replay_loss_loop(
    *,
    model,
    example: TrainExample,
    device: str,
) -> torch.Tensor:
    token_logprobs = target_logprobs_from_model(
        model=model,
        input_ids=example.input_ids,
        target_ids=example.target_ids,
        device=device,
    )
    old_logprobs = torch.tensor(example.old_logprobs, dtype=torch.float32, device=device)
    advantages = torch.tensor(example.advantages, dtype=torch.float32, device=device)
    loss_mask = torch.tensor(example.loss_mask, dtype=torch.bool, device=device)
    behavior_logprob_mask = torch.tensor(
        example.behavior_logprob_mask,
        dtype=torch.bool,
        device=device,
    )

    loss = torch.tensor(0.0, dtype=torch.float32, device=device)
    for idx in range(token_logprobs.shape[0]):
        if not bool(loss_mask[idx]):
            continue
        if float(advantages[idx].item()) == 0.0:
            continue
        if not bool(behavior_logprob_mask[idx]):
            raise ValueError("A trained token is missing a behavior-policy logprob.")
        ratio = torch.exp(token_logprobs[idx] - old_logprobs[idx])
        loss = loss + (-ratio * advantages[idx])
    return loss


def replay_loss_vectorized(
    *,
    model,
    example: TrainExample,
    device: str,
) -> torch.Tensor:
    token_logprobs = target_logprobs_from_model(
        model=model,
        input_ids=example.input_ids,
        target_ids=example.target_ids,
        device=device,
    )
    old_logprobs = torch.tensor(example.old_logprobs, dtype=torch.float32, device=device)
    advantages = torch.tensor(example.advantages, dtype=torch.float32, device=device)
    loss_mask = torch.tensor(example.loss_mask, dtype=torch.bool, device=device)
    behavior_logprob_mask = torch.tensor(
        example.behavior_logprob_mask,
        dtype=torch.bool,
        device=device,
    )
    trained_positions = loss_mask & (advantages != 0.0)
    if bool((trained_positions & ~behavior_logprob_mask).any().detach().cpu().item()):
        raise ValueError("A trained token is missing a behavior-policy logprob.")
    if not torch.any(trained_positions):
        return torch.tensor(0.0, dtype=torch.float32, device=device)
    ratio = torch.exp(token_logprobs[trained_positions] - old_logprobs[trained_positions])
    return torch.sum(-ratio * advantages[trained_positions])


def collect_trainable_grads(model) -> dict[str, torch.Tensor]:
    grads: dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.grad is None:
            grads[name] = torch.zeros_like(param.detach(), device="cpu", dtype=torch.float32)
            continue
        grads[name] = param.grad.detach().cpu().float().clone()
    return grads
