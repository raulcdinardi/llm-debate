from __future__ import annotations

from dataclasses import dataclass
import math
import os
from typing import Sequence

from .model_input import ModelInput
from .types import SampleResponse, SampleSequence, SamplingParams


def _encode_stop_sequences(tokenizer, stop: str | Sequence[str] | Sequence[int] | None) -> list[list[int]]:
    if stop is None:
        return []
    if isinstance(stop, str):
        toks = tokenizer.encode(stop, add_special_tokens=False)
        return [list(toks)] if toks else []

    stop_list = list(stop)
    if len(stop_list) == 0:
        return []
    if isinstance(stop_list[0], int):
        for x in stop_list:
            if not isinstance(x, int):
                raise TypeError("SamplingParams.stop must be all ints or all strs.")
        return [list(stop_list)]

    if isinstance(stop_list[0], str):
        for x in stop_list:
            if not isinstance(x, str):
                raise TypeError("SamplingParams.stop must be all ints or all strs.")
        stop_token_seqs: list[list[int]] = []
        for s in stop_list:
            toks = tokenizer.encode(s, add_special_tokens=False)
            if toks:
                stop_token_seqs.append(list(toks))
        return stop_token_seqs

    raise TypeError("SamplingParams.stop must be a str, sequence[str], sequence[int], or None.")


def _debug(msg: str) -> None:
    if os.environ.get("TINKER_MEM_DEBUG") != "1":
        return
    print(msg, flush=True)


def _maybe_trim_stop_suffix(tokens: list[int], logprobs: list[float], stop_seqs: list[list[int]]) -> tuple[list[int], list[float]]:
    for stop_ids in stop_seqs:
        n = len(stop_ids)
        if n == 0:
            continue
        if len(tokens) >= n and tokens[-n:] == stop_ids:
            return tokens[:-n], logprobs[:-n]
    return tokens, logprobs


def _build_min_p_logits_processor(min_p: float):
    if not (0.0 <= min_p <= 1.0):
        raise ValueError(f"min_p must be in [0,1], got {min_p}")
    if min_p <= 0.0:
        return None

    from transformers import LogitsProcessorList
    from transformers.generation.logits_process import LogitsProcessor

    class MinPLogitsProcessor(LogitsProcessor):
        def __init__(self, min_p_value: float) -> None:
            self._min_p = float(min_p_value)

        def __call__(self, input_ids, scores):
            if self._min_p <= 0.0:
                return scores
            threshold = scores.max(dim=-1, keepdim=True).values + math.log(self._min_p)
            return scores.masked_fill(scores < threshold, float("-inf"))

    processors = LogitsProcessorList()
    processors.append(MinPLogitsProcessor(min_p))
    return processors


@dataclass(frozen=True)
class SamplingClient:
    _model: object
    _tokenizer: object
    _device: str

    async def sample_token_batch_async(
        self,
        *,
        prompt_tokens_list: list[list[int]],
        sampling_params: SamplingParams,
    ) -> list[SampleResponse]:
        import torch
        from transformers import GenerationConfig

        tokenizer = self._tokenizer
        model = self._model

        if len(prompt_tokens_list) == 0:
            return []

        pad_token_id = tokenizer.pad_token_id
        assert pad_token_id is not None

        stop_token_seqs = _encode_stop_sequences(tokenizer, sampling_params.stop)
        if len(stop_token_seqs) > 1:
            raise NotImplementedError("tinker-local batched sampling supports at most 1 stop sequence.")
        stop_token_id: int | None = None
        if len(stop_token_seqs) == 1:
            if len(stop_token_seqs[0]) != 1:
                raise NotImplementedError("tinker-local batched sampling requires a single-token stop sequence.")
            stop_token_id = stop_token_seqs[0][0]

        max_new_tokens = sampling_params.max_tokens or 256
        temperature = float(sampling_params.temperature)
        do_sample = temperature > 0

        top_k = int(sampling_params.top_k)
        top_p = float(sampling_params.top_p)
        if not (0.0 < top_p <= 1.0):
            raise ValueError(f"top_p must be in (0,1], got {top_p}")
        if top_k < -1:
            raise ValueError(f"top_k must be -1 (no limit) or >= 0, got {top_k}")
        tf_top_k = 0 if top_k == -1 else top_k
        min_p = float(sampling_params.min_p)
        logits_processor = _build_min_p_logits_processor(min_p)

        # Use an explicit GenerationConfig so the effective sampling distribution matches
        # these params (no hidden model.generation_config defaults).
        gen_config = GenerationConfig(
            do_sample=do_sample,
            temperature=(max(1e-6, temperature) if do_sample else 1.0),
            top_k=tf_top_k,
            top_p=top_p,
            pad_token_id=int(pad_token_id),
            eos_token_id=stop_token_id,
        )
        if bool(gen_config.do_sample) != bool(do_sample):
            raise RuntimeError("GenerationConfig.do_sample mismatch")

        token_budget_env = os.environ.get("TINKER_LOCAL_SAMPLE_TOKEN_BUDGET")
        if token_budget_env is None:
            token_budget = 120_000
        else:
            token_budget = int(token_budget_env)
        if token_budget <= 0:
            raise ValueError(f"TINKER_LOCAL_SAMPLE_TOKEN_BUDGET must be > 0, got {token_budget}")

        # Split into smaller sub-batches to avoid OOM when a single very long prompt sets the
        # max padding length for the entire batch (especially in debate R3).
        order = sorted(range(len(prompt_tokens_list)), key=lambda i: len(prompt_tokens_list[i]), reverse=True)
        batches: list[list[int]] = []
        cur: list[int] = []
        cur_max_len = 0
        for idx in order:
            n = len(prompt_tokens_list[idx])
            new_max_len = max(cur_max_len, n)
            est_len = new_max_len + int(max_new_tokens)
            new_size = len(cur) + 1
            if cur and new_size * est_len > token_budget:
                batches.append(cur)
                cur = [idx]
                cur_max_len = n
                continue
            cur.append(idx)
            cur_max_len = new_max_len
        if cur:
            batches.append(cur)
        _debug(
            f"[sample] prompts={len(prompt_tokens_list)} batches={len(batches)} "
            f"max_new={max_new_tokens} temp={temperature} top_p={top_p} "
            f"top_k={top_k} min_p={min_p} token_budget={token_budget}"
        )

        responses_out: list[SampleResponse | None] = [None] * len(prompt_tokens_list)

        generator = None
        if sampling_params.seed is not None:
            generator = torch.Generator(device=self._device).manual_seed(int(sampling_params.seed))

        model.eval()
        for batch_indices in batches:
            lengths = [len(prompt_tokens_list[i]) for i in batch_indices]
            max_len = max(lengths)
            _debug(
                f"[sample] batch_size={len(batch_indices)} min_len={min(lengths)} "
                f"max_len={max_len} est_tokens={len(batch_indices) * (max_len + max_new_tokens)}"
            )

            input_ids = torch.full(
                (len(batch_indices), max_len),
                fill_value=int(pad_token_id),
                dtype=torch.long,
                device=self._device,
            )
            attention_mask = torch.zeros(
                (len(batch_indices), max_len),
                dtype=torch.long,
                device=self._device,
            )

            # Left-pad for decoder-only models.
            for row_i, orig_i in enumerate(batch_indices):
                toks = prompt_tokens_list[orig_i]
                n = len(toks)
                start = max_len - n
                input_ids[row_i, start:max_len] = torch.tensor(toks, dtype=torch.long, device=self._device)
                attention_mask[row_i, start:max_len] = 1

            with torch.no_grad():
                out = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                generation_config=gen_config,
                generator=generator,
                logits_processor=logits_processor,
                return_dict_in_generate=True,
                output_scores=True,
            )

            if out.scores is None:
                raise RuntimeError("generate() did not return scores; cannot compute logprobs.")
            sequences = out.sequences  # [B, max_len + <=max_new_tokens]

            bsz = len(batch_indices)
            toks_by_row: list[list[int]] = [[] for _ in range(bsz)]
            lps_by_row: list[list[float]] = [[] for _ in range(bsz)]
            done = [False] * bsz

            for step_idx, step_scores in enumerate(out.scores):
                next_tokens = sequences[:, max_len + step_idx]  # [B]
                denom = torch.logsumexp(step_scores, dim=-1)  # [B]
                num = step_scores.gather(dim=-1, index=next_tokens.unsqueeze(-1)).squeeze(-1)  # [B]
                lp_vec = num - denom

                for row_i in range(bsz):
                    if done[row_i]:
                        continue
                    tok = int(next_tokens[row_i].item())
                    if stop_token_id is not None and tok == stop_token_id:
                        done[row_i] = True
                        continue
                    if tok == int(pad_token_id):
                        done[row_i] = True
                        continue
                    toks_by_row[row_i].append(tok)
                    lps_by_row[row_i].append(float(lp_vec[row_i].detach().cpu().item()))

            for row_i, orig_i in enumerate(batch_indices):
                toks, lps = _maybe_trim_stop_suffix(toks_by_row[row_i], lps_by_row[row_i], stop_token_seqs)
                responses_out[orig_i] = SampleResponse(sequences=[SampleSequence(tokens=toks, logprobs=lps)])

            # Ensure large generation tensors are released before training begins.
            del out, sequences
            torch.cuda.empty_cache()

        missing = [i for i, r in enumerate(responses_out) if r is None]
        if missing:
            raise RuntimeError(f"Missing batched sampling outputs for indices: {missing[:10]}")

        return [r for r in responses_out if r is not None]

    async def sample_async(
        self,
        *,
        prompt: ModelInput,
        num_samples: int,
        sampling_params: SamplingParams,
    ) -> SampleResponse:
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList
        from transformers import GenerationConfig

        if num_samples != 1:
            raise NotImplementedError("tinker-local only supports num_samples=1 for now.")

        tokenizer = self._tokenizer
        model = self._model

        prompt_tokens = prompt.to_ints()
        input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=self._device)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=self._device)

        max_new_tokens = sampling_params.max_tokens or 256
        temperature = float(sampling_params.temperature)
        do_sample = temperature > 0

        stop_token_seqs = _encode_stop_sequences(tokenizer, sampling_params.stop)

        class StopOnTokenSequences(StoppingCriteria):
            def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor, **kwargs) -> bool:
                seq = input_ids[0].tolist()
                for stop_ids in stop_token_seqs:
                    n = len(stop_ids)
                    if n == 0:
                        continue
                    if len(seq) >= n and seq[-n:] == stop_ids:
                        return True
                return False

        stopping = StoppingCriteriaList()
        if stop_token_seqs:
            stopping.append(StopOnTokenSequences())

        top_k = int(sampling_params.top_k)
        top_p = float(sampling_params.top_p)
        if not (0.0 < top_p <= 1.0):
            raise ValueError(f"top_p must be in (0,1], got {top_p}")
        if top_k < -1:
            raise ValueError(f"top_k must be -1 (no limit) or >= 0, got {top_k}")
        tf_top_k = 0 if top_k == -1 else top_k
        min_p = float(sampling_params.min_p)
        logits_processor = _build_min_p_logits_processor(min_p)

        gen_config = GenerationConfig(
            do_sample=do_sample,
            temperature=(max(1e-6, temperature) if do_sample else 1.0),
            top_k=tf_top_k,
            top_p=top_p,
            pad_token_id=int(tokenizer.pad_token_id),
        )
        if bool(gen_config.do_sample) != bool(do_sample):
            raise RuntimeError("GenerationConfig.do_sample mismatch")

        model.eval()
        with torch.no_grad():
            generator = None
            if sampling_params.seed is not None:
                generator = torch.Generator(device=self._device).manual_seed(int(sampling_params.seed))
            out = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                stopping_criteria=stopping,
                generation_config=gen_config,
                generator=generator,
                logits_processor=logits_processor,
                return_dict_in_generate=True,
                output_scores=True,
            )

        if out.scores is None:
            raise RuntimeError("generate() did not return scores; cannot compute logprobs.")
        sequences = out.sequences  # [B, prompt+new]
        prompt_len = int(input_ids.shape[1])
        gen_tokens = sequences[0, prompt_len:].tolist()

        logprobs: list[float] = []
        for step_idx in range(min(len(gen_tokens), len(out.scores))):
            tok = int(gen_tokens[step_idx])
            row_scores = out.scores[step_idx][0]
            lp = row_scores[tok] - torch.logsumexp(row_scores, dim=-1)
            logprobs.append(float(lp.detach().cpu().item()))

        gen_tokens, logprobs = _maybe_trim_stop_suffix(gen_tokens, logprobs, stop_token_seqs)
        del out, sequences
        torch.cuda.empty_cache()
        return SampleResponse(sequences=[SampleSequence(tokens=gen_tokens, logprobs=logprobs)])
