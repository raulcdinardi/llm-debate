"""Gradient equivalence and saved-activation regression for token chunking."""
import unittest

import torch

from llm_local_rl.trainer import _flat_target_logprobs, _selected_lm_head_token_logprobs


class LmHeadCheckpointingTest(unittest.TestCase):
    def test_loss_entropy_and_gradients(self):
        for dtype in (torch.float32, torch.bfloat16):
            for frozen in (False, True):
                for restricted in (False, True):
                    with self.subTest(dtype=dtype, frozen=frozen, restricted=restricted):
                        torch.manual_seed(42)
                        hidden = torch.randn(2, 5, 7, dtype=dtype, requires_grad=True)
                        head = torch.nn.Linear(7, 31, bias=True, dtype=dtype)
                        head.requires_grad_(not frozen)
                        positions = torch.tensor([[1, 0, 1, 1, 0], [1, 1, 0, 1, 1]], dtype=torch.bool)
                        targets = torch.arange(10).reshape(2, 5) % 3
                        entropy_positions = positions.clone()
                        entropy_positions[0, 0] = False
                        allowed = [(), (0, 1, 2), (0, 1, 2), (), (0, 1, 2), (), (0, 1, 2)] if restricted else None
                        # Nonuniform signed weights expose errors in gradient propagation
                        # and accidental reuse of the last chunk's targets/restrictions.
                        weights = torch.linspace(-0.7, 1.2, 7)
                        actual, entropy = _selected_lm_head_token_logprobs(
                            hidden_states=hidden, lm_head=head, target_ids=targets,
                            selected_positions=positions, entropy_positions=entropy_positions,
                            behavior_temperature=0.8,
                            allowed_token_ids_by_selected_position=allowed,
                            max_positions_per_chunk=3,
                        )
                        params = (hidden,) if frozen else (hidden, head.weight, head.bias)
                        grads = torch.autograd.grad((actual * weights).sum(), params)
                        # Same projection chunks preserve bf16 GEMM shape/numerics.
                        from llm_local_rl.trainer import _lm_head_logprob_chunk
                        hs = hidden[positions]; ts = targets[positions]
                        pieces = []
                        expected_entropy = 0.0
                        for start in range(0, 7, 3):
                            lp, ent = _lm_head_logprob_chunk(
                                hs[start:start+3], head.weight, head.bias, ts[start:start+3],
                                entropy_positions[positions][start:start+3], 0.8,
                                None if allowed is None else allowed[start:start+3], False,
                            )
                            pieces.append(lp); expected_entropy += ent.item()
                        expected = torch.cat(pieces)
                        expected_grads = torch.autograd.grad((expected * weights).sum(), params)
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                        self.assertEqual(entropy, expected_entropy)
                        for a, b in zip(grads, expected_grads):
                            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_no_vocabulary_sized_activations_saved_across_chunks(self):
        torch.manual_seed(7)
        head = torch.nn.Linear(8, 101, bias=False).requires_grad_(False)
        hidden = torch.randn(2, 9, 8, requires_grad=True)
        targets = torch.zeros(2, 9, dtype=torch.long)
        selected = torch.ones(2, 9, dtype=torch.bool)
        saved = []
        def pack(tensor):
            saved.append(tensor)
            return tensor
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
            lp, _ = _selected_lm_head_token_logprobs(
                hidden_states=hidden, lm_head=head, target_ids=targets,
                selected_positions=selected, max_positions_per_chunk=4,
            )
        # Shared parameter storage is allowed; no [chunk_positions, vocabulary]
        # tensor may survive the forward pass, even with many chunks and a tail.
        self.assertFalse(any(t.ndim == 2 and t.shape[-1] == 101 for t in saved))
        lp.sum().backward()
        self.assertTrue(torch.isfinite(hidden.grad).all())
        baseline_saved = []
        with torch.autograd.graph.saved_tensors_hooks(
            lambda t: (baseline_saved.append(t) or t), lambda t: t,
        ):
            baseline = _flat_target_logprobs(
                logits_float=head(hidden).reshape(-1, 101), target_ids=targets.reshape(-1),
            )
        self.assertTrue(any(t.ndim == 2 and t.shape[-1] == 101 for t in baseline_saved))
        baseline.sum().backward()

    def test_no_grad_inference(self):
        head = torch.nn.Linear(5, 11)
        with torch.no_grad():
            values, entropy = _selected_lm_head_token_logprobs(
                hidden_states=torch.randn(1, 3, 5), lm_head=head,
                target_ids=torch.zeros(1, 3, dtype=torch.long),
                selected_positions=torch.ones(1, 3, dtype=torch.bool), max_positions_per_chunk=2,
            )
        self.assertFalse(values.requires_grad)
        self.assertGreater(entropy, 0)


if __name__ == '__main__':
    unittest.main()
