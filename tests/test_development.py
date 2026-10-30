"""Small public developer checks; no external benchmark harness is included."""
import importlib.util
import os
from pathlib import Path
import sys
import unittest

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.generate_fixture import _expected, build_payload


class ReferenceTests(unittest.TestCase):
    def test_bottom_right_causal_alignment(self):
        q = torch.zeros((1, 1, 3, 16))
        k = torch.zeros((1, 1, 2, 16))
        v = torch.zeros_like(k)
        v[:, :, 0] = 2
        v[:, :, 1] = 6
        actual = _expected(q, k, v, causal=True)
        expected = torch.zeros_like(q)
        expected[:, :, 1] = 2
        expected[:, :, 2] = 4
        torch.testing.assert_close(actual, expected)

    def test_padding_and_empty_rows(self):
        q = torch.zeros((2, 1, 3, 16))
        k = torch.zeros((2, 1, 4, 16))
        v = torch.ones_like(k) * 7
        actual = _expected(
            q, k, v, torch.tensor([2, 3]), torch.tensor([4, 0]),
            torch.ones((2, 4), dtype=torch.bool), causal=False)
        expected = torch.zeros_like(q)
        expected[0, :, :2] = 7
        torch.testing.assert_close(actual, expected)

    def test_unequal_negative_scores(self):
        q = torch.zeros((1, 1, 1, 16))
        q[..., 0] = 1
        k = torch.zeros((1, 1, 2, 16))
        k[:, :, 0, 0] = -(2.0 ** 40)
        k[:, :, 1, 0] = -(2.0 ** 41)
        v = torch.zeros_like(k)
        v[:, :, 1] = 1
        actual = _expected(q, k, v, causal=False, scale=1.0)
        torch.testing.assert_close(actual, torch.zeros_like(q))

    def test_demo_is_deterministic(self):
        first, second = build_payload(), build_payload()
        self.assertEqual(set(first), {'prefill', 'ragged-padding'})
        for name, case in first.items():
            self.assertTrue(torch.isfinite(case['expected']).all())
            for key, tensor in case.items():
                self.assertTrue(torch.equal(tensor, second[name][key]))


HAS_TRITON = importlib.util.find_spec('triton') is not None
INTERPRET = os.environ.get('TRITON_INTERPRET') == '1'
CAN_RUN_KERNEL = HAS_TRITON and (INTERPRET or torch.cuda.is_available())


@unittest.skipUnless(CAN_RUN_KERNEL, 'requires Triton and CUDA, or explicit Triton interpreter mode')
class KernelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from fused_attention import fused_attention
        cls.attention = staticmethod(fused_attention)
        cls.device = 'cpu' if INTERPRET else 'cuda'

    def check_attention(self, q, k, v, q_lengths=None, kv_lengths=None,
                        key_padding_mask=None, *, causal, scale=None):
        expected = _expected(q, k, v, q_lengths, kv_lengths, key_padding_mask,
                             causal=causal, scale=scale)
        # Preserve non-dense strides when transferring views to CUDA.
        tensors = [torch.empty_strided(t.shape, t.stride(), dtype=t.dtype,
                                      device=self.device).copy_(t) if t is not None else None
                   for t in (q, k, v, q_lengths, kv_lengths, key_padding_mask)]
        before = [t.clone() if t is not None else None for t in tensors]
        actual = self.attention(*tensors, causal=causal, scale=scale)
        tolerance = 2e-2 if q.dtype == torch.bfloat16 else 5e-3
        torch.testing.assert_close(actual.cpu(), expected, atol=tolerance, rtol=tolerance)
        self.assertEqual(actual.dtype, q.dtype)
        self.assertTrue(actual.is_contiguous())
        self.assertTrue(torch.isfinite(actual).all())
        for tensor, snapshot in zip(tensors, before):
            if tensor is not None:
                self.assertTrue(torch.equal(tensor, snapshot))

    def test_synthetic_demo(self):
        for name, case in build_payload().items():
            with self.subTest(case=name):
                inputs = {k: v for k, v in case.items() if k != 'expected'}
                self.check_attention(**inputs, causal=name == 'prefill',
                                     scale=None if name == 'prefill' else 0.25)

    def test_noncontiguous_ragged_inputs(self):
        generator = torch.Generator().manual_seed(82)
        q = torch.randn((2, 2, 3, 32), generator=generator)[..., ::2]
        k = torch.randn((2, 2, 65, 32), generator=generator)[..., ::2]
        v = torch.randn((2, 2, 65, 32), generator=generator)[..., ::2]
        q_lengths = torch.tensor([3, 0, 2, 0], dtype=torch.int32)[::2]
        kv_lengths = torch.tensor([65, 0, 31, 0], dtype=torch.int64)[::2]
        mask = torch.ones((2, 130), dtype=torch.bool)[:, ::2]
        mask[0, :16] = False
        mask[1] = False
        for causal in (False, True):
            with self.subTest(causal=causal):
                self.check_attention(q, k, v, q_lengths, kv_lengths, mask, causal=causal)

    def test_negative_score_order_all_head_dimensions(self):
        for dimension in (16, 32, 64, 128):
            for causal in (False, True):
                with self.subTest(head_dim=dimension, causal=causal):
                    q = torch.zeros((1, 1, 1, dimension))
                    q[..., 0] = 1
                    k = torch.zeros((1, 1, 2, dimension))
                    k[:, :, 0, 0] = -(2.0 ** 40)
                    k[:, :, 1, 0] = -(2.0 ** 41)
                    v = torch.zeros_like(k)
                    v[:, :, 1] = 1
                    self.check_attention(q, k, v, causal=causal, scale=1.0)


if __name__ == '__main__':
    unittest.main()
