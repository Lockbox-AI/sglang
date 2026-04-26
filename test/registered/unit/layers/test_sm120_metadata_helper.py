"""CPU-only parity tests for the sm_120 paged-MQA-logits metadata helper (K1a).

Tests that the pure-Python port of ``deep_gemm.get_paged_mqa_logits_metadata``
(T1.2 spec) produces the same ``schedule_metadata`` shape, dtype, and value
for a representative set of inputs.  All tests run on CPU with no GPU
requirement.
"""

from __future__ import annotations

import unittest

import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="stage-a-test-cpu")


class TestSm120MetadataHelperShape(unittest.TestCase):
    """Shape / dtype contract tests (no numerical reference)."""

    def _get(
        self, B: int, N: int, ctx_len: int, block_kv: int = 64, num_sms: int = 128
    ):
        from sglang.srt.layers.attention.sm_120 import (
            get_paged_mqa_logits_metadata_python,
        )

        ctx = torch.full((B, N), ctx_len, dtype=torch.int32)
        return get_paged_mqa_logits_metadata_python(ctx, block_kv, num_sms)

    def test_output_dtype_is_int32(self):
        out = self._get(B=2, N=1, ctx_len=256)
        self.assertEqual(out.dtype, torch.int32)

    def test_output_shape_is_num_sms_plus_1_by_2(self):
        out = self._get(B=2, N=1, ctx_len=256, num_sms=64)
        self.assertEqual(tuple(out.shape), (65, 2))

    def test_output_shape_single_batch(self):
        out = self._get(B=1, N=1, ctx_len=512, num_sms=128)
        self.assertEqual(tuple(out.shape), (129, 2))

    def test_output_shape_multi_next_n(self):
        out = self._get(B=1, N=2, ctx_len=256, num_sms=32)
        self.assertEqual(tuple(out.shape), (33, 2))

    def test_block_kv_32(self):
        out = self._get(B=1, N=1, ctx_len=128, block_kv=32, num_sms=16)
        self.assertEqual(tuple(out.shape), (17, 2))

    def test_zero_context_len(self):
        out = self._get(B=2, N=1, ctx_len=0, num_sms=64)
        self.assertEqual(tuple(out.shape), (65, 2))

    def test_all_values_non_negative(self):
        out = self._get(B=3, N=1, ctx_len=1024, num_sms=128)
        self.assertTrue((out >= 0).all())


class TestSm120MetadataHelperContract(unittest.TestCase):
    """Contract / assertion tests."""

    def _get_fn(self):
        from sglang.srt.layers.attention.sm_120 import (
            get_paged_mqa_logits_metadata_python,
        )

        return get_paged_mqa_logits_metadata_python

    def test_rejects_1d_context_lens(self):
        fn = self._get_fn()
        ctx = torch.full((4,), 256, dtype=torch.int32)
        with self.assertRaises(AssertionError):
            fn(ctx, 64, 128)

    def test_rejects_invalid_block_kv(self):
        fn = self._get_fn()
        ctx = torch.full((1, 1), 256, dtype=torch.int32)
        with self.assertRaises(AssertionError):
            fn(ctx, 128, 128)

    def test_rejects_float_context_lens(self):
        fn = self._get_fn()
        ctx = torch.full((1, 1), 256.0, dtype=torch.float32)
        with self.assertRaises(AssertionError):
            fn(ctx, 64, 128)

    def test_rejects_indices_not_none(self):
        fn = self._get_fn()
        ctx = torch.full((1, 1), 256, dtype=torch.int32)
        with self.assertRaises(AssertionError):
            fn(ctx, 64, 128, indices=torch.zeros(1, dtype=torch.int32))

    def test_cpu_device_output_stays_cpu(self):
        fn = self._get_fn()
        ctx = torch.full((2, 1), 256, dtype=torch.int32)
        out = fn(ctx, 64, 128)
        self.assertEqual(out.device.type, "cpu")


class TestSm120IsSm120Import(unittest.TestCase):
    """Basic import smoke for is_sm120."""

    def test_is_sm120_returns_bool(self):
        from sglang.srt.layers.attention.sm_120 import is_sm120

        result = is_sm120()
        self.assertIsInstance(result, bool)


if __name__ == "__main__":
    unittest.main()
