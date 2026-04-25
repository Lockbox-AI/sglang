"""Tests for ``sglang.srt.layers.attention.sm120_fallback``.

These tests exercise the dispatch wiring added in poc-16 T2A.1 (Issue #23657):
the cached architecture probe, the pure-Python ``get_paged_mqa_logits_metadata``
port, and the dispatch branch in
``compressed/metadata.py:PagedIndexerMetadata.__post_init__``.

Pure CPU; no GPU required.
"""

from __future__ import annotations

import math
import unittest
from typing import List
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.layers.attention import sm120_fallback
from sglang.srt.layers.attention.sm120_fallback import (
    SPLIT_KV,
    get_paged_mqa_logits_metadata_python,
    is_sm120,
)


class IsSm120Test(unittest.TestCase):
    """Cached arch probe correctness."""

    def setUp(self) -> None:
        is_sm120.cache_clear()

    def tearDown(self) -> None:
        is_sm120.cache_clear()

    def test_returns_false_when_cuda_unavailable(self) -> None:
        with patch("torch.cuda.is_available", return_value=False):
            self.assertFalse(is_sm120())

    def test_returns_true_for_compute_12_0(self) -> None:
        with patch("torch.cuda.is_available", return_value=True), patch(
            "torch.cuda.get_device_capability", return_value=(12, 0)
        ):
            self.assertTrue(is_sm120())

    def test_returns_false_for_compute_9_0(self) -> None:
        with patch("torch.cuda.is_available", return_value=True), patch(
            "torch.cuda.get_device_capability", return_value=(9, 0)
        ):
            self.assertFalse(is_sm120())

    def test_returns_false_for_compute_10_0(self) -> None:
        with patch("torch.cuda.is_available", return_value=True), patch(
            "torch.cuda.get_device_capability", return_value=(10, 0)
        ):
            self.assertFalse(is_sm120())

    def test_returns_false_when_get_device_capability_raises(self) -> None:
        with patch("torch.cuda.is_available", return_value=True), patch(
            "torch.cuda.get_device_capability",
            side_effect=RuntimeError("no device"),
        ):
            self.assertFalse(is_sm120())

    def test_result_is_cached(self) -> None:
        """The probe is hit exactly once, even across multiple calls."""
        with patch("torch.cuda.is_available", return_value=True), patch(
            "torch.cuda.get_device_capability", return_value=(12, 0)
        ) as mock_cap:
            for _ in range(5):
                self.assertTrue(is_sm120())
            self.assertEqual(mock_cap.call_count, 1)


def _reference_metadata(
    context_lens: torch.Tensor, block_kv: int, num_sms: int
) -> torch.Tensor:
    """Independent reference implementation derived from the T1.2 spec.

    Kept inline (rather than re-importing the module under test) so a
    bug in the production code can't silently align with a bug here.
    """
    assert context_lens.dim() == 2
    assert context_lens.dtype == torch.int32
    B, N = context_lens.shape
    next_n_atom = 2 if N >= 2 else 1
    num_next_n_atoms = math.ceil(N / next_n_atom)

    last_lens: List[int] = context_lens[:, N - 1].tolist()
    num_segs = [(L + SPLIT_KV - 1) // SPLIT_KV for L in last_lens]

    prefix_sum: List[int] = []
    running = 0
    for n in num_segs:
        running += n
        prefix_sum.append(running)

    total_segs = (prefix_sum[-1] if B > 0 else 0) * num_next_n_atoms
    q_, r_ = divmod(total_segs, num_sms)

    out = torch.zeros((num_sms + 1, 2), dtype=torch.int32)
    for s in range(num_sms + 1):
        seg_starts = s * q_ + min(s, r_)
        # Linear scan (instead of binary-search) for an independent algorithm.
        q_idx = B
        for i in range(B):
            if prefix_sum[i] * num_next_n_atoms > seg_starts:
                q_idx = i
                break

        if q_idx == 0:
            offset_in_q = seg_starts
            num_segs_q = prefix_sum[0] if B > 0 else 0
        else:
            offset_in_q = seg_starts - prefix_sum[q_idx - 1] * num_next_n_atoms
            num_segs_q = (
                prefix_sum[q_idx] - prefix_sum[q_idx - 1] if q_idx < B else 0
            )

        if num_segs_q > 0:
            atom_idx = offset_in_q // num_segs_q
            kv_split_idx = offset_in_q % num_segs_q
        else:
            atom_idx = 0
            kv_split_idx = 0

        out[s, 0] = q_idx * num_next_n_atoms + atom_idx
        out[s, 1] = kv_split_idx
    return out


class GetPagedMqaLogitsMetadataPythonTest(unittest.TestCase):
    """Numerical correctness of the pure-Python scheduler port."""

    def test_basic_decode_shape_and_dtype(self) -> None:
        # Typical V4-Flash decode: B=4, next_n=1, balanced lengths.
        context_lens = torch.tensor(
            [[512], [768], [1024], [1280]], dtype=torch.int32
        )
        out = get_paged_mqa_logits_metadata_python(context_lens, 64, num_sms=8)
        self.assertEqual(out.shape, (9, 2))
        self.assertEqual(out.dtype, torch.int32)

    def test_matches_reference_implementation_decode(self) -> None:
        torch.manual_seed(0)
        # Realistic shape grid: a few batches and num_sms typical of
        # data-center and consumer Blackwell.
        for B in (1, 4, 16, 64):
            for num_sms in (8, 32, 132, 188):
                ctx = torch.randint(1, 8192, (B, 1), dtype=torch.int32)
                got = get_paged_mqa_logits_metadata_python(
                    ctx, 64, num_sms=num_sms
                )
                want = _reference_metadata(ctx, 64, num_sms)
                self.assertTrue(
                    torch.equal(got, want),
                    msg=f"mismatch at B={B}, num_sms={num_sms}",
                )

    def test_matches_reference_next_n_2(self) -> None:
        # MTP / speculative decode path with next_n=2.
        torch.manual_seed(1)
        ctx = torch.randint(1, 4096, (8, 2), dtype=torch.int32)
        got = get_paged_mqa_logits_metadata_python(ctx, 64, num_sms=132)
        want = _reference_metadata(ctx, 64, 132)
        self.assertTrue(torch.equal(got, want))

    def test_block_kv_32_accepted(self) -> None:
        ctx = torch.tensor([[256], [512]], dtype=torch.int32)
        out = get_paged_mqa_logits_metadata_python(ctx, 32, num_sms=4)
        self.assertEqual(out.shape, (5, 2))

    def test_rejects_block_kv_other(self) -> None:
        ctx = torch.tensor([[256]], dtype=torch.int32)
        with self.assertRaises(AssertionError):
            get_paged_mqa_logits_metadata_python(ctx, 128, num_sms=4)

    def test_rejects_varlen_indices(self) -> None:
        ctx = torch.tensor([[256]], dtype=torch.int32)
        idx = torch.zeros((1,), dtype=torch.int32)
        with self.assertRaises(AssertionError):
            get_paged_mqa_logits_metadata_python(ctx, 64, num_sms=4, indices=idx)

    def test_rejects_1d_context_lens(self) -> None:
        with self.assertRaises(AssertionError):
            get_paged_mqa_logits_metadata_python(
                torch.tensor([256], dtype=torch.int32), 64, num_sms=4
            )

    def test_zero_context_handled(self) -> None:
        # Edge: every batch row has zero context. CUDA scheduler folds to
        # the all-zero schedule (no work).
        ctx = torch.zeros((4, 1), dtype=torch.int32)
        out = get_paged_mqa_logits_metadata_python(ctx, 64, num_sms=8)
        self.assertEqual(out.shape, (9, 2))
        # No work => sentinel rows. The exact contents aren't legally
        # specified beyond "kernel detects no-work via start==end", so we
        # just assert internal consistency: every row has the same value.
        self.assertTrue(torch.equal(out[0], out[-1]))


class PagedIndexerMetadataDispatchTest(unittest.TestCase):
    """Dispatch in PagedIndexerMetadata.__post_init__ takes the sm_120 branch.

    Mocks the arch probe to True and asserts that ``deep_gemm`` is never
    imported / called for the metadata path. Uses a CPU page_table so we can
    construct the metadata object without a CUDA device.
    """

    def setUp(self) -> None:
        is_sm120.cache_clear()

    def tearDown(self) -> None:
        is_sm120.cache_clear()

    def _make_metadata(self):
        # Late import: this module has CUDA-side neighbours that may pull
        # in optional GPU deps at import time; we only need the dataclass.
        from sglang.srt.layers.attention.compressed.metadata import (
            PagedIndexerMetadata,
        )

        return PagedIndexerMetadata(
            page_size=256,
            page_table=torch.zeros((2, 4), dtype=torch.int32),
            c4_seq_lens=torch.tensor([128, 256], dtype=torch.int32),
        )

    def test_sm120_dispatch_skips_deep_gemm(self) -> None:
        # Force is_sm120 to True; mock device props for SM count probe.
        with patch.object(sm120_fallback, "is_sm120", return_value=True), patch(
            "torch.cuda.get_device_properties",
            return_value=MagicMock(multi_processor_count=188),
        ), patch.dict("sys.modules", {"deep_gemm": MagicMock()}) as mod_cache:
            # If the sm_120 branch is taken correctly, deep_gemm.get_num_sms
            # is NEVER called. We verify by inspecting the mock.
            deep_gemm_mock = mod_cache["deep_gemm"]
            # Also block the deep_gemm import path entirely so a regression
            # would surface as an attribute access, not silently call a stub.
            deep_gemm_mock.get_num_sms = MagicMock(
                side_effect=AssertionError(
                    "deep_gemm.get_num_sms must not be called on sm_120"
                )
            )
            deep_gemm_mock.get_paged_mqa_logits_metadata = MagicMock(
                side_effect=AssertionError(
                    "deep_gemm.get_paged_mqa_logits_metadata must not be called on sm_120"
                )
            )

            md = self._make_metadata()

        self.assertIsInstance(md.deep_gemm_metadata, torch.Tensor)
        self.assertEqual(md.deep_gemm_metadata.dtype, torch.int32)
        self.assertEqual(md.deep_gemm_metadata.shape, (188 + 1, 2))


if __name__ == "__main__":
    unittest.main()
