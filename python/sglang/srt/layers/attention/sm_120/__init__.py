"""sm_120 (RTX Pro 6000 Blackwell / RTX 5090) fallback paths for V4 attention.

This module bypasses upstream kernels that hard-fail on sm_120 hardware,
re-routing them through TileLang / Triton fallbacks that compile cleanly
for compute capability 12.0. This file covers K1a only.

  * **K1a — DeepGEMM ``fp8_paged_mqa_logits`` (Lightning Indexer)**
    DeepGEMM does not support sm_120
    (https://github.com/sgl-project/sglang/issues/23657,
    https://github.com/deepseek-ai/DeepGEMM/issues/236). The metadata
    helper is small enough to run on the host (a few microseconds for
    typical batch shapes), and the FP8 paged MQA logits kernel itself
    already has an in-tree TileLang implementation at
    ``sglang.srt.layers.attention.nsa.tilelang_kernel.tilelang_fp8_paged_mqa_logits``
    that is correct on sm_120.

Public API
----------
Architecture probe:
- :func:`is_sm120`: cached architecture probe.

K1a (Lightning Indexer):
- :func:`get_paged_mqa_logits_metadata_python`: pure-Python port of the
  DeepGEMM scheduler.
- :func:`tilelang_fp8_paged_mqa_logits_sm120`: shape-adapter shim that
  forwards into the in-tree TileLang ``tilelang_fp8_paged_mqa_logits``.
"""

from __future__ import annotations

import functools
import logging
from typing import Any, Optional

import torch

logger = logging.getLogger(__name__)

# DeepGEMM pages the KV cache in ``SPLIT_KV``-token chunks.
SPLIT_KV = 256


@functools.lru_cache(maxsize=1)
def is_sm120() -> bool:
    """Return ``True`` iff the active CUDA device is sm_120 (compute 12.0).

    Cached for the lifetime of the process; tests that need to exercise the
    branch should call ``is_sm120.cache_clear()`` after patching
    ``torch.cuda.get_device_capability``.
    """
    if not torch.cuda.is_available():
        return False
    try:
        major, minor = torch.cuda.get_device_capability()
    except Exception:  # noqa: BLE001
        return False
    return (major, minor) == (12, 0)


def get_paged_mqa_logits_metadata_python(
    context_lens: torch.Tensor,
    block_kv: int,
    num_sms: int,
    indices: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Pure-Python port of ``deep_gemm.get_paged_mqa_logits_metadata``.

    Builds the ``schedule_metadata`` tensor consumed by the SM100/SM90 paged
    MQA logits kernel. Each row ``[s, :]`` records the ``(q_atom_idx,
    kv_split_idx)`` start position for SM ``s``. The TileLang fallback we
    use on sm_120 ignores this tensor (its grid is non-persistent), but the
    Lightning Indexer metadata path still expects a tensor of the right
    shape and dtype.
    """
    assert (
        indices is None
    ), "varlen mode (indices != None) is not used by sglang's V4 decode path"
    assert (
        context_lens.dim() == 2
    ), f"expected 2-D context_lens, got {context_lens.shape}"
    assert (
        context_lens.dtype == torch.int32
    ), f"expected int32 context_lens, got {context_lens.dtype}"
    assert block_kv in (32, 64), f"expected block_kv in (32, 64), got {block_kv}"
    assert SPLIT_KV % block_kv == 0
    assert num_sms > 0

    B, N = context_lens.shape
    next_n_atom = 2 if N >= 2 else 1
    num_next_n_atoms = (N + next_n_atom - 1) // next_n_atom

    last_lens = context_lens[:, N - 1].to(torch.int64).cpu().tolist()
    num_segs = [(L + SPLIT_KV - 1) // SPLIT_KV for L in last_lens]

    prefix_sum = [0] * B
    running = 0
    for b in range(B):
        running += num_segs[b]
        prefix_sum[b] = running

    total_segs = (prefix_sum[-1] if B > 0 else 0) * num_next_n_atoms
    q_, r_ = divmod(total_segs, num_sms)

    out_cpu = torch.zeros((num_sms + 1, 2), dtype=torch.int32)
    for s in range(num_sms + 1):
        seg_starts = s * q_ + min(s, r_)

        lo, hi = 0, B
        while lo < hi:
            mid = (lo + hi) // 2
            if prefix_sum[mid] * num_next_n_atoms <= seg_starts:
                lo = mid + 1
            else:
                hi = mid
        q_idx = lo

        if q_idx == 0:
            offset_in_q = seg_starts
            num_segs_q = prefix_sum[0] if B > 0 else 0
        else:
            offset_in_q = seg_starts - prefix_sum[q_idx - 1] * num_next_n_atoms
            num_segs_q = prefix_sum[q_idx] - prefix_sum[q_idx - 1] if q_idx < B else 0

        if num_segs_q > 0:
            atom_idx = offset_in_q // num_segs_q
            kv_split_idx = offset_in_q % num_segs_q
        else:
            atom_idx = 0
            kv_split_idx = 0

        q_atom_idx = q_idx * num_next_n_atoms + atom_idx
        out_cpu[s, 0] = q_atom_idx
        out_cpu[s, 1] = kv_split_idx

    if context_lens.device.type == "cpu":
        return out_cpu
    return out_cpu.to(context_lens.device, non_blocking=True)


def tilelang_fp8_paged_mqa_logits_sm120(
    q_fp8: torch.Tensor,
    kvcache_fp8: torch.Tensor,
    weight: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    deep_gemm_metadata: Any,
    max_seq_len: int,
    clean_logits: bool = False,
) -> torch.Tensor:
    """Adapter that forwards into the in-tree TileLang FP8 paged MQA logits.

    The Lightning Indexer dispatch site unsqueezes ``c4_seq_lens`` from
    ``(B,)`` to ``(B, 1)`` so the DeepGEMM CUDA path sees the documented
    2-D layout. The TileLang wrapper asserts a 1-D ``(B,)`` shape. We
    squeeze here so callers don't have to know which backend is selected.
    ``deep_gemm_metadata`` is accepted for API parity but ignored.
    """
    if seq_lens.dim() == 2 and seq_lens.shape[-1] == 1:
        seq_lens = seq_lens.squeeze(-1)

    from sglang.srt.layers.attention.nsa.tilelang_kernel import (
        tilelang_fp8_paged_mqa_logits,
    )

    return tilelang_fp8_paged_mqa_logits(
        q_fp8,
        kvcache_fp8,
        weight,
        seq_lens,
        page_table,
        deep_gemm_metadata,
        max_seq_len,
        clean_logits,
    )


__all__ = [
    "SPLIT_KV",
    "is_sm120",
    "get_paged_mqa_logits_metadata_python",
    "tilelang_fp8_paged_mqa_logits_sm120",
]
