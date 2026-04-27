"""sm_120 (RTX Pro 6000 Blackwell / RTX 5090) fallback paths for V4 attention.

This module bypasses upstream kernels that hard-fail on sm_120 hardware,
re-routing them through TileLang / Triton fallbacks that compile cleanly
for compute capability 12.0. This file covers both K1a and K2a.

  * **K1a — DeepGEMM ``fp8_paged_mqa_logits`` (Lightning Indexer)**
    DeepGEMM does not support sm_120
    (https://github.com/sgl-project/sglang/issues/23657,
    https://github.com/deepseek-ai/DeepGEMM/issues/236). The metadata
    helper is small enough to run on the host (a few microseconds for
    typical batch shapes), and the FP8 paged MQA logits kernel itself
    already has an in-tree TileLang implementation at
    ``sglang.srt.layers.attention.nsa.tilelang_kernel.tilelang_fp8_paged_mqa_logits``
    that is correct on sm_120.

  * **K2a — FlashMLA ``sparse_decode_fwd`` (V32 / V4-Flash)**
    FlashMLA's V32 sparse decode kernel is sm_90a / sm_100f only, with a
    hard ``TORCH_CHECK(false, "Unsupported architecture for sparse decode
    fwd")`` at ``csrc/api/sparse_decode.h:380`` for any other arch. We
    bypass this with a TileLang re-implementation in
    ``sm_120.tilelang_sparse_decode`` (T2A.3 K2a).

Public API
----------
Architecture probe:
- :func:`is_sm120`: cached architecture probe.

K1a (Lightning Indexer):
- :func:`get_paged_mqa_logits_metadata_python`: pure-Python port of the
  DeepGEMM scheduler.
- :func:`tilelang_fp8_paged_mqa_logits_sm120`: shape-adapter shim.

K2a (FlashMLA sparse decode):
- :func:`tilelang_fp8_sparse_decode_sm120`: shape-adapter shim.
- :func:`triton_combine_partials_sm120`: combine kernel pass-through.
"""

from __future__ import annotations

import functools
import logging
from typing import Any, Optional

import torch

logger = logging.getLogger(__name__)

_WARNED_ONCE: set = set()


def _warn_once(key: str, msg: str) -> None:
    if key not in _WARNED_ONCE:
        _WARNED_ONCE.add(key)
        logger.warning(msg)


_PROBE_ATTN_SINK_CALLS = 0
_PROBE_ATTN_SINK_LIMIT = 4


def _probe_log_attn_sink(attn_sink: torch.Tensor) -> None:
    """Log per-head stats for the first ``_PROBE_ATTN_SINK_LIMIT`` sink
    tensors that flow through ``tilelang_fp8_sparse_decode_sm120``.

    Phase-5 / T5.2 diagnostic. The log line includes shape, dtype, mean,
    abs-mean, min, max, and the first 16 head values. Throttled per
    process to keep the server log clean even on long GSM8K runs.
    """
    global _PROBE_ATTN_SINK_CALLS
    if _PROBE_ATTN_SINK_CALLS >= _PROBE_ATTN_SINK_LIMIT:
        return
    _PROBE_ATTN_SINK_CALLS += 1
    try:
        flat = attn_sink.detach().to(torch.float32).flatten()
        n = flat.numel()
        head = flat[: min(16, n)].cpu().tolist()
        nan_count = int(torch.isnan(flat).sum().item())
        inf_count = int(torch.isinf(flat).sum().item())
        logger.warning(
            "[sm_120 phase-5 T5.2 sink-probe %d/%d] shape=%s dtype=%s "
            "n=%d mean=%.4f abs_mean=%.4f min=%.4f max=%.4f "
            "nan=%d inf=%d head[:16]=%s",
            _PROBE_ATTN_SINK_CALLS,
            _PROBE_ATTN_SINK_LIMIT,
            tuple(attn_sink.shape),
            attn_sink.dtype,
            n,
            float(flat.mean().item()),
            float(flat.abs().mean().item()),
            float(flat.min().item()),
            float(flat.max().item()),
            nan_count,
            inf_count,
            [round(v, 4) for v in head],
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("[sm_120 phase-5 T5.2 sink-probe] log failed: %r", e)


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

    Squeezes ``seq_lens`` from ``(B, 1)`` → ``(B,)`` if needed.
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


def tilelang_fp8_sparse_decode_sm120(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    head_dim_v: int,
    block_table: Optional[torch.Tensor] = None,
    cache_seqlens: Optional[torch.Tensor] = None,
    tile_scheduler_metadata: Optional[torch.Tensor] = None,
    num_splits: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    is_fp8_kvcache: bool = True,
    indices: Optional[torch.Tensor] = None,
    attn_sink: Optional[torch.Tensor] = None,
    extra_k_cache: Optional[torch.Tensor] = None,
    extra_indices_in_kvcache: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
    extra_topk_length: Optional[torch.Tensor] = None,
) -> tuple:
    """sm_120 shape-adapter shim for FlashMLA's ``sparse_decode_fwd`` (K2a).

    Drop-in replacement for ``flash_mla.flash_mla_with_kvcache`` on the
    sparse-decode path. Forwards ``attn_sink`` to the kernel. Logs a
    warning (once) for unsupported kwargs.
    """
    assert isinstance(q, torch.Tensor) and q.dim() == 4, (
        f"q must be a 4-D torch.Tensor [b, s_q, h_q, d_qk]; got "
        f"{type(q).__name__} shape={getattr(q, 'shape', None)}"
    )
    assert (
        q.dtype == torch.bfloat16
    ), f"q must be bfloat16 (V32 contract), got {q.dtype}"
    assert isinstance(k_cache, torch.Tensor) and k_cache.dim() == 4, (
        f"k_cache must be 4-D [num_pages, page_size, h_kv=1, 584]; got "
        f"{type(k_cache).__name__} shape={getattr(k_cache, 'shape', None)}"
    )
    assert (
        head_dim_v == 512
    ), f"head_dim_v must be 512 for V32 (T1.3 §4); got {head_dim_v}"
    assert is_fp8_kvcache is True
    assert causal is False
    assert indices is not None
    assert block_table is None
    assert cache_seqlens is None

    if topk_length is not None:
        _warn_once(
            "topk_length",
            "sm_120 sparse-decode: topk_length not yet supported; "
            "treating decode positions as fully attended within top-k window",
        )

    # Phase-5 / T5.2 diagnostic probe: log the first N attn_sink tensors
    # observed at this dispatch point (per TP rank) so we can localise the
    # S1 residual to "fold math wrong" vs. "sink values themselves drifted
    # on sm_120". Active only when SGLANG_SM120_PROBE_ATTN_SINK=1.
    try:
        from sglang.srt.layers.sm120_diagnostic import probe_attn_sink

        if probe_attn_sink() and attn_sink is not None:
            _probe_log_attn_sink(attn_sink)
    except Exception:
        pass

    from sglang.srt.layers.attention.sm_120.tilelang_sparse_decode import (
        tilelang_fp8_sparse_decode,
    )
    from sglang.srt.layers.attention.sm_120.triton_combine import (
        triton_combine_partials_sm120,
    )
    from sglang.srt.layers.sm120_diagnostic import eager_sparse_decode

    # Phase-5 / T5.1 diagnostic: SGLANG_SM120_EAGER_SPARSE_DECODE=1 swaps
    # the TileLang FP8 sparse-decode kernel for an eager BF16 reference
    # that ALSO folds attn_sink in-kernel (so the post-hoc combine fold
    # is bypassed for the eager rows). Used to A/B both K2a kernel
    # correctness and the combine sink fold against the same oracle.
    if eager_sparse_decode():
        from sglang.srt.layers.attention.sm_120._eager_sparse_decode import (
            eager_fp8_sparse_decode,
        )

        sparse_decode_fn = eager_fp8_sparse_decode
        # Eager path applies attn_sink IN-KERNEL; pass None into combine
        # so we don't double-fold.
        sparse_decode_attn_sink_pass = attn_sink
        combine_attn_sink_pass = None
    else:
        sparse_decode_fn = tilelang_fp8_sparse_decode
        sparse_decode_attn_sink_pass = None
        combine_attn_sink_pass = attn_sink

    # SWA / "primary" KV path (always present on the V4 sparse-decode path).
    # The TileLang kernel itself ignores attn_sink; we fold it in during
    # combine below. The eager-reference kernel applies it in-kernel
    # (see toggle above).
    out_swa, lse_swa = sparse_decode_fn(
        q=q,
        kv_cache=k_cache,
        block_table=block_table,
        seq_lens=cache_seqlens,
        indices=indices,
        sm_scale=float(softmax_scale),
        attn_sink=sparse_decode_attn_sink_pass,
        is_fp8_kvcache=is_fp8_kvcache,
        h_kv=1,
        d_v=head_dim_v,
    )

    has_extra = extra_k_cache is not None and extra_indices_in_kvcache is not None

    if has_extra:
        # Hierarchical KV (c4 / c128 compressed path). Run K2a a second time
        # against the extra cache, then combine via log-sum-exp. This matches
        # FlashMLA's split-KV pipeline: the per-SM-part decode kernel emits
        # partial (out, lse) tuples and the combine kernel merges them
        # (csrc/smxx/decode/combine/combine.cu).
        out_extra, lse_extra = sparse_decode_fn(
            q=q,
            kv_cache=extra_k_cache,
            block_table=None,
            seq_lens=None,
            indices=extra_indices_in_kvcache,
            sm_scale=float(softmax_scale),
            attn_sink=sparse_decode_attn_sink_pass,
            is_fp8_kvcache=is_fp8_kvcache,
            h_kv=1,
            d_v=head_dim_v,
        )

        # Stack as 2 partials: shape [num_splits=2, b, s_q, h_q, d_v=512].
        # The kernel returns out as bf16 [b,s_q,h_q,d_v] and lse as fp32 [b,s_q,h_q].
        partials_out = torch.stack(
            [out_swa.float(), out_extra.float()], dim=0
        )  # [2, b, s_q, h_q, d_v]
        partials_lse = torch.stack([lse_swa, lse_extra], dim=0)  # [2, b, s_q, h_q]

        return triton_combine_partials_sm120(
            partials_out=partials_out,
            partials_lse=partials_lse,
            attn_sink=combine_attn_sink_pass,
        )

    # Single-split path: still go through combine to apply attn_sink
    # consistently (the combine pass is a pass-through for num_splits=1
    # other than the attn_sink fold).
    partials_out = out_swa.float().unsqueeze(0)  # [1, b, s_q, h_q, d_v]
    partials_lse = lse_swa.unsqueeze(0)  # [1, b, s_q, h_q]
    return triton_combine_partials_sm120(
        partials_out=partials_out,
        partials_lse=partials_lse,
        attn_sink=combine_attn_sink_pass,
    )


from sglang.srt.layers.attention.sm_120.triton_combine import (  # noqa: E402
    triton_combine_partials_sm120,
)

__all__ = [
    "SPLIT_KV",
    "is_sm120",
    "get_paged_mqa_logits_metadata_python",
    "tilelang_fp8_paged_mqa_logits_sm120",
    "tilelang_fp8_sparse_decode_sm120",
    "triton_combine_partials_sm120",
]
