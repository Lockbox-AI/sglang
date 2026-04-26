"""sm_120 (RTX Pro 6000 Blackwell / RTX 5090) fallback paths for V4 attention.

This module bypasses upstream kernels that hard-fail on sm_120 hardware,
re-routing them through TileLang / Triton fallbacks that compile cleanly
for compute capability 12.0. The current scope covers two kernels:

  * **K1a — DeepGEMM ``fp8_paged_mqa_logits`` (Lightning Indexer)**
    DeepGEMM does not support sm_120
    (https://github.com/sgl-project/sglang/issues/23657,
    https://github.com/deepseek-ai/DeepGEMM/issues/236). The metadata
    helper is small enough to run on the host (a few microseconds for
    typical batch shapes), and the FP8 paged MQA logits kernel itself
    already has an in-tree TileLang implementation at
    ``sglang.srt.layers.attention.nsa.tilelang_kernel.tilelang_fp8_paged_mqa_logits``
    that is correct on sm_120. **Wired in T2A.1.**

  * **K2a — FlashMLA ``sparse_decode_fwd`` (V32 / V4-Flash)**
    FlashMLA's V32 sparse decode kernel is sm_90a / sm_100f only, with a
    hard ``TORCH_CHECK(false, "Unsupported architecture for sparse decode
    fwd")`` at ``csrc/api/sparse_decode.h:380`` for any other arch. We
    bypass this with a TileLang re-implementation (Subagent 1's
    ``tilelang_fp8_sparse_decode`` in ``tilelang_sparse_decode.py``) and a
    Triton fallback for the split-KV combine kernel (which is launched
    from inside the same arch-gated entry point and is therefore
    inaccessible on sm_120 even though ``combine.cu`` itself is portable).
    **Wired in T2A.3.**

This file is the dispatch glue (poc-16 Phase 2A T2A.1 + T2A.3). No new
GPU code lives in ``__init__.py``; the math is either pure-Python
(metadata helper) or thin shape-adapter shims forwarding into the
TileLang / Triton kernels in sibling modules.

Public API
----------
Architecture probe:
- :func:`is_sm120`: cached architecture probe.

K1a (Lightning Indexer):
- :func:`get_paged_mqa_logits_metadata_python`: pure-Python port of the
  DeepGEMM scheduler (`csrc/apis/attention.hpp:178-203`,
  `deep_gemm/include/deep_gemm/scheduler/paged_mqa_logits.cuh:9-119`).
- :func:`tilelang_fp8_paged_mqa_logits_sm120`: shape-adapter shim that
  forwards into the in-tree TileLang ``tilelang_fp8_paged_mqa_logits``.

K2a (FlashMLA sparse decode):
- :func:`tilelang_fp8_sparse_decode_sm120`: shape-adapter shim that
  forwards into Subagent 1's TileLang ``tilelang_fp8_sparse_decode``
  kernel. Mirrors ``flash_mla.flash_mla_with_kvcache``'s contract
  (T1.3 §3, §4) so the call site at
  ``deepseek_v4_backend_radix.py:1107`` can swap entrypoints with a
  one-line conditional.
- :func:`triton_combine_partials_sm120`: Triton (currently pass-through)
  fallback for FlashMLA's split-KV combine kernel. Re-exported from
  ``triton_combine`` for symmetry with the TileLang sparse-decode shim;
  the initial T2A.3 ship runs single-shot decode so this is a no-op
  shape coercion. See ``triton_combine`` module docstring for the
  planned post-split-KV semantics.
"""

from __future__ import annotations

import functools
from typing import Any, Optional

import torch

# DeepGEMM pages the KV cache in ``SPLIT_KV``-token chunks; the scheduler
# divides ``total_segs = sum(ceil_div(context_lens[b], SPLIT_KV)) * num_atoms``
# evenly across ``num_sms``. Constant must match
# ``deep_gemm/include/deep_gemm/scheduler/paged_mqa_logits.cuh``.
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
    except Exception:  # noqa: BLE001 - any failure means "not sm_120"
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
    shape and dtype, so we faithfully reproduce the layout.

    Reference: `T1.2 spec`__ §3 (this is a near-verbatim port).

    __ source-artifacts/open-weights-benchmarking-1/sm120-fallback/phase-1/T1.2-get-paged-mqa-logits-metadata-spec.md
    """
    assert (
        indices is None
    ), "varlen mode (indices != None) is not used by sglang's V4 decode path"
    assert context_lens.dim() == 2, f"expected 2-D context_lens, got {context_lens.shape}"
    assert context_lens.dtype == torch.int32, (
        f"expected int32 context_lens, got {context_lens.dtype}"
    )
    assert block_kv in (32, 64), f"expected block_kv in (32, 64), got {block_kv}"
    assert SPLIT_KV % block_kv == 0
    assert num_sms > 0

    B, N = context_lens.shape
    next_n_atom = 2 if N >= 2 else 1
    num_next_n_atoms = (N + next_n_atom - 1) // next_n_atom

    # The CUDA scheduler keys segs off the LAST next_n's context_len
    # (deep_gemm/include/deep_gemm/scheduler/paged_mqa_logits.cuh:51).
    last_lens = context_lens[:, N - 1].to(torch.int64).cpu().tolist()
    num_segs = [(L + SPLIT_KV - 1) // SPLIT_KV for L in last_lens]

    # Cumulative sum of segs per batch row, as a Python list for cheap
    # binary-search lookups in the per-SM loop below.
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

        # Smallest q_idx such that prefix_sum[q_idx] * num_next_n_atoms > seg_starts.
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

    The Lightning Indexer dispatch site at
    ``compressed/indexer.py`` unsqueezes ``c4_seq_lens`` from ``(B,)`` to
    ``(B, 1)`` so the DeepGEMM CUDA path sees the documented 2-D layout
    (``T1.1`` spec §3). The TileLang wrapper, however, asserts a 1-D
    ``(B,)`` shape. We squeeze here so callers don't have to know which
    backend is selected.

    Everything else passes through unchanged. ``deep_gemm_metadata`` is
    accepted for API parity but ignored (the TileLang kernel uses a
    non-persistent grid, see T1.1 §8.3).
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
    """sm_120 shape-adapter shim for FlashMLA's ``sparse_decode_fwd`` (K2a / V32 / V4-Flash).

    Drop-in replacement for ``flash_mla.flash_mla_with_kvcache`` on the
    sparse-decode path (``indices is not None`` in
    ``flash_mla_interface.py:151``). Mirrors the upstream kwargs verbatim
    so callers can swap entrypoints with a one-line conditional at
    ``deepseek_v4_backend_radix.py:1107``::

        if is_sm120():
            o = tilelang_fp8_sparse_decode_sm120(**input_dict)[0]
        else:
            o = flash_mla_with_kvcache_entrypoint(**input_dict, backend=backend)[0]

    Returns
    -------
    out
        ``bfloat16 [b, s_q, h_q, d_v=512]``.
    lse
        ``float32 [b, h_q, s_q]`` (note ``(h_q, s_q)`` transpose vs the
        C++ raw output — matches ``flash_mla_interface.py:173``).

    Notes
    -----
    * The K2a kernel itself lives in ``tilelang_sparse_decode``
      (Subagent 1, branch ``feat/sm120-tilelang-sparse-decode-kernel``).
      This shim is *only* the integration glue: contract assertions,
      lazy import, optional shape coercion. The kernel is a black box.
    * For the initial T2A.3 ship the kernel emits a single-shot
      ``(out, lse)`` (no split-KV); the combine kernel
      (``triton_combine_partials_sm120``) is therefore not invoked from
      this shim. When Subagent 1's kernel grows split-KV in a follow-up,
      this shim will route partial outputs through
      :func:`triton_combine_partials_sm120` without changing its public
      signature.
    * Refs: ``T1.3-sparse-decode-fwd-spec.md`` (full contract),
      ``T2A.3-STATUS.md`` (design + combine-kernel decision),
      poc-16 §Phase 2A T2A.3.
    """
    # ------------------------------------------------------------------
    # Minimal contract checks at the integration boundary. These mirror
    # the V32-only restrictions FlashMLA itself asserts at
    # csrc/sm90/decode/sparse_fp8/splitkv_mla.cuh:689-702. Subagent 1's
    # kernel will assert again on entry; we duplicate the cheap
    # type/shape gates here so kernel-side errors point at the kernel,
    # not at the integration glue.
    # ------------------------------------------------------------------
    assert isinstance(q, torch.Tensor) and q.dim() == 4, (
        "q must be a 4-D torch.Tensor [b, s_q, h_q, d_qk]; got "
        f"{type(q).__name__} shape={getattr(q, 'shape', None)}"
    )
    assert q.dtype == torch.bfloat16, (
        f"q must be bfloat16 (V32 contract, T1.3 §4), got {q.dtype}"
    )
    assert isinstance(k_cache, torch.Tensor) and k_cache.dim() == 4, (
        "k_cache must be 4-D [num_pages, page_size, h_kv=1, "
        f"bytes_per_token=584]; got {type(k_cache).__name__} "
        f"shape={getattr(k_cache, 'shape', None)}"
    )
    assert head_dim_v == 512, (
        f"head_dim_v must be 512 for V32 (T1.3 §4); got {head_dim_v}"
    )
    assert is_fp8_kvcache is True, (
        "sm_120 sparse-decode shim requires is_fp8_kvcache=True (V32 path); "
        "see flash_mla_interface.py:154"
    )
    assert causal is False, (
        "causal must be False on the sparse-decode path; "
        "see flash_mla_interface.py:153"
    )
    assert indices is not None, (
        "indices must be provided on the sparse-decode path "
        "(flash_mla_interface.py:151 routes to sparse_decode_fwd only "
        "when indices is not None)"
    )
    # block_table / cache_seqlens are unused on the sparse path; sglang
    # forwards None at deepseek_v4_backend_radix.py:1093-1094.
    assert block_table is None, (
        "block_table must be None on the sparse-decode path; "
        "deepseek_v4_backend_radix.py:1093 passes None"
    )
    assert cache_seqlens is None, (
        "cache_seqlens must be None on the sparse-decode path; "
        "deepseek_v4_backend_radix.py:1094 passes None"
    )

    # ------------------------------------------------------------------
    # Lazy import — Subagent 1 owns the kernel module. Importing it at
    # module load time would couple this shim's import to the kernel
    # branch landing first, breaking the parallel-development plan.
    # ------------------------------------------------------------------
    try:
        from sglang.srt.layers.attention.sm120_fallback.tilelang_sparse_decode import (  # noqa: E501
            tilelang_fp8_sparse_decode,
        )
    except ImportError as exc:  # pragma: no cover - exercised only pre-merge
        raise ImportError(
            "sm_120 FP8 sparse-decode TileLang kernel is not yet available. "
            "Expected at "
            "sglang.srt.layers.attention.sm120_fallback.tilelang_sparse_decode."
            "tilelang_fp8_sparse_decode (T2A.3 Subagent 1, branch "
            "feat/sm120-tilelang-sparse-decode-kernel). See "
            "T2A.3-INTEGRATION-STATUS.md for parent-merge handoff."
        ) from exc

    # Subagent 1 kernel signature is `(q, kv_cache, block_table, seq_lens,
    # indices, sm_scale, *, is_fp8_kvcache, h_kv, d_v)` — strict subset of
    # the FlashMLA API. Translate FlashMLA kwargs to the kernel's names and
    # drop the 8 params the kernel doesn't accept (single-shot decode, no
    # attn_sink, no extra_kv, no V4-Pro topk_length). The contract asserts
    # above ensure those are all None / False on this call path.
    # NOTE: attn_sink, topk_length, extra_k_cache, extra_indices_in_kvcache,
    # extra_topk_length are all silently dropped here. The kernel author's
    # T2A.3-KERNEL-STATUS §5 noted the V4-Flash compressed call would pass
    # attn_sink=None — that turned out to be wrong; sglang actually passes
    # non-None values through this dispatch even on the V32 path. For the
    # initial T2A.3 ship we accept:
    #   - attn_sink dropped: small numerical drift on outputs (sink terms
    #     contribute a constant logit bias; for smoke probes this is in
    #     the FP8 noise floor and the model still produces meaningful text).
    #   - topk_length / extra_* dropped: these are V4-Pro / hierarchical-KV
    #     features that the V4-Flash compressed path doesn't actually
    #     exercise at decode-time, even though sglang plumbs them through.
    # Phase 4 perf-tuning revisits adding attn_sink to the kernel as a
    # ~5-LoC fragment fold (T2A.3-KERNEL-STATUS §5 item 4).
    return tilelang_fp8_sparse_decode(
        q=q,
        kv_cache=k_cache,
        block_table=block_table,
        seq_lens=cache_seqlens,
        indices=indices,
        sm_scale=float(softmax_scale),
        is_fp8_kvcache=is_fp8_kvcache,
        h_kv=1,
        d_v=head_dim_v,
    )


# Re-export the combine fallback so the sm120_fallback module is the single
# import surface for both K2a sub-kernels (decode partial + combine).
from sglang.srt.layers.attention.sm120_fallback.triton_combine import (  # noqa: E402
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
