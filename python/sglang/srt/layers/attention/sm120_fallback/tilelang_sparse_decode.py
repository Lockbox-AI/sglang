"""sm_120 TileLang FP8 sparse-decode kernel for V4 RadixAttention (V32 / K2a).

This module provides the TileLang implementation of FlashMLA's
``sparse_decode_fwd`` for sm_120 (RTX Pro 6000 Blackwell, RTX 5090). Upstream
FlashMLA only ships sm_90a (WGMMA) and sm_100a (TCGEN05/TMEM) implementations
of this kernel; both rely on tensor-core / shared-memory features that don't
exist on sm_120. We re-author the algorithm in TileLang against plain
``mma.m16n8k16`` (4th-gen tensor core), online softmax in log2 domain, and
``cp.async``-style smem loads.

This is the **kernel-only** deliverable of T2A.3 (subagent 1 of 3). The wrapper
(``flash_mla_with_kvcache``-shaped FlashMLA shim) and the combine fallback are
authored separately under ``sm120_fallback/__init__.py`` and a forthcoming
``combine.py``.

References
----------
- Spec: ``source-artifacts/open-weights-benchmarking-1/sm120-fallback/phase-1/T1.3-sparse-decode-fwd-spec.md``
  (read in particular §3 signature, §4 V32 layout, §6 algorithm walkthrough,
  §7 scheduling, §8.2 Option A — the design we implement here).
- Design framing: ``…/phase-2a/T2A.3-STATUS.md``.
- Author target: kernel + smoke that compiles and runs on sm_120; numerical
  correctness against an eager PyTorch reference is owned by Subagent 3.
- Closest in-tree TileLang cousin: ``nsa/tilelang_kernel.py``
  ``sparse_attention_fwd_kernel_v1`` (BF16 sparse-decode, no FP8 dequant),
  whose ``H_per_block × D`` Q-tile and online-softmax structure we mirror.

V32 KV cache layout (656 bytes per token, per T1.3 §4.1)
--------------------------------------------------------

::

    bytes 0..512    FP8 e4m3 NoPE (512 elements)
    bytes 512..528  FP32 per-128 scales (4 elements)
    bytes 528..656  BF16 RoPE (64 elements)

We split the byte-packed ``[num_blocks, page_block_size, 656]`` ``uint8``
tensor into three typed strided views in Python (zero-copy) and pass them
to the kernel as ``T.StridedTensor`` arguments. This mirrors how
``tilelang_fp8_paged_mqa_logits`` (``nsa/tilelang_kernel.py:864-898``)
splits its packed FP8+FP32 cache.

Live shared-memory accounting (target ≤ 40 KB per T1.3 §8.2 Option A;
hard cap is 99 KB opt-in / 48 KB default per
``cuda-toolkit-distilled-for-sm120-work.md`` §1)
-----------------------------------------------------------------------

::

    K_nope_smem   [BLOCK_TOPK=32, D_NOPE=512] BF16  = 32 KB     (32,768 B)
    K_rope_smem   [BLOCK_TOPK=32, D_ROPE= 64] BF16  =  4 KB     ( 4,096 B)
    S_smem        [H_Q≤128, BLOCK_TOPK=32]    BF16  ≤ 8 KB      (≤8,192 B)
    indices_smem  [BLOCK_TOPK=32]             INT32 =  128 B
    is_kv_valid   [BLOCK_TOPK=32]             bool  =   32 B
    {alpha,m_i,m_i_prev,sumexp}_smem [H_Q]    FP32  ≤ 2 KB      (≤2,048 B)
    -------------------------------------------------------------
    Total live smem (H_Q=64)  ≈ 41 KB (41,280 B)
    Total live smem (H_Q=128) ≈ 47 KB (47,232 B)

    Both fit comfortably under the 48 KB sm_120 default cap; no opt-in
    needed for the V4-Flash configuration (H_Q=64).

The four small ``*_smem`` scratch buffers exist to break TileLang 0.1.8's
fragment-layout-inference chain on this BLOCK_TOPK=32 shape — see the
comment at their allocation. Footprint is negligible (≤ 2 KB total at
H_Q=128) and the same pattern is used by the in-tree v2 H100 kernel at
``nsa/tilelang_kernel.py:481`` (``alpha_shared``, ``sum_exp_shared``).

Q is held in **register fragments** (no Q_smem) per T1.3 §8.2 Option A —
the Q tile alone is ``H_Q * D_qk * 2 = 64*576*2 = 72 KB``, which would blow
both the default 48 KB and the opt-in 99 KB cap once K-buffers are added.

Single-buffered KV smem (``num_stages=1``) is intentional: with ``num_stages
≥ 2`` we'd need 2× the K-buffer footprint, again pushing past 48 KB.

Scheduling — single-shot decode (NOT split-KV)
----------------------------------------------

This kernel is **single-program-per-(B, S_q)** (grid = ``(B, S_q)``). It
attends across the entire ``topk`` set and writes the final ``(out, lse)``
in one pass, with no per-SM-part split. The combine kernel (Subagent 2's
deliverable) is therefore not invoked from this kernel; the output shapes
match what the wrapper passes on to downstream consumers.

This intentionally diverges from FlashMLA's sm_90 reference, which is a
3-warpgroup persistent kernel with split-KV partitioning. T1.3 §7 explains
why we drop persistence on sm_120: no warpgroup MMA, no TMA, no DSMEM
clusters, and a much tighter smem budget. The split-KV combine pass remains
a Phase 4 perf optimization; for correctness ``rtol=1e-3`` ceiling, a
single-shot decode is sufficient.

``BLOCK_TOPK = 32`` (halved from sm_90's 64)
---------------------------------------------

The sm_90 reference uses ``TOPK_BLOCK_SIZE = 64`` and asserts
``topk % 64 == 0`` upstream. On sm_120 we halve to 32 to keep K_nope_smem
at 32 KB (one 32-row tile) instead of 64 KB; this combined with dropping
the K double-buffer is what gets us under the 48 KB cap. The wrapper
should therefore relax the upstream guard to ``topk % 32 == 0`` on the
sm_120 path. Standard sglang topk values (512, 2048) are multiples of 32
already.
"""

import functools
from typing import Optional, Tuple

import tilelang
import tilelang.language as T
import torch

tilelang.set_log_level("WARNING")


# ---------------------------------------------------------------------------
# V32 packed KV layout (T1.3 §4.1, T1.4 stub).
# ---------------------------------------------------------------------------
_V32_D_NOPE: int = 512
_V32_D_ROPE: int = 64
_V32_NUM_SCALES: int = _V32_D_NOPE // 128  # = 4
_V32_BYTES_PER_TOKEN: int = (
    _V32_D_NOPE                      # 512  fp8 NoPE
    + _V32_NUM_SCALES * 4            #  16  fp32 scales
    + _V32_D_ROPE * 2                # 128  bf16 RoPE
)
assert _V32_BYTES_PER_TOKEN == 656

# sm_120 design constants (T1.3 §8.2 Option A; see module docstring).
# Spec target is BLOCK_TOPK=32. We expose this as a private knob to allow a
# fallback to 64 if TileLang's layout inference rejects the 32-wide
# acc_s = [H_q, 32] shape on a given hardware/version combination.
_BLOCK_TOPK: int = 32

# log2(e) — used to convert sm_scale * x → exp2(...) for online softmax.
_LOG2E: float = 1.44269504


# TileLang dtype aliases (match conventions in nsa/tilelang_kernel.py).
_BF16 = "bfloat16"
_FP8 = "float8_e4m3"
_FP32 = "float32"
_INT32 = "int32"


_pass_configs = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
}


@tilelang.jit(pass_configs=_pass_configs)
def _fp8_sparse_decode_kernel_sm120(
    h_q: int,
    d_v: int,
    d_nope: int,
    d_rope: int,
    block_topk: int,
    num_scales: int,
    sm_scale_log2e_q: int,
    threads: int = 256,
    num_stages: int = 1,
):
    """Build (and JIT-cache) the TileLang kernel for one (h_q, d_*, sm_scale).

    ``sm_scale_log2e_q`` is the ``round(sm_scale * log2(e) * 1e9)``
    quantized integer key — keeps caching stable across float perturbations.

    The decoration pattern (``@tilelang.jit`` on the *builder* + ``@T.prim_func``
    on the inner ``main``) mirrors the in-tree convention at
    ``nsa/tilelang_kernel.py:207-373`` (``sparse_attention_fwd_kernel_v1``);
    TileLang JITs the prim_func returned by the builder and caches by the
    builder's kwargs.
    """
    sm_scale_log2e: float = sm_scale_log2e_q * 1e-9
    d_qk: int = d_nope + d_rope

    B = T.symbolic("B")
    S_q = T.symbolic("S_q")
    N_total = T.symbolic("N_total")  # = num_blocks * page_block_size
    topk = T.symbolic("topk")
    NI = T.ceildiv(topk, block_topk)

    # Dynamic per-row strides for the three byte-reinterpreted views.
    # Concrete values at the call site (with bytes_per_token=656):
    #   FP8  : stride = 656 elements/row (1 B/elt)
    #   FP32 : stride = 164 elements/row (4 B/elt)
    #   BF16 : stride = 328 elements/row (2 B/elt)
    s_kvf, s_kvs, s_kvr = T.dynamic("s_kvf, s_kvs, s_kvr")

    @T.prim_func
    def main(
        Q: T.Tensor[(B, S_q, h_q, d_qk), _BF16],  # type: ignore[name-defined]
        KV_fp8: T.StridedTensor[(N_total, d_nope), (s_kvf, 1), _FP8],  # type: ignore[name-defined]
        KV_scales: T.StridedTensor[(N_total, num_scales), (s_kvs, 1), _FP32],  # type: ignore[name-defined]
        KV_rope: T.StridedTensor[(N_total, d_rope), (s_kvr, 1), _BF16],  # type: ignore[name-defined]
        Indices: T.Tensor[(B, S_q, topk), _INT32],  # type: ignore[name-defined]
        Out: T.Tensor[(B, S_q, h_q, d_v), _BF16],  # type: ignore[name-defined]
        Lse: T.Tensor[(B, S_q, h_q), _FP32],  # type: ignore[name-defined]
    ):
        with T.Kernel(B, S_q, threads=threads) as (b_i, s_i):
            # -- Smem (live ~40 KB; see module docstring) ----------------
            K_nope_smem = T.alloc_shared([block_topk, d_nope], _BF16)
            K_rope_smem = T.alloc_shared([block_topk, d_rope], _BF16)
            S_smem = T.alloc_shared([h_q, block_topk], _BF16)
            indices_smem = T.alloc_shared([block_topk], _INT32)
            is_kv_valid_smem = T.alloc_shared([block_topk], "bool")
            # Tiny scratch buffers (≤ 2 KB total) for the online-softmax
            # state — kept in smem to BREAK the fragment-layout-inference
            # chain that otherwise conflicts at BLOCK_TOPK=32. (TileLang
            # 0.1.8 layout inference rejects ``[h_q, 32]`` acc_s reductions
            # feeding 1-D fragment reads + 2-D fragment reads in the same
            # SSA chain. The v2 H100 kernel at ``nsa/tilelang_kernel.py:481``
            # uses the same ``alpha_shared`` / ``sum_exp_shared`` pattern.)
            alpha_smem = T.alloc_shared([h_q], _FP32)
            m_i_smem = T.alloc_shared([h_q], _FP32)
            m_i_prev_smem = T.alloc_shared([h_q], _FP32)
            sumexp_smem = T.alloc_shared([h_q], _FP32)

            # -- Fragments (registers) -----------------------------------
            # Q stays in registers across all topk iterations; loaded once
            # below from gmem. Reusing across iters is strictly better than
            # T1.3 §8.2 Option A's "stream Q every iter" suggestion as long
            # as register pressure permits — the spec accepts either.
            Q_nope_frag = T.alloc_fragment([h_q, d_nope], _BF16)
            Q_rope_frag = T.alloc_fragment([h_q, d_rope], _BF16)
            K_scales_frag = T.alloc_fragment([block_topk, num_scales], _FP32)

            acc_o = T.alloc_fragment([h_q, d_v], _FP32)
            acc_s = T.alloc_fragment([h_q, block_topk], _FP32)
            sumexp = T.alloc_fragment([h_q], _FP32)
            sumexp_i = T.alloc_fragment([h_q], _FP32)
            alpha = T.alloc_fragment([h_q], _FP32)
            m_i = T.alloc_fragment([h_q], _FP32)

            # -- Init online-softmax accumulator state -------------------
            T.fill(acc_o, 0.0)
            T.fill(sumexp, 0.0)
            # Use finite "−inf" (-2^30) to avoid -inf - inf = NaN later.
            T.fill(m_i, -(2.0**30))

            # -- Load Q once (no Q_smem; T1.3 §8.2 Option A) -------------
            T.copy(Q[b_i, s_i, :, 0:d_nope], Q_nope_frag)
            T.copy(Q[b_i, s_i, :, d_nope:d_qk], Q_rope_frag)

            for i_i in T.Pipelined(NI, num_stages=num_stages):
                # -- Read indices, mark validity, clamp for safe gather --
                # Invalid entries are encoded as -1 upstream (T1.3 §4).
                # We clamp to 0 so the gathers below stay in-bounds; the
                # is_kv_valid_smem mask drops those lanes to -INF in acc_s
                # before the QK softmax, so garbage K reads don't pollute
                # the result.
                for bi_i in T.Parallel(block_topk):
                    raw = Indices[b_i, s_i, i_i * block_topk + bi_i]
                    is_kv_valid_smem[bi_i] = raw >= 0
                    indices_smem[bi_i] = T.max(raw, 0)

                # -- Pre-load 4 FP32 scales per token into a fragment ----
                for bi_i, ns_i in T.Parallel(block_topk, num_scales):
                    K_scales_frag[bi_i, ns_i] = KV_scales[
                        indices_smem[bi_i], ns_i
                    ]

                # -- Gather + dequantize FP8 NoPE → BF16 (per-128 scale) --
                # Each FP8 element gets multiplied by its quant-tile scale
                # (4 scales × 128 elements = 512 NoPE dims). Match the
                # reference's `cvt_fp8x8_bf16x8(data, scale_bf162)` pattern
                # at FlashMLA's `splitkv_mla.cuh:586-598`.
                for bi_i, d_i in T.Parallel(block_topk, d_nope):
                    fp8_val = KV_fp8[indices_smem[bi_i], d_i]
                    K_nope_smem[bi_i, d_i] = T.Cast(
                        _BF16,
                        T.Cast(_FP32, fp8_val)
                        * K_scales_frag[bi_i, d_i // 128],
                    )

                # -- Gather BF16 RoPE (no dequant; already bf16) ---------
                for bi_i, d_i in T.Parallel(block_topk, d_rope):
                    K_rope_smem[bi_i, d_i] = KV_rope[
                        indices_smem[bi_i], d_i
                    ]

                # -- Init acc_s with mask: -INF for invalid lanes --------
                # -INF + (finite QK contribution) = -INF in IEEE-754, so
                # invalid lanes survive the two GEMMs below as -INF and
                # then go to 0 after exp2(...).
                for h_i, bi_i in T.Parallel(h_q, block_topk):
                    acc_s[h_i, bi_i] = T.if_then_else(
                        is_kv_valid_smem[bi_i],
                        0.0,
                        -T.infinity(_FP32),
                    )

                # -- QK^T = (Q_nope · K_nope^T) + (Q_rope · K_rope^T) ----
                # Both gemms are BF16×BF16→FP32, lowering to mma.m16n8k16
                # on sm_120 (4th-gen tensor core).
                T.gemm(
                    Q_nope_frag,
                    K_nope_smem,
                    acc_s,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullCol,
                )
                T.gemm(
                    Q_rope_frag,
                    K_rope_smem,
                    acc_s,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullCol,
                )

                # -- Online softmax in log2 domain (T1.3 §5.1, §6) -------
                # exp2(s * sm_scale * log2(e)) ≡ exp(s * sm_scale).
                # The smem buffers below break the fragment-layout chain
                # at BLOCK_TOPK=32 (see comment at the smem allocation).
                T.copy(m_i, m_i_prev_smem)
                T.reduce_max(acc_s, m_i, dim=1, clear=False)
                T.copy(m_i, m_i_smem)
                for h_i in T.Parallel(h_q):
                    alpha[h_i] = T.exp2(
                        (m_i_prev_smem[h_i] - m_i_smem[h_i]) * sm_scale_log2e
                    )
                T.copy(alpha, alpha_smem)
                for h_i, bi_i in T.Parallel(h_q, block_topk):
                    acc_s[h_i, bi_i] = T.exp2(
                        acc_s[h_i, bi_i] * sm_scale_log2e
                        - m_i_smem[h_i] * sm_scale_log2e
                    )
                T.reduce_sum(acc_s, sumexp_i, dim=1)
                for h_i in T.Parallel(h_q):
                    sumexp[h_i] = sumexp[h_i] * alpha_smem[h_i] + sumexp_i[h_i]
                for h_i, d_i in T.Parallel(h_q, d_v):
                    acc_o[h_i, d_i] = acc_o[h_i, d_i] * alpha_smem[h_i]

                # -- S·V where V == K_nope_bf16 (T1.3 §6 step 7) ---------
                # Note: V is the dequantized NoPE tile, NOT the full K_qk.
                # FlashMLA's reference is identical (`splitkv_mla.cuh:613`
                # comment: "We do not need to mask the RoPE part for V3.2
                # since it isn't involved in the SV gemm").
                T.copy(acc_s, S_smem)
                T.gemm(
                    S_smem,
                    K_nope_smem,
                    acc_o,
                    policy=T.GemmWarpPolicy.FullCol,
                )

            # -- Final rescale + LSE write -------------------------------
            T.copy(sumexp, sumexp_smem)
            for h_i, d_i in T.Parallel(h_q, d_v):
                acc_o[h_i, d_i] = acc_o[h_i, d_i] / sumexp_smem[h_i]
            for h_i in T.Parallel(h_q):
                # Convert back to natural-log LSE (T1.3 §5.1, §6).
                Lse[b_i, s_i, h_i] = (
                    T.log2(sumexp_smem[h_i]) + m_i_smem[h_i] * sm_scale_log2e
                ) / _LOG2E

            # Implicit FP32→BF16 conversion via T.copy (matches v1 line 371).
            T.copy(acc_o, Out[b_i, s_i, :, :])

    return main


def tilelang_fp8_sparse_decode(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: Optional[torch.Tensor],
    seq_lens: Optional[torch.Tensor],
    indices: torch.Tensor,
    sm_scale: float,
    *,
    is_fp8_kvcache: bool = True,
    h_kv: int = 1,
    d_v: int = 512,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """sm_120 TileLang FP8 sparse-decode (V32 / K2a) — single-shot decode.

    The wrapper that adapts FlashMLA's ``flash_mla_with_kvcache`` signature
    to this function lives in ``sm120_fallback/__init__.py`` and is owned
    by Subagent 2 (T2A.3 wiring track).

    Parameters
    ----------
    q : ``bfloat16 [B, S_q, H_q, D_qk]``
        Query tensor, last-dim contiguous. ``D_qk == 576`` for V32
        (``D_NOPE=512 + D_ROPE=64``); ``H_q ∈ {64, 128}`` (V4-Flash uses 64).
    kv_cache : ``uint8 / int8 / float8_e4m3fn [num_blocks, page_block_size, 656]``
        Packed paged KV cache. Each 656-byte token record holds
        ``[FP8 NoPE 0..512 | FP32 scales 512..528 | BF16 RoPE 528..656]``
        (T1.3 §4.1; T1.4 stub corrects the Run-1 layout). Must be contiguous
        — the kernel reinterprets bytes via three zero-copy strided views.
        A 4-D ``[num_blocks, page_block_size, h_kv=1, 656]`` shape (FlashMLA's
        native shape) is also accepted and squeezed.
    block_table : ``Optional[int32 [B, max_blocks_per_seq]]``
        Unused on the sparse-decode path (FlashMLA's wrapper passes ``None``);
        accepted for API parity with downstream callers that pass a 4-D
        block table from the V4 paged-KV builder.
    seq_lens : ``Optional[int32 [B]]``
        Unused on the sparse-decode path; accepted for API parity. The
        ``indices`` tensor encodes the valid-topk set directly via -1 sentinels.
    indices : ``int32 [B, S_q, topk]``
        Flat token indices into the paged KV cache:
        ``indices[b, s, k] = page_block_idx * page_block_size + offset_in_block``
        (per ``flash_mla_interface.py:86-87``). Invalid entries: ``-1``.
        Last-dim contiguous. ``topk`` must be a multiple of ``BLOCK_TOPK=32``
        on this sm_120 path (vs the upstream sm_90 ``topk % 64`` requirement).
    sm_scale : ``float``
        Pre-softmax scale (typically ``1/sqrt(D_qk)`` for V4-Flash).
    is_fp8_kvcache : ``bool``, keyword-only
        Must be ``True`` (V32 path). Accepted for API parity.
    h_kv : ``int``, keyword-only
        Must be ``1`` (MQA). Accepted for API parity.
    d_v : ``int``, keyword-only
        Must be ``512`` (V32 hard-asserts; ``sparse_decode.h:236``).

    Returns
    -------
    out : ``bfloat16 [B, S_q, H_q, D_v=512]``
    lse : ``float32 [B, S_q, H_q]``
        Log-sum-exp in natural-log units. Shape matches what the C++ kernel
        allocates pre-transpose at ``sparse_decode.h:316``; the FlashMLA
        Python wrapper transposes to ``[B, H_q, S_q]`` at the API boundary,
        which is the wrapper layer's (Subagent 2's) responsibility.

    Notes
    -----
    Single-shot decode — no split-KV partitioning. See the module docstring's
    "Scheduling" section for why we drop persistence on sm_120. If the
    downstream wrapper needs split-KV ``(o_accum, lse_accum)`` partials for
    a combine pass, it can either run this kernel and skip the combine
    (output already final) or call us once per shard and combine externally.
    """
    # ----- Validate q ------------------------------------------------------
    assert isinstance(q, torch.Tensor), f"q must be a Tensor, got {type(q)}"
    assert q.dim() == 4, f"q must be [B, S_q, H_q, D_qk], got dim={q.dim()}"
    assert q.dtype == torch.bfloat16, f"q must be bfloat16, got {q.dtype}"
    assert q.stride(-1) == 1, "q must have last-dim contiguous"
    B, S_q, H_q, D_qk = q.shape
    D_NOPE, D_ROPE = _V32_D_NOPE, _V32_D_ROPE
    assert D_qk == D_NOPE + D_ROPE, (
        f"D_qk={D_qk} must equal D_NOPE+D_ROPE={D_NOPE + D_ROPE} for V32"
    )
    assert H_q in (64, 128), f"H_q must be 64 or 128, got {H_q}"
    assert B > 0 and S_q > 0, f"B={B} and S_q={S_q} must be positive"

    # ----- Validate keyword-only knobs -------------------------------------
    assert is_fp8_kvcache is True, "FP8 KV path is the only V32 mode"
    assert h_kv == 1, f"h_kv must be 1 (MQA), got {h_kv}"
    assert d_v == _V32_D_NOPE, f"d_v must be {_V32_D_NOPE} for V32, got {d_v}"

    # ----- Validate kv_cache (accept 3-D or 4-D-with-h_kv=1) ---------------
    assert isinstance(kv_cache, torch.Tensor)
    if kv_cache.dim() == 4:
        assert kv_cache.shape[2] == 1, (
            f"kv_cache 4-D form must have h_kv=1, got h_kv={kv_cache.shape[2]}"
        )
        kv_cache = kv_cache.view(
            kv_cache.shape[0], kv_cache.shape[1], kv_cache.shape[3]
        )
    assert kv_cache.dim() == 3, (
        f"kv_cache must be 3-D [num_blocks, page_block_size, 656] (or 4-D "
        f"with h_kv=1), got dim={kv_cache.dim()}"
    )
    num_blocks, page_block_size, bytes_per_token = kv_cache.shape
    assert num_blocks > 0 and page_block_size > 0
    assert bytes_per_token == _V32_BYTES_PER_TOKEN, (
        f"bytes_per_token must be {_V32_BYTES_PER_TOKEN}, got {bytes_per_token}"
    )
    assert kv_cache.dtype in (torch.uint8, torch.int8, torch.float8_e4m3fn), (
        f"kv_cache dtype must be uint8/int8/float8_e4m3fn, got {kv_cache.dtype}"
    )
    assert kv_cache.is_contiguous(), (
        "kv_cache must be contiguous; the byte-reinterpretation views below "
        "depend on a uniform 656-B-stride row layout"
    )

    # ----- Validate indices ------------------------------------------------
    assert isinstance(indices, torch.Tensor)
    assert indices.dim() == 3 and indices.shape[:2] == (B, S_q), (
        f"indices must be [B={B}, S_q={S_q}, topk], got {tuple(indices.shape)}"
    )
    assert indices.dtype == torch.int32, (
        f"indices must be int32, got {indices.dtype}"
    )
    assert indices.stride(-1) == 1, "indices must have last-dim contiguous"
    topk = indices.shape[-1]
    assert topk > 0 and topk % _BLOCK_TOPK == 0, (
        f"sm_120 path requires topk % {_BLOCK_TOPK} == 0; got topk={topk}. "
        f"Upstream sm_90 cap is 64; this path halves to {_BLOCK_TOPK} per "
        f"T1.3 §8.2 Option A (smem budget)."
    )

    # ----- block_table / seq_lens are unused on FlashMLA sparse-decode -----
    # Accept either None or a tensor for API parity; we don't read them.
    _ = block_table
    _ = seq_lens

    # ----- sm_scale --------------------------------------------------------
    assert isinstance(sm_scale, (int, float)) and sm_scale > 0, (
        f"sm_scale must be a positive number, got {sm_scale}"
    )

    # ----- Build typed strided views of the 656-byte packed layout --------
    # We rely on .reshape + slice + .view(dtype) being all zero-copy. The
    # outer stride of each row remains 656 bytes; the typed views inherit
    # row strides 656 / sizeof(dtype) elements (656, 164, 328 respectively).
    #
    # Mirrors `tilelang_fp8_paged_mqa_logits` (`nsa/tilelang_kernel.py:893-897`),
    # which slices a packed FP8+FP32 cache the same way.
    n_total = num_blocks * page_block_size
    kv_2d = kv_cache.reshape(n_total, _V32_BYTES_PER_TOKEN)

    fp8_offset = 0
    scales_offset = _V32_D_NOPE  # 512
    rope_offset = _V32_D_NOPE + _V32_NUM_SCALES * 4  # 528

    kv_fp8 = kv_2d[:, fp8_offset:scales_offset].view(torch.float8_e4m3fn)
    kv_scales = kv_2d[:, scales_offset:rope_offset].view(torch.float32)
    kv_rope = kv_2d[:, rope_offset:_V32_BYTES_PER_TOKEN].view(torch.bfloat16)

    assert kv_fp8.shape == (n_total, _V32_D_NOPE)
    assert kv_scales.shape == (n_total, _V32_NUM_SCALES)
    assert kv_rope.shape == (n_total, _V32_D_ROPE)
    # Sanity-check the row strides we documented above.
    assert kv_fp8.stride() == (_V32_BYTES_PER_TOKEN, 1), (
        f"kv_fp8 stride sanity-check failed: {kv_fp8.stride()}"
    )
    assert kv_scales.stride() == (_V32_BYTES_PER_TOKEN // 4, 1), (
        f"kv_scales stride sanity-check failed: {kv_scales.stride()}"
    )
    assert kv_rope.stride() == (_V32_BYTES_PER_TOKEN // 2, 1), (
        f"kv_rope stride sanity-check failed: {kv_rope.stride()}"
    )

    # ----- Allocate outputs (kernel writes into these) --------------------
    out = q.new_empty((B, S_q, H_q, d_v), dtype=torch.bfloat16)
    lse = q.new_empty((B, S_q, H_q), dtype=torch.float32)

    # ----- JIT-compile (or cache-hit) and invoke --------------------------
    # We quantize sm_scale*log2e to int(1e9) to keep the LRU stable.
    sm_scale_log2e = float(sm_scale) * _LOG2E
    sm_scale_log2e_q = int(round(sm_scale_log2e * 1e9))

    kernel = _fp8_sparse_decode_kernel_sm120(
        h_q=H_q,
        d_v=d_v,
        d_nope=D_NOPE,
        d_rope=D_ROPE,
        block_topk=_BLOCK_TOPK,
        num_scales=_V32_NUM_SCALES,
        sm_scale_log2e_q=sm_scale_log2e_q,
    )
    kernel(q, kv_fp8, kv_scales, kv_rope, indices, out, lse)
    return out, lse


__all__ = ["tilelang_fp8_sparse_decode"]
