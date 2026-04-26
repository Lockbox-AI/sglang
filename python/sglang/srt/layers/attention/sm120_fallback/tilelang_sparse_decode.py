"""sm_120 TileLang FP8 sparse-decode kernel — V2 (sglang V4-Flash runtime layout).

This module re-implements FlashMLA's ``sparse_decode_fwd`` for sm_120
(RTX Pro 6000 Blackwell, RTX 5090) against **sglang's actual V4-Flash KV
cache layout** rather than FlashMLA's published-docs layout. The earlier
V1 kernel (``feat/sm120-tilelang-sparse-decode-kernel``) targeted the
656 B / 512-NoPE / FP32-scale layout from FlashMLA's V32 documentation;
``T2A.3-MERGE-STATUS.md`` discovered that sglang's actual quantizer at
``nsa/quant_k_cache_v4.py`` produces the 584 B / 448-NoPE / UE8M0-scale
layout, with a per-page region split (NoPE+RoPE per token in region A;
scales segregated to a per-page tail region B).

Phase 1 Run 4 revalidated T1.3 against sglang's runtime and Subagent C's
RUN-4C-PIVOT-STATUS confirmed no in-tree pivot exists. The structural
sm_120 redesign of V1 (single warpgroup, ``mma.m16n8k16``, BLOCK_TOPK=32
with smem-buffer fragment-layout workaround, online softmax in log2,
register-resident Q, single-buffered KV smem, invalid-index masking)
**survives unchanged** — only the layout-tied byte arithmetic, dim
constants, scale dequant formula, and the input/output dim contract
change.

V2 preserves the V1 module unchanged on the sister kernel branch for
forensic comparison; the parent merge step swaps which kernel branch
lands in ``feat/sm120-tilelang-sparse-decode``.

References
----------
- Spec: ``T1.3-sparse-decode-fwd-spec.md`` (Run 4 revalidated, especially
  §4.1 KV layout, §4.3 V dim, §4.4 Q dim, §6 algorithm walkthrough,
  §8.2 smem budget).
- Layout source of truth:
  ``python/sglang/srt/layers/attention/nsa/quant_k_cache_v4.py:71-90``
  (producer) and
  ``python/sglang/srt/layers/attention/nsa/index_buf_accessor_v4.py:15-31``
  (``NopeFp8RopeBf16Pack`` consumer).
- Per-page layout: ``index_buf_accessor_v4.py:50-188``
  (``_set_k_and_s_triton_kernel`` — region A then region B).
- Prior V1 kernel (preserved): branch
  ``feat/sm120-tilelang-sparse-decode-kernel @ a76b135``.

V4-Flash KV cache layout (sglang runtime, NOT FlashMLA published docs)
---------------------------------------------------------------------

Per logical token (584 B total)::

    448 B   k_nope_fp8           FP8 e4m3fn,    448 dims × 1 B
    128 B   k_rope_bf16          BF16,           64 dims × 2 B
      7 B   scale_k_nope_ue8m0   UE8M0 uint8,     7 tiles × 1 B
      1 B   pad                  scale_pad = 1 in DeepSeekV4SingleKVPool

UE8M0 dequant per 64-element NoPE tile: ``f32_scale = 2 ** (uint8 - 127)``.

**On-page two-region split** (page_size = 256 for SWA, 64 for sparse decode).
For one page::

    region A (NoPE+RoPE per token, interleaved):  bytes 0 .. page_size*576
        token i:
          [i*576 + 0   .. i*576 + 447]   k_nope_fp8 (448 B)
          [i*576 + 448 .. i*576 + 575]   k_rope_bf16 (128 B = 64×bf16)

    region B (scales per token, +1 B pad):        bytes page_size*576 .. page_size*584
        token i:
          [page_size*576 + i*8 + 0 .. + 6]   scale_k_nope_ue8m0 (7 B)
          [page_size*576 + i*8 + 7]          padding

    Per-page raw size:   page_size * 584
    Per-page padded:     ceil_div(page_size * 584, 576) * 576

The 4-D ``[num_pages, page_size, h_kv=1, 584]`` view that sglang's
radix backend hands to ``flash_mla_with_kvcache_entrypoint`` is a
*re-view* of the underlying 2-D ``[num_pages, bytes_per_page_padded]``
buffer with inner stride 584. **The 584 inner stride is wrong for the
actual byte layout** — token *i*'s NoPE bytes start at page-byte
``i*576``, not ``i*584``. Reading ``kv_cache[p, i, 0, 0:448]`` for
``i > 0`` returns garbage. The Python wrapper below detects the 4-D
form and re-views to the underlying 2-D buffer (region-A / region-B
extraction with the correct 576-byte per-token NoPE+RoPE stride).

Q dimensionality (sglang absorbed-latent, NOT FlashMLA packed)
-------------------------------------------------------------

sglang's V4-Flash compressed-attention dispatch passes
``q.shape[-1] == 512 = D_NOPE 448 + D_ROPE 64`` (verified by spot-check
on the live sm_120 instance, 2026-04-26). Q's last 64 dims **are
RoPE-rotated** (sample magnitudes ~1-5, not zero), matching K's BF16
RoPE half. The kernel computes:

    S = (Q[:, :448] @ K_nope_bf16^T) + (Q[:, 448:512] @ K_rope_bf16^T)

V1's wrapper zero-padded Q from 512 to 576 to satisfy a now-stale
``D_qk == 576`` assertion; that hack is dropped here. The V2 kernel
accepts ``D_qk == 512`` natively.

Output dim (head_dim_v = 512; K-NoPE = 448 ⇒ trailing 64 zero-padded)
-------------------------------------------------------------------

``model_runner.model_config.v_head_dim == 512`` for V4-Flash (verified
by spot-check). FlashMLA's V·P GEMM uses ``V == K_NoPE``, so the
natural output width is ``D_NOPE = 448`` in sglang's V4 layout. The
V·P contract still requires ``[B, S_q, H_q, 512]``, so the kernel
writes the 448-wide result into ``out[..., :448]`` and zero-pads
``out[..., 448:512]`` (T1.3 §4.3 option 1).

Live shared-memory accounting (V2; target ≤ 99 KB opt-in)
---------------------------------------------------------

::

    K_nope_smem   [BLOCK_TOPK=32, D_NOPE=448]      BF16   = 28 KB  (28,672 B)
    K_rope_smem   [BLOCK_TOPK=32, D_ROPE= 64]      BF16   =  4 KB  ( 4,096 B)
    S_smem        [H_Q≤128, BLOCK_TOPK=32]         BF16   ≤ 8 KB   (≤8,192 B)
    indices_smem  [BLOCK_TOPK=32]                  INT32  =  128 B
    pages_smem    [BLOCK_TOPK=32]                  INT32  =  128 B   (NEW for V2: divmod cache)
    offs_smem     [BLOCK_TOPK=32]                  INT32  =  128 B   (NEW for V2)
    is_kv_valid   [BLOCK_TOPK=32]                  bool   =   32 B
    {alpha,m_i,m_i_prev,sumexp}_smem [H_Q]         FP32   ≤ 2 KB   (≤2,048 B)
    -------------------------------------------------------------
    Total live smem (H_Q=64, V4-Flash production)  ≈ 37 KB  (37,920 B)
    Total live smem (H_Q=128, V4-Pro pre-FP4)      ≈ 43 KB  (44,000 B)

Both fit comfortably under sm_120's 48 KB default cap; no opt-in
needed for either. The drop from V1's 41/47 KB tracks the K_nope_smem
shrink from [32, 512] to [32, 448] (28 KB vs 32 KB).

V1-vs-V2 delta summary
----------------------

==================================  =======================  =======================
What                                V1 (FlashMLA-spec docs)  V2 (sglang V4 runtime)
==================================  =======================  =======================
``BYTES_PER_TOKEN``                 656                      584
``D_NOPE``                          512                      448
``D_ROPE``                          64                       64 (unchanged)
Quant tile width                    128                      64
``NUM_NOPE_TILES``                  4                        7
Scale dtype / size                  FP32 (4 B/scale)         UE8M0 uint8 (1 B/scale, +1 B pad/tok)
Scale dequant                       ``f32 * scale``          ``f32 * exp2(u8 - 127)``
``D_qk`` (Q-side)                   576 (NoPE+RoPE concat)   512 (sglang absorbed-latent)
``Q_rope`` semantics                Zero-padded by V1 shim   Real RoPE half from sglang
``D_v`` (output width)              512 (matches K-NoPE)     512 (K-NoPE 448 → zero-pad tail)
On-page byte layout                 Single 656 B/token run   Two regions per page (576 B+8 B)
Gather pattern                      1-D flat index lookup    2-D ``(page, off)`` lookup
==================================  =======================  =======================

Pre-author runtime spot-check evidence (2026-04-26, ec2-user@3.84.68.158)
-----------------------------------------------------------------------

T1.3 §14 #11 (Scale region layout): per-page tail (region B at offset
``page_size * 576``), NOT per-token interleaved. Stride within region B
is ``scale_dim + scale_pad = 8`` per token. Confirmed via direct
inspection of ``DeepSeekV4SingleKVPool``: ``bytes_per_page_padded ==
37440`` for ``page_size == 64``, of which 36864 = region A and 512 =
region B (+64 pad to align to 576-byte multiple).

T1.3 §14 #12 (Q's last 64 dims): patched the live shim to print
``q.shape, q.dtype, q[0, 0, 0, 448:512]`` on first call. Result::

    q.shape=(6, 1, 64, 512) q.dtype=torch.bfloat16
    q[0,0,0,:448].abs().mean()    = 0.613709  (NoPE-like magnitudes)
    q[0,0,0,448:512].abs().mean() = 1.362188  (RoPE-rotated; large magnitudes)
    q[0,0,0,448:512] sample first 8: [-5.40625, -4.9375, 3.671875, ...]

⇒ Q's last 64 dims ARE Q-RoPE (already rotated upstream by sglang's
compressed path). Kernel must include the Q-RoPE × K-RoPE contribution.

T1.3 §14 #13 (V dim): ``head_dim_v == 512`` (resolved from
``model_runner.model_config.v_head_dim``, which derives from
``head_dim - qk_rope_head_dim`` under ``SGLANG_DSV4_MODE=2604`` —
the V4-Flash default). ``v_head_dim`` is *not* in V4-Flash's
``config.json`` so it falls back to ``head_dim = 512`` per
``model_config.py:509``. Output shape ``[B, S_q, H_q, 512]``, with
``out[..., 448:512] = 0``.
"""

import functools
from typing import Optional, Tuple

import tilelang
import tilelang.language as T
import torch

tilelang.set_log_level("WARNING")


# ---------------------------------------------------------------------------
# V4-Flash packed KV layout (T1.3 Run 4 §4.1; verified at runtime).
# ---------------------------------------------------------------------------
_V4_D_NOPE: int = 448
_V4_D_ROPE: int = 64
_V4_D_QK: int = _V4_D_NOPE + _V4_D_ROPE  # 512 (sglang absorbed-latent Q dim)
_V4_QUANT_TILE_SIZE: int = 64
_V4_NUM_SCALES: int = _V4_D_NOPE // _V4_QUANT_TILE_SIZE  # 7
_V4_SCALE_PAD: int = 1
_V4_PADDED_SCALE_BYTES_PER_TOKEN: int = _V4_NUM_SCALES + _V4_SCALE_PAD  # 8
_V4_NOPE_ROPE_BYTES_PER_TOKEN: int = _V4_D_NOPE + _V4_D_ROPE * 2  # 576
_V4_BYTES_PER_TOKEN: int = (
    _V4_NOPE_ROPE_BYTES_PER_TOKEN  # 576 (NoPE FP8 + RoPE BF16)
    + _V4_PADDED_SCALE_BYTES_PER_TOKEN  # 8 (UE8M0 scales + pad)
)
assert _V4_BYTES_PER_TOKEN == 584

# Output width — matches model_config.v_head_dim (512 for V4-Flash); see §4.3.
_V4_D_V: int = 512

# sm_120 design constants (T1.3 §8.2 Option A; see module docstring).
_BLOCK_TOPK: int = 32

# log2(e) — used to convert sm_scale * x → exp2(...) for online softmax.
_LOG2E: float = 1.44269504

# UE8M0 bias — `scale_fp32 = 2 ** (scale_uint8 - _UE8M0_BIAS)`.
_UE8M0_BIAS: float = 127.0


# TileLang dtype aliases (match conventions in nsa/tilelang_kernel.py).
_BF16 = "bfloat16"
_FP8 = "float8_e4m3"
_UINT8 = "uint8"
_FP32 = "float32"
_INT32 = "int32"


_pass_configs = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
}


@tilelang.jit(pass_configs=_pass_configs)
def _fp8_sparse_decode_kernel_sm120_v2(
    h_q: int,
    d_v: int,
    d_nope: int,
    d_rope: int,
    page_size: int,
    block_topk: int,
    num_scales: int,
    sm_scale_log2e_q: int,
    threads: int = 256,
    num_stages: int = 1,
):
    """Build (and JIT-cache) the V2 TileLang kernel.

    The decoration pattern mirrors V1 / the in-tree
    ``sparse_attention_fwd_kernel_v1`` convention.
    """
    sm_scale_log2e: float = sm_scale_log2e_q * 1e-9
    d_qk: int = d_nope + d_rope

    B = T.symbolic("B")
    S_q = T.symbolic("S_q")
    NumPages = T.symbolic("NumPages")
    topk = T.symbolic("topk")
    NI = T.ceildiv(topk, block_topk)

    # Dynamic per-row strides for the three byte-reinterpreted KV views.
    # Concrete values at the call site (page_size=256, V4 layout):
    #   FP8  : stride dim 0 = bytes_per_page_padded (e.g. 149760 for ps=256)
    #          stride dim 1 = 576 (NoPE+RoPE inner stride per token in region A)
    #   BF16 : stride dim 0 = bytes_per_page_padded / 2  (in bf16 elements)
    #          stride dim 1 = 288  (576 B / 2)
    #   uint8: stride dim 0 = bytes_per_page_padded (uint8 = 1 B)
    #          stride dim 1 = 8 (PADDED_SCALE_BYTES_PER_TOKEN)
    s_kvf_p, s_kvr_p, s_kvs_p = T.dynamic("s_kvf_p, s_kvr_p, s_kvs_p")
    s_kvf_t, s_kvr_t, s_kvs_t = T.dynamic("s_kvf_t, s_kvr_t, s_kvs_t")

    @T.prim_func
    def main(
        Q: T.Tensor[(B, S_q, h_q, d_qk), _BF16],  # type: ignore[name-defined]
        KV_fp8: T.StridedTensor[(NumPages, page_size, d_nope), (s_kvf_p, s_kvf_t, 1), _FP8],  # type: ignore[name-defined]
        KV_rope: T.StridedTensor[(NumPages, page_size, d_rope), (s_kvr_p, s_kvr_t, 1), _BF16],  # type: ignore[name-defined]
        KV_scales: T.StridedTensor[(NumPages, page_size, num_scales), (s_kvs_p, s_kvs_t, 1), _UINT8],  # type: ignore[name-defined]
        Indices: T.Tensor[(B, S_q, topk), _INT32],  # type: ignore[name-defined]
        Out: T.Tensor[(B, S_q, h_q, d_v), _BF16],  # type: ignore[name-defined]
        Lse: T.Tensor[(B, S_q, h_q), _FP32],  # type: ignore[name-defined]
    ):
        with T.Kernel(B, S_q, threads=threads) as (b_i, s_i):
            # -- Smem (live ~37 KB; see module docstring) ----------------
            K_nope_smem = T.alloc_shared([block_topk, d_nope], _BF16)
            K_rope_smem = T.alloc_shared([block_topk, d_rope], _BF16)
            S_smem = T.alloc_shared([h_q, block_topk], _BF16)
            indices_smem = T.alloc_shared([block_topk], _INT32)
            pages_smem = T.alloc_shared([block_topk], _INT32)
            offs_smem = T.alloc_shared([block_topk], _INT32)
            is_kv_valid_smem = T.alloc_shared([block_topk], "bool")
            # Tiny scratch buffers (≤ 2 KB total) for the online-softmax
            # state — kept in smem to BREAK the fragment-layout-inference
            # chain that otherwise conflicts at BLOCK_TOPK=32 (TileLang
            # 0.1.8 quirk; same workaround as V1 + nsa/tilelang_kernel.py:481).
            alpha_smem = T.alloc_shared([h_q], _FP32)
            m_i_smem = T.alloc_shared([h_q], _FP32)
            m_i_prev_smem = T.alloc_shared([h_q], _FP32)
            sumexp_smem = T.alloc_shared([h_q], _FP32)

            # -- Fragments (registers) -----------------------------------
            # Q stays in registers across all topk iterations; loaded once.
            Q_nope_frag = T.alloc_fragment([h_q, d_nope], _BF16)
            Q_rope_frag = T.alloc_fragment([h_q, d_rope], _BF16)
            # UE8M0 scales: load uint8, cast to FP32 with bias subtraction
            # before being used as the FP8→BF16 dequant multiplier.
            K_scales_u8_frag = T.alloc_fragment([block_topk, num_scales], _UINT8)
            K_scales_frag = T.alloc_fragment([block_topk, num_scales], _FP32)

            # acc_o is sized to D_NOPE (448), NOT D_V (512). The V·P GEMM
            # produces [H_q, D_NOPE]; the trailing 64 dims of out[..., 448:512]
            # are written separately as zero (T1.3 §4.3 option 1).
            acc_o = T.alloc_fragment([h_q, d_nope], _FP32)
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
            # V4-Flash absorbed-latent: q[..., :448] = Q-NoPE,
            # q[..., 448:512] = Q-RoPE (RoPE-rotated upstream).
            T.copy(Q[b_i, s_i, :, 0:d_nope], Q_nope_frag)
            T.copy(Q[b_i, s_i, :, d_nope:d_qk], Q_rope_frag)

            for i_i in T.Pipelined(NI, num_stages=num_stages):
                # -- Read indices, divmod into (page, off), validity mask --
                # Invalid entries are encoded as -1 upstream (T1.3 §4).
                # Clamp negative to 0 so the gathers stay in-bounds; the
                # is_kv_valid_smem mask drops those lanes to -INF in acc_s
                # before the QK softmax, so garbage K reads don't pollute
                # the result.
                for bi_i in T.Parallel(block_topk):
                    raw = Indices[b_i, s_i, i_i * block_topk + bi_i]
                    is_kv_valid_smem[bi_i] = raw >= 0
                    safe = T.max(raw, 0)
                    indices_smem[bi_i] = safe
                    pages_smem[bi_i] = safe // page_size
                    offs_smem[bi_i] = safe % page_size

                # -- Load 7 UE8M0 scales (1 byte each) per token ---------
                for bi_i, ns_i in T.Parallel(block_topk, num_scales):
                    K_scales_u8_frag[bi_i, ns_i] = KV_scales[
                        pages_smem[bi_i], offs_smem[bi_i], ns_i
                    ]

                # -- UE8M0 → FP32 dequant (T1.3 §4.1.3) ------------------
                # `scale_fp32 = 2 ** (scale_uint8 - 127)`.
                for bi_i, ns_i in T.Parallel(block_topk, num_scales):
                    K_scales_frag[bi_i, ns_i] = T.exp2(
                        T.Cast(_FP32, K_scales_u8_frag[bi_i, ns_i]) - _UE8M0_BIAS
                    )

                # -- Gather + dequantize FP8 NoPE → BF16 (per-64 scale) --
                # Each FP8 element gets multiplied by its quant-tile scale
                # (7 scales × 64 elements = 448 NoPE dims). Match the
                # producer's per-tile quant: ``scale_pow2 = exp2(ceil(log2(max_abs/FP8_MAX)))``.
                for bi_i, d_i in T.Parallel(block_topk, d_nope):
                    fp8_val = KV_fp8[pages_smem[bi_i], offs_smem[bi_i], d_i]
                    K_nope_smem[bi_i, d_i] = T.Cast(
                        _BF16,
                        T.Cast(_FP32, fp8_val)
                        * K_scales_frag[bi_i, d_i // _V4_QUANT_TILE_SIZE],
                    )

                # -- Gather BF16 RoPE (no dequant; already bf16) ---------
                for bi_i, d_i in T.Parallel(block_topk, d_rope):
                    K_rope_smem[bi_i, d_i] = KV_rope[
                        pages_smem[bi_i], offs_smem[bi_i], d_i
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
                # on sm_120 (4th-gen tensor core). Reduce dim is
                # 448 (NoPE) + 64 (RoPE) = 512.
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
                # at BLOCK_TOPK=32 (V1 quirk; same workaround).
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
                for h_i, d_i in T.Parallel(h_q, d_nope):
                    acc_o[h_i, d_i] = acc_o[h_i, d_i] * alpha_smem[h_i]

                # -- S·V where V == K_nope_bf16 (T1.3 §6 step 7) ---------
                # Note: V is the dequantized NoPE tile (448 dims), NOT
                # the full K_qk. FlashMLA's reference is identical
                # (`splitkv_mla.cuh:613` comment: "We do not need to mask
                # the RoPE part for V3.2 since it isn't involved in the
                # SV gemm").
                T.copy(acc_s, S_smem)
                T.gemm(
                    S_smem,
                    K_nope_smem,
                    acc_o,
                    policy=T.GemmWarpPolicy.FullCol,
                )

            # -- Final rescale + LSE write -------------------------------
            T.copy(sumexp, sumexp_smem)
            for h_i, d_i in T.Parallel(h_q, d_nope):
                acc_o[h_i, d_i] = acc_o[h_i, d_i] / sumexp_smem[h_i]
            for h_i in T.Parallel(h_q):
                # Convert back to natural-log LSE (T1.3 §5.1, §6).
                Lse[b_i, s_i, h_i] = (
                    T.log2(sumexp_smem[h_i]) + m_i_smem[h_i] * sm_scale_log2e
                ) / _LOG2E

            # Write the 448-wide accumulator into out[..., :D_NOPE], then
            # zero-pad out[..., D_NOPE:D_V] (the 64-dim K-NoPE-vs-d_v gap;
            # T1.3 §4.3 option 1). Implicit FP32→BF16 conversion via T.copy
            # (matches V1 line 357).
            T.copy(acc_o, Out[b_i, s_i, :, 0:d_nope])
            for h_i, d_i in T.Parallel(h_q, d_v - d_nope):
                Out[b_i, s_i, h_i, d_nope + d_i] = T.Cast(_BF16, 0.0)

    return main


def _underlying_2d_buffer(kv_4d: torch.Tensor) -> torch.Tensor:
    """Recover the underlying ``[num_pages, bytes_per_page_padded]`` buffer.

    sglang's radix backend hands us a 4-D
    ``[num_pages, page_size, h_kv=1, 584]`` view of a 2-D buffer with
    *padded* per-page stride
    (``bytes_per_page_padded = ceil_div(page_size * 584, 576) * 576``).
    The 4-D inner stride 584 is **wrong** for V4's actual byte layout
    (token *i*'s NoPE bytes start at page-byte ``i * 576``, not
    ``i * 584``). We re-construct a 2-D view of the underlying buffer
    so the per-page region split (T1.3 §4.1.2) can be applied.

    Per ``mem_cache/deepseekv4_memory_pool.py:103-115`` the underlying
    storage is exactly ``[num_pages, bytes_per_page_padded]`` ``uint8``;
    this helper restores that view.
    """
    assert kv_4d.dim() == 4, f"expected 4-D kv view, got dim={kv_4d.dim()}"
    num_pages, page_size, h_kv, bpt = kv_4d.shape
    assert h_kv == 1, f"h_kv must be 1, got {h_kv}"
    assert bpt == _V4_BYTES_PER_TOKEN, (
        f"per-token logical record must be {_V4_BYTES_PER_TOKEN} B, got {bpt}"
    )
    # The dim(0) stride of the 4-D view = bytes_per_page_padded (in bytes,
    # since the underlying dtype's element_size is 1 for uint8/int8/fp8).
    bppp_in_elements = kv_4d.stride(0)
    elem_size = kv_4d.element_size()
    # For uint8/int8/fp8_e4m3fn, elem_size == 1 ⇒ bppp_in_bytes == bppp_in_elements.
    bytes_per_page_padded = bppp_in_elements * elem_size
    # Build a view of the underlying (num_pages, bytes_per_page_padded)
    # buffer that strips dim 1+2+3 collapse into the inner stride. We
    # use as_strided which is zero-copy and tolerates the padded dim-0
    # stride (149760 vs page_size*584=149504).
    underlying = kv_4d.as_strided(
        size=(num_pages, bytes_per_page_padded),
        stride=(bppp_in_elements, 1),
    )
    return underlying


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
    """sm_120 TileLang FP8 sparse-decode (V4-Flash / K2a) — single-shot decode.

    Drop-in replacement for FlashMLA's ``sparse_decode_fwd`` on sm_120,
    consuming sglang's actual V4-Flash KV cache layout (NOT FlashMLA's
    published-docs 656-byte layout). See module docstring for the
    layout, dim, and Q/V interpretation evidence.

    Parameters
    ----------
    q : ``bfloat16 [B, S_q, H_q, D_qk=512]``
        Query tensor. ``D_qk == 512`` for sglang's V4-Flash dispatch
        (absorbed-latent: NoPE 448 + RoPE 64). Last-dim contiguous.
        ``H_q ∈ {64, 128}`` (V4-Flash uses 64).
    kv_cache : ``[num_pages, page_size, h_kv=1, 584]`` view of a 2-D
        ``[num_pages, bytes_per_page_padded]`` underlying buffer.
        Each 584-byte logical token record holds NoPE 448 B FP8 +
        RoPE 128 B BF16 *interleaved per token in region A* and the
        7 + 1 pad UE8M0 scale bytes in *region B at per-page offset
        page_size * 576*. dtype must be uint8 / int8 / float8_e4m3fn.
    block_table, seq_lens : optional, accepted for API parity, unused.
    indices : ``int32 [B, S_q, topk]``
        Flat token indices into the paged KV cache:
        ``indices[b, s, k] = page_idx * page_size + offset_in_page``.
        Invalid entries: ``-1``. ``topk`` must be a multiple of 32.
    sm_scale : pre-softmax scale (typically ``1/sqrt(D_qk)``).
    is_fp8_kvcache : must be True (V32 path).
    h_kv : must be 1 (MQA).
    d_v : must be 512 (V4-Flash contract; out's trailing 64 dims are
        zero-padded since K-NoPE = 448).

    Returns
    -------
    out : ``bfloat16 [B, S_q, H_q, D_v=512]``
        ``out[..., 448:512] = 0`` (T1.3 §4.3 option 1).
    lse : ``float32 [B, S_q, H_q]`` natural-log LSE.
    """
    # ----- Validate q ------------------------------------------------------
    assert isinstance(q, torch.Tensor), f"q must be a Tensor, got {type(q)}"
    assert q.dim() == 4, f"q must be [B, S_q, H_q, D_qk], got dim={q.dim()}"
    assert q.dtype == torch.bfloat16, f"q must be bfloat16, got {q.dtype}"
    assert q.stride(-1) == 1, "q must have last-dim contiguous"
    B, S_q, H_q, D_qk = q.shape
    D_NOPE, D_ROPE = _V4_D_NOPE, _V4_D_ROPE
    # V2: kernel natively accepts D_qk=512 (sglang absorbed-latent).
    # No zero-pad to 576 (V1's `605cc3e` hack) — the kernel uses Q-RoPE
    # via the second GEMM Q[..., 448:512] @ K_rope^T.
    assert D_qk == _V4_D_QK, (
        f"D_qk={D_qk} must equal {_V4_D_QK} (sglang absorbed-latent: "
        f"NoPE {_V4_D_NOPE} + RoPE {_V4_D_ROPE}); see module docstring §Q dim."
    )
    assert H_q in (64, 128), f"H_q must be 64 or 128, got {H_q}"
    assert B > 0 and S_q > 0, f"B={B} and S_q={S_q} must be positive"

    # ----- Validate keyword-only knobs -------------------------------------
    assert is_fp8_kvcache is True, "FP8 KV path is the only V32 mode"
    assert h_kv == 1, f"h_kv must be 1 (MQA), got {h_kv}"
    assert d_v == _V4_D_V, f"d_v must be {_V4_D_V} for V4-Flash, got {d_v}"

    # ----- Validate kv_cache (accept 4-D V4-Flash form) -------------------
    assert isinstance(kv_cache, torch.Tensor)
    assert kv_cache.dim() == 4, (
        f"kv_cache must be 4-D [num_pages, page_size, h_kv=1, 584]; "
        f"got dim={kv_cache.dim()}"
    )
    num_pages, page_size, h_kv_dim, bytes_per_token = kv_cache.shape
    assert h_kv_dim == 1, f"kv_cache h_kv dim must be 1, got {h_kv_dim}"
    assert num_pages > 0 and page_size > 0
    assert bytes_per_token == _V4_BYTES_PER_TOKEN, (
        f"bytes_per_token must be {_V4_BYTES_PER_TOKEN} (sglang V4 layout), "
        f"got {bytes_per_token}"
    )
    assert kv_cache.dtype in (torch.uint8, torch.int8, torch.float8_e4m3fn), (
        f"kv_cache dtype must be uint8/int8/float8_e4m3fn, got {kv_cache.dtype}"
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
    _ = block_table
    _ = seq_lens

    # ----- sm_scale --------------------------------------------------------
    assert isinstance(sm_scale, (int, float)) and sm_scale > 0, (
        f"sm_scale must be a positive number, got {sm_scale}"
    )

    # ----- Build typed strided views of the V4 layout ---------------------
    # Step 1: re-view the 4-D `[num_pages, page_size, 1, 584]` as the
    # underlying 2-D `[num_pages, bytes_per_page_padded]` raw buffer.
    underlying_2d = _underlying_2d_buffer(kv_cache)  # uint8/int8/fp8 [num_pages, bppp]
    bppp = underlying_2d.shape[1]
    # Sanity: bppp must accommodate region A (page_size * 576) + region B
    # (page_size * 8) + optional 0..575 byte pad to align to a multiple of 576.
    expected_min = page_size * _V4_NOPE_ROPE_BYTES_PER_TOKEN + page_size * _V4_PADDED_SCALE_BYTES_PER_TOKEN
    assert bppp >= expected_min, (
        f"underlying bytes_per_page_padded={bppp} < expected_min={expected_min} "
        f"(page_size={page_size} * (576 + 8))"
    )

    # Step 2: extract region A (NoPE+RoPE per token, interleaved) as a
    # 3-D `[num_pages, page_size, 576]` view. Bytes 0..(page_size*576-1)
    # of each row.
    region_a_bytes = page_size * _V4_NOPE_ROPE_BYTES_PER_TOKEN
    # underlying_2d[:, :region_a_bytes] is shape (num_pages, region_a_bytes)
    # with strides (bppp_elem, 1). Reshape to (num_pages, page_size, 576).
    region_a = underlying_2d[:, :region_a_bytes].view(
        num_pages, page_size, _V4_NOPE_ROPE_BYTES_PER_TOKEN
    )
    # Slice NoPE + RoPE within each token:
    #   bytes [0, 448)   → FP8 NoPE
    #   bytes [448, 576) → BF16 RoPE (64 elements × 2 B)
    kv_fp8 = region_a[:, :, :_V4_D_NOPE].view(torch.float8_e4m3fn)
    kv_rope_bytes = region_a[:, :, _V4_D_NOPE:_V4_NOPE_ROPE_BYTES_PER_TOKEN].contiguous()
    # ^^^ .contiguous() because .view(bf16) requires the last-dim slice to be
    # contiguous in memory; the slice is contiguous within each token (stride 1)
    # but the view machinery may complain if dim 0/1 strides aren't bf16-aligned.
    # The contiguous() call materialises the rope bytes into a fresh buffer
    # (extra ~num_pages*page_size*128 bytes). For V4-Flash production this is
    # ~1136*256*128 = 37 MB — small relative to the 6 GB KV cache, but
    # measurably non-zero. TODO(perf): use as_strided to avoid the copy if
    # TileLang accepts the strided BF16 view directly.
    # Actually, the slice `region_a[:, :, _V4_D_NOPE:]` has strides
    # (bppp, 576, 1) which IS contiguous in dim 2 — so .view(bf16) should
    # just work without the .contiguous() copy. Try without first; fall
    # back to .contiguous() if .view fails.
    rope_slice = region_a[:, :, _V4_D_NOPE:_V4_NOPE_ROPE_BYTES_PER_TOKEN]
    try:
        kv_rope = rope_slice.view(torch.bfloat16)
        # Reshape last dim from 128 bytes → 64 bf16 elements (auto via .view).
    except RuntimeError:
        # Fallback: materialise and view.
        kv_rope = rope_slice.contiguous().view(torch.bfloat16)

    # Step 3: extract region B (scales) as a 3-D `[num_pages, page_size, 8]`
    # view. Bytes [page_size*576, page_size*584).
    region_b_bytes = page_size * _V4_PADDED_SCALE_BYTES_PER_TOKEN
    region_b = underlying_2d[
        :, region_a_bytes : region_a_bytes + region_b_bytes
    ].view(num_pages, page_size, _V4_PADDED_SCALE_BYTES_PER_TOKEN)
    # Take the first 7 of 8 bytes per token (skip the 1-byte pad).
    # Also reinterpret as uint8 (in case original dtype was fp8/int8;
    # uint8 is the canonical storage for the UE8M0 scale bytes per
    # `nsa/index_buf_accessor_v4.py:113-114`).
    if region_b.dtype != torch.uint8:
        region_b = region_b.view(torch.uint8)
    kv_scales = region_b[:, :, :_V4_NUM_SCALES]

    # Sanity-check shapes + dtypes (dev-only; cheap).
    assert kv_fp8.shape == (num_pages, page_size, _V4_D_NOPE)
    assert kv_fp8.dtype == torch.float8_e4m3fn
    assert kv_rope.shape == (num_pages, page_size, _V4_D_ROPE)
    assert kv_rope.dtype == torch.bfloat16
    assert kv_scales.shape == (num_pages, page_size, _V4_NUM_SCALES)
    assert kv_scales.dtype == torch.uint8

    # ----- Allocate outputs (kernel writes into these) --------------------
    out = q.new_empty((B, S_q, H_q, d_v), dtype=torch.bfloat16)
    lse = q.new_empty((B, S_q, H_q), dtype=torch.float32)

    # ----- JIT-compile (or cache-hit) and invoke --------------------------
    # We quantize sm_scale*log2e to int(1e9) to keep the LRU stable.
    sm_scale_log2e = float(sm_scale) * _LOG2E
    sm_scale_log2e_q = int(round(sm_scale_log2e * 1e9))

    kernel = _fp8_sparse_decode_kernel_sm120_v2(
        h_q=H_q,
        d_v=d_v,
        d_nope=D_NOPE,
        d_rope=D_ROPE,
        page_size=page_size,
        block_topk=_BLOCK_TOPK,
        num_scales=_V4_NUM_SCALES,
        sm_scale_log2e_q=sm_scale_log2e_q,
    )
    kernel(q, kv_fp8, kv_rope, kv_scales, indices, out, lse)
    return out, lse


__all__ = ["tilelang_fp8_sparse_decode"]
