"""Pure-PyTorch eager reference for FlashMLA's ``sparse_decode_fwd`` (V32).

**This file is the test oracle, not a production kernel.** It computes
the same attention output as
:func:`sglang.srt.layers.attention.sm120_fallback.sparse_decode_fp8_triton`
(K2a) using only ``torch.bmm`` / ``torch.softmax`` / elementwise math —
no Triton, no TileLang, no TMA, no FP8 hardware. Numerical correctness
is the only objective; the implementation is intentionally slow and
per-batch-row.

Algorithm faithfully follows
``T1.3-sparse-decode-fwd-spec.md §6`` (the "decode_one" pseudocode), which
in turn matches the kernel-internal arithmetic in
``FlashMLA/csrc/sm90/decode/sparse_fp8/splitkv_mla.cuh`` SHA ``71c73792``.

Numerical noise floor expected when comparing
``reference vs. triton``:

- FP8 ``e4m3`` mantissa is 3 bits → ~12.5% relative quantization error
  per element worst case, but per-128 BF16 scaling brings the effective
  per-tile error well below ~0.5% RMS (see
  `cuda-toolkit-distilled-for-sm120-work.md §3`).
- BF16 reduction order is non-associative
  (`CUDA C++ Best Practices Guide §7.3.2`).
- Final ``rtol=1e-3, atol=1e-3`` is the project-wide FP8 attention gate
  ([`cuda-toolkit-distilled §3`](
  ../../../../../../../source-artifacts/open-weights-benchmarking-1/cuda-toolkit-distilled-for-sm120-work.md))
  and matches the values used by
  ``test/registered/attention/test_triton_attention_kernels.py``
  upstream and the ``nvmath-python`` FP8 matmul tutorial.

The reference does **not** implement split-KV — it is the merged /
no-split path. The kernel's split-KV partitioning + combine is purely
a perf optimization; the merged math is what callers see. So this
reference is the right oracle for the public ``(out, lse)`` returned
by the K2a entrypoint.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch

# V32 (V4-Flash) layout constants — duplicated from
# ``sparse_decode_fp8_triton`` so the reference is self-contained and
# can't silently diverge from a stub the test-file imports.
_V32_D_NOPE = 512
_V32_D_ROPE = 64
_V32_D_QK = _V32_D_NOPE + _V32_D_ROPE  # 576
_V32_D_V = _V32_D_NOPE  # 512 (SV uses K_nope only; spec §6 step 7)
_V32_NUM_SCALES = _V32_D_NOPE // 128  # 4 per-128 fp32 scales / token
_V32_BYTES_PER_TOKEN = (
    _V32_D_NOPE  # 512 fp8 NoPE
    + _V32_NUM_SCALES * 4  # 16 fp32 scales
    + _V32_D_ROPE * 2  # 128 bf16 RoPE
)
assert _V32_BYTES_PER_TOKEN == 656

# Byte offsets inside one 656-byte token record. Verified against
# ``splitkv_mla.cuh:545,547,602`` for FlashMLA SHA ``71c73792``.
_OFF_NOPE = 0
_OFF_SCALES = _V32_D_NOPE  # 512
_OFF_ROPE = _OFF_SCALES + _V32_NUM_SCALES * 4  # 528


def _decode_one_position(
    Q_bh: torch.Tensor,
    indices_bh: torch.Tensor,
    k_cache: torch.Tensor,
    page_block_size: int,
    softmax_scale: float,
    attn_sink: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute attention for a single ``(batch, s_q)`` decode position.

    Parameters
    ----------
    Q_bh : ``bfloat16 [H_q, D_qk=576]``
    indices_bh : ``int32 [topk]`` flat token indices (``-1`` = invalid)
    k_cache : ``uint8 [num_blocks, page_block_size, 1, 656]`` packed cache
    page_block_size : KV cache page block size (typically 64)
    softmax_scale : pre-softmax scaling (1/sqrt(d_qk))
    attn_sink : optional ``float32 [H_q]`` per-head log-domain sink

    Returns
    -------
    out_bh : ``bfloat16 [H_q, D_v=512]``
    lse_bh : ``float32 [H_q]`` natural-log lse
    """
    H_q, D_qk = Q_bh.shape
    assert D_qk == _V32_D_QK
    T = indices_bh.shape[0]

    # ------------------------------------------------------------------
    # 1. Recover (block, off_in_block) from flat indices.
    #    Spec §6 step 1 + ``splitkv_mla.cuh:538-539``.
    # ------------------------------------------------------------------
    flat_idx = indices_bh.to(torch.int64)
    invalid = flat_idx == -1
    safe_idx = flat_idx.clamp(min=0)
    block_idx = safe_idx // page_block_size
    off_in_block = safe_idx % page_block_size

    # Bound block_idx to a valid row to avoid OOB; the kernel relies on
    # the upstream Lightning Indexer to never emit positive OOB indices,
    # but we mirror the safe-load behaviour around -1 explicitly.
    num_blocks = k_cache.shape[0]
    block_idx = block_idx.clamp(max=num_blocks - 1)

    # ------------------------------------------------------------------
    # 2. Gather raw KV bytes — [T, 656].
    # ------------------------------------------------------------------
    raw = k_cache[block_idx, off_in_block, 0, :].contiguous()
    if raw.dtype != torch.uint8:
        # Accept uint8 / int8 / float8_e4m3fn (per stub assertion). The
        # reference operates on raw bytes so we always normalise to uint8
        # via untyped storage view. The cleanest route is .view(uint8)
        # since all three dtypes are 1 byte per element.
        raw = raw.view(torch.uint8)
    assert raw.shape == (T, _V32_BYTES_PER_TOKEN), (
        f"raw shape {tuple(raw.shape)} != ({T}, {_V32_BYTES_PER_TOKEN})"
    )

    # ------------------------------------------------------------------
    # 3. Decompose: [FP8 NoPE | FP32 scales | BF16 RoPE]
    #    Spec §4.1; verified against ``splitkv_mla.cuh:545,547,602``.
    # ------------------------------------------------------------------
    nope_bytes = raw[:, _OFF_NOPE:_OFF_SCALES].contiguous()
    scales_bytes = raw[:, _OFF_SCALES:_OFF_ROPE].contiguous()
    rope_bytes = raw[:, _OFF_ROPE:_V32_BYTES_PER_TOKEN].contiguous()

    # FP8 NoPE — keep bytes; cast via .view + .to(float32).
    K_nope_fp8 = nope_bytes.view(torch.float8_e4m3fn)  # [T, 512]
    K_scales = scales_bytes.view(torch.float32).reshape(T, _V32_NUM_SCALES)
    K_rope = rope_bytes.view(torch.bfloat16).reshape(T, _V32_D_ROPE)

    # ------------------------------------------------------------------
    # 4. Dequant FP8 NoPE -> BF16 with per-128 scale.
    #    Matches ``cvt_fp8x8_bf16x8(data, scale_bf162)`` at
    #    ``components/dequant.h:21-34``.
    # ------------------------------------------------------------------
    K_nope_fp32 = K_nope_fp8.to(torch.float32)  # [T, 512]
    K_nope_dequant = (
        K_nope_fp32.reshape(T, _V32_NUM_SCALES, 128) * K_scales.unsqueeze(-1)
    )
    K_nope_bf16 = K_nope_dequant.reshape(T, _V32_D_NOPE).to(torch.bfloat16)

    # ------------------------------------------------------------------
    # 5. Concatenate NoPE + RoPE -> full K [T, 576].
    # ------------------------------------------------------------------
    K = torch.cat([K_nope_bf16, K_rope], dim=-1)  # [T, 576] bf16

    # Zero-out invalid lanes in K so they contribute nothing to S
    # (matches the kernel's ``is_kv_valid`` mask side-effect at
    # ``splitkv_mla.cuh:577-595``).
    if invalid.any():
        K = K.clone()
        K[invalid] = 0
        K_nope_bf16 = K_nope_bf16.clone()
        K_nope_bf16[invalid] = 0

    # ------------------------------------------------------------------
    # 6. Q · K^T in fp32 (kernel uses bf16 mma → fp32 accum).
    # ------------------------------------------------------------------
    Q_fp32 = Q_bh.float()  # [H_q, 576]
    K_fp32 = K.float()  # [T, 576]
    scores = Q_fp32 @ K_fp32.T  # [H_q, T]

    # Mask invalid columns to -inf so they contribute zero post-softmax.
    if invalid.any():
        scores = scores.masked_fill(invalid.unsqueeze(0), float("-inf"))

    # ------------------------------------------------------------------
    # 7. Online softmax in base-2 (mirrors kernel arithmetic).
    #    Spec §5.1 + ``splitkv_mla.cuh:32-84 scale_softmax``.
    # ------------------------------------------------------------------
    log2e = math.log2(math.e)
    scaled = scores * (softmax_scale * log2e)  # [H_q, T]
    max_b2 = scaled.amax(dim=-1, keepdim=True)  # [H_q, 1]
    # If every column is invalid (-inf), max is -inf; replace to avoid NaN
    # in the subtraction (kernel sets the row to all-zeros in that case).
    all_invalid = torch.isneginf(max_b2.squeeze(-1))
    safe_max = torch.where(
        all_invalid.unsqueeze(-1), torch.zeros_like(max_b2), max_b2
    )
    P = torch.exp2(scaled - safe_max)  # [H_q, T]
    if invalid.any():
        P = P * (~invalid).to(P.dtype).unsqueeze(0)
    sum_P = P.sum(dim=-1)  # [H_q]
    # Guard against sum_P==0 to avoid 0/0 (matches kernel: writes 0).
    sum_P_safe = torch.where(sum_P > 0, sum_P, torch.ones_like(sum_P))
    lse_b2 = torch.log2(sum_P_safe) + safe_max.squeeze(-1)  # [H_q]
    # If a row had no valid lanes, lse is -inf (kernel writes -INF to lse).
    lse_b2 = torch.where(
        all_invalid, torch.full_like(lse_b2, float("-inf")), lse_b2
    )

    # ------------------------------------------------------------------
    # 8. Folded attn_sink (no-split path; spec §6 + splitkv_mla.cuh:305).
    # ------------------------------------------------------------------
    sink_scale = None
    if attn_sink is not None:
        sink_b2 = attn_sink.float() * log2e  # [H_q]
        # new_lse = log2(2^lse + 2^sink); for -inf lse, new_lse = sink.
        new_lse_b2 = torch.where(
            torch.isneginf(lse_b2),
            sink_b2,
            lse_b2 + torch.log2(1 + torch.exp2(sink_b2 - lse_b2)),
        )
        # O scale factor that absorbs the sink fold-in.
        sink_scale = torch.exp2(lse_b2 - new_lse_b2)  # [H_q]
        sink_scale = torch.where(
            torch.isneginf(lse_b2), torch.zeros_like(sink_scale), sink_scale
        )
        lse_b2 = new_lse_b2

    # ------------------------------------------------------------------
    # 9. P_norm @ V (V = K_nope_bf16, the first 512 dims).
    #    Spec §6 step 7 + ``components/config.h:123-131`` (PV MMA).
    # ------------------------------------------------------------------
    P_norm = P / sum_P_safe.unsqueeze(-1)  # [H_q, T]
    out = P_norm @ K_nope_bf16.float()  # [H_q, 512]

    if sink_scale is not None:
        out = out * sink_scale.unsqueeze(-1)

    # Convert lse back to natural-log domain for the public contract.
    lse_natural = lse_b2 / log2e  # [H_q]
    return out.to(torch.bfloat16), lse_natural.to(torch.float32)


def sparse_decode_fwd_reference(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    indices: torch.Tensor,
    softmax_scale: float,
    attn_sink: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Slow PyTorch eager reference for ``sparse_decode_fp8_triton`` (V32).

    Parameters
    ----------
    q
        ``bfloat16 [b, s_q, h_q, d_qk=576]`` query tensor.
    k_cache
        ``uint8`` (or ``int8`` / ``float8_e4m3fn``) ``[num_blocks,
        page_block_size, 1, 656]`` packed paged KV cache.
    indices
        ``int32 [b, s_q, topk]`` flat-index lookup; ``-1`` = invalid.
    softmax_scale
        Pre-softmax scaling factor (typically ``1/sqrt(d_qk)``).
    attn_sink
        Optional ``float32 [h_q]`` per-head log-domain sink.

    Returns
    -------
    out
        ``bfloat16 [b, s_q, h_q, d_v=512]``
    lse
        ``float32 [b, h_q, s_q]``  (note: last two axes transposed vs
        the natural ``[b, s_q, h_q]`` layout, matching the FlashMLA
        Python wrapper's output at
        ``flash_mla_interface.py:166-173``).
    """
    assert q.dim() == 4 and q.dtype == torch.bfloat16
    assert k_cache.dim() == 4
    assert indices.dim() == 3 and indices.dtype == torch.int32
    b, s_q, h_q, d_qk = q.shape
    assert d_qk == _V32_D_QK, f"reference is V32-only: d_qk={d_qk}"
    assert k_cache.shape[2] == 1, "h_kv must be 1 (MQA)"
    assert k_cache.shape[3] == _V32_BYTES_PER_TOKEN, (
        f"k_cache bytes_per_token={k_cache.shape[3]} != {_V32_BYTES_PER_TOKEN}"
    )
    page_block_size = k_cache.shape[1]
    if attn_sink is not None:
        assert attn_sink.shape == (h_q,) and attn_sink.dtype == torch.float32

    out = torch.zeros((b, s_q, h_q, _V32_D_V), dtype=torch.bfloat16, device=q.device)
    # The public lse layout is [b, h_q, s_q] — see spec §5.
    lse = torch.zeros((b, h_q, s_q), dtype=torch.float32, device=q.device)

    for bi in range(b):
        for si in range(s_q):
            out_bh, lse_bh = _decode_one_position(
                Q_bh=q[bi, si],
                indices_bh=indices[bi, si],
                k_cache=k_cache,
                page_block_size=page_block_size,
                softmax_scale=softmax_scale,
                attn_sink=attn_sink,
            )
            out[bi, si] = out_bh
            lse[bi, :, si] = lse_bh

    return out, lse


def make_synthetic_inputs(
    *,
    b: int,
    s_q: int,
    h_q: int,
    topk: int,
    num_blocks: int,
    page_block_size: int = 64,
    seed: int = 0,
    device: str = "cpu",
    valid_ratio: float = 1.0,
) -> dict:
    """Build a self-consistent synthetic V32 sparse-decode call kit.

    Used by the numerical tests + Compute-Sanitizer tests. Uses ``seed``
    for reproducibility; values are scaled so FP8 quantisation noise is
    representative of real V4-Flash workloads.

    Returns a dict whose keys map onto the ``sparse_decode_fp8_triton``
    keyword arguments.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    # Q: bf16 with mild magnitude so ``Q·K`` stays in the FP32 mantissa
    # range and softmax doesn't saturate.
    q = torch.randn((b, s_q, h_q, _V32_D_QK), generator=g, dtype=torch.float32)
    q = q.to(torch.bfloat16)

    # KV cache: synthesise NoPE (FP8), per-128 scales (FP32), RoPE (BF16)
    # in fp32 first, then re-pack into the 656-byte uint8 layout.
    nope_fp32 = torch.randn(
        (num_blocks, page_block_size, _V32_D_NOPE), generator=g, dtype=torch.float32
    )
    # Per-tile FP32 scale; clamped to a reasonable range so dequant
    # doesn't blow up.
    scales_fp32 = (
        torch.rand(
            (num_blocks, page_block_size, _V32_NUM_SCALES),
            generator=g,
            dtype=torch.float32,
        )
        * 0.5
        + 0.1
    )
    rope_fp32 = torch.randn(
        (num_blocks, page_block_size, _V32_D_ROPE), generator=g, dtype=torch.float32
    )

    # Quantise NoPE to FP8 e4m3.
    nope_dequant_factor = scales_fp32.unsqueeze(-1)  # [B, P, 4, 1]
    nope_per_tile = nope_fp32.reshape(
        num_blocks, page_block_size, _V32_NUM_SCALES, 128
    )
    # Inverse-scale before quantising so dequant recovers the value.
    nope_pre_q = nope_per_tile / nope_dequant_factor.clamp(min=1e-6)
    nope_fp8 = nope_pre_q.to(torch.float8_e4m3fn)

    rope_bf16 = rope_fp32.to(torch.bfloat16)

    kv = torch.zeros(
        (num_blocks, page_block_size, 1, _V32_BYTES_PER_TOKEN), dtype=torch.uint8
    )
    nope_bytes = (
        nope_fp8.reshape(num_blocks, page_block_size, _V32_D_NOPE)
        .contiguous()
        .view(torch.uint8)
    )
    scales_bytes = (
        scales_fp32.reshape(num_blocks, page_block_size, _V32_NUM_SCALES)
        .contiguous()
        .view(torch.uint8)
    )
    rope_bytes = (
        rope_bf16.reshape(num_blocks, page_block_size, _V32_D_ROPE)
        .contiguous()
        .view(torch.uint8)
    )
    kv[:, :, 0, _OFF_NOPE:_OFF_SCALES] = nope_bytes
    kv[:, :, 0, _OFF_SCALES:_OFF_ROPE] = scales_bytes
    kv[:, :, 0, _OFF_ROPE:_V32_BYTES_PER_TOKEN] = rope_bytes

    # Indices: pick valid_ratio fraction valid token positions, rest = -1.
    max_flat = num_blocks * page_block_size
    indices = torch.randint(
        0, max_flat, (b, s_q, topk), generator=g, dtype=torch.int32
    )
    if valid_ratio < 1.0:
        mask = torch.rand((b, s_q, topk), generator=g) > valid_ratio
        indices[mask] = -1

    softmax_scale = 1.0 / math.sqrt(_V32_D_QK)

    if device != "cpu":
        q = q.to(device)
        kv = kv.to(device)
        indices = indices.to(device)

    return {
        "q": q,
        "k_cache": kv,
        "block_table": None,
        "cache_seqlens": None,
        "head_dim_v": _V32_D_V,
        "tile_scheduler_metadata": None,
        "num_splits": None,
        "softmax_scale": softmax_scale,
        "causal": False,
        "is_fp8_kvcache": True,
        "indices": indices,
        "attn_sink": None,
        "extra_k_cache": None,
        "extra_indices_in_kvcache": None,
        "topk_length": None,
        "extra_topk_length": None,
    }


__all__ = [
    "sparse_decode_fwd_reference",
    "make_synthetic_inputs",
]
