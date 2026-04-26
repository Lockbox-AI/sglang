"""Pure-PyTorch eager reference for FlashMLA's ``sparse_decode_fwd`` (V4-Flash).

**This file is the test oracle, not a production kernel.** It computes
the same attention output as
:func:`sglang.srt.layers.attention.sm120_fallback.tilelang_sparse_decode.tilelang_fp8_sparse_decode`
(V2 / K2a) using only ``torch.bmm`` / ``torch.softmax`` / elementwise math —
no Triton, no TileLang, no TMA, no FP8 hardware. Numerical correctness
is the only objective; the implementation is intentionally slow and
per-batch-row.

**Updated 2026-04-26 for the V4-Flash runtime layout** (584 B/token,
448 NoPE FP8 + UE8M0 scales + 64 BF16 RoPE; per-page two-region split).
The previous V32-FlashMLA-spec reference (656 B/token, 512 NoPE +
FP32 scales) is preserved on the original kernel branch
``feat/sm120-tilelang-sparse-decode-kernel``.

Algorithm follows ``T1.3-sparse-decode-fwd-spec.md §6`` (Run 4
revalidated). The math is structurally identical to FlashMLA's
``splitkv_mla.cuh`` (SHA ``71c73792``); only the byte arithmetic, the
per-tile scale dequant (UE8M0 not FP32), and the trailing 64-dim
zero-pad on the output differ.

Numerical noise floor expected when comparing reference vs. kernel:

- FP8 ``e4m3`` mantissa is 3 bits → ~12.5% relative quantization error
  per element worst case, but per-64 BF16 scaling brings the effective
  per-tile error well below ~0.5% RMS.
- BF16 reduction order is non-associative.
- Final ``rtol=1e-3, atol=1e-3`` is the project-wide FP8 attention gate.

The reference does **not** implement split-KV — it is the merged /
no-split path. The kernel's split-KV partitioning + combine is purely
a perf optimization; the merged math is what callers see.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch

# V4-Flash layout constants — duplicated from the kernel module.
_V4_D_NOPE = 448
_V4_D_ROPE = 64
_V4_D_QK = _V4_D_NOPE + _V4_D_ROPE  # 512 (sglang absorbed-latent)
_V4_D_V = 512  # head_dim_v from V4-Flash model config
_V4_QUANT_TILE_SIZE = 64
_V4_NUM_SCALES = _V4_D_NOPE // _V4_QUANT_TILE_SIZE  # 7
_V4_SCALE_PAD = 1
_V4_PADDED_SCALE_BYTES_PER_TOKEN = _V4_NUM_SCALES + _V4_SCALE_PAD  # 8
_V4_NOPE_ROPE_BYTES_PER_TOKEN = _V4_D_NOPE + _V4_D_ROPE * 2  # 576
_V4_BYTES_PER_TOKEN = _V4_NOPE_ROPE_BYTES_PER_TOKEN + _V4_PADDED_SCALE_BYTES_PER_TOKEN  # 584
_UE8M0_BIAS = 127.0


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _bytes_per_page_padded(page_size: int) -> int:
    raw = page_size * _V4_BYTES_PER_TOKEN
    return _ceil_div(raw, 576) * 576


def _decode_one_position(
    Q_bh: torch.Tensor,
    indices_bh: torch.Tensor,
    k_cache_4d: torch.Tensor,
    page_size: int,
    softmax_scale: float,
    attn_sink: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute attention for a single ``(batch, s_q)`` decode position.

    Parameters
    ----------
    Q_bh : ``bfloat16 [H_q, D_qk=512]``
    indices_bh : ``int32 [topk]`` flat token indices (``-1`` = invalid)
    k_cache_4d : ``uint8 [num_pages, page_size, 1, 584]`` V4 view
    page_size : KV cache page block size (256 for SWA, 64 for sparse decode)
    softmax_scale : pre-softmax scaling (``1/sqrt(d_qk)``)
    attn_sink : optional ``float32 [H_q]`` per-head log-domain sink

    Returns
    -------
    out_bh : ``bfloat16 [H_q, D_v=512]`` (``out_bh[..., 448:512] = 0``)
    lse_bh : ``float32 [H_q]`` natural-log lse
    """
    H_q, D_qk = Q_bh.shape
    assert D_qk == _V4_D_QK, f"reference expects D_qk={_V4_D_QK}, got {D_qk}"
    T = indices_bh.shape[0]

    # Recover (page, off_in_page) from flat indices.
    flat_idx = indices_bh.to(torch.int64)
    invalid = flat_idx == -1
    safe_idx = flat_idx.clamp(min=0)
    page_idx = safe_idx // page_size
    off_in_page = safe_idx % page_size

    num_pages = k_cache_4d.shape[0]
    page_idx = page_idx.clamp(max=num_pages - 1)

    # Re-construct the underlying 2-D buffer view from the 4-D view.
    bppp = k_cache_4d.stride(0) * k_cache_4d.element_size()
    underlying = k_cache_4d.as_strided(
        size=(num_pages, bppp),
        stride=(k_cache_4d.stride(0), 1),
    )
    # Make sure we operate on uint8 raw bytes regardless of the source
    # dtype (uint8 / int8 / fp8_e4m3fn — all 1 B/elem).
    if underlying.dtype != torch.uint8:
        underlying = underlying.view(torch.uint8)

    # Region A: [num_pages, page_size, 576] — NoPE (448 B) + RoPE (128 B) per token
    region_a_bytes = page_size * _V4_NOPE_ROPE_BYTES_PER_TOKEN
    region_a = underlying[:, :region_a_bytes].view(
        num_pages, page_size, _V4_NOPE_ROPE_BYTES_PER_TOKEN
    )
    # Region B: [num_pages, page_size, 8] — 7 UE8M0 scales + 1 pad per token
    region_b_bytes = page_size * _V4_PADDED_SCALE_BYTES_PER_TOKEN
    region_b = underlying[
        :, region_a_bytes : region_a_bytes + region_b_bytes
    ].view(num_pages, page_size, _V4_PADDED_SCALE_BYTES_PER_TOKEN)

    # Gather T tokens.
    nope_bytes = region_a[page_idx, off_in_page, :_V4_D_NOPE].contiguous()
    rope_bytes = (
        region_a[page_idx, off_in_page, _V4_D_NOPE:_V4_NOPE_ROPE_BYTES_PER_TOKEN]
        .contiguous()
    )
    scale_bytes = region_b[page_idx, off_in_page, :_V4_NUM_SCALES].contiguous()

    K_nope_fp8 = nope_bytes.view(torch.float8_e4m3fn)  # [T, 448]
    K_rope = rope_bytes.view(torch.bfloat16).reshape(T, _V4_D_ROPE)  # [T, 64]
    scale_uint8 = scale_bytes.view(torch.uint8)  # [T, 7]

    # UE8M0 dequant: scale_fp32 = 2 ** (uint8 - 127) per 64-elt tile.
    scale_fp32 = torch.exp2(scale_uint8.to(torch.float32) - _UE8M0_BIAS)  # [T, 7]
    K_nope_fp32 = K_nope_fp8.to(torch.float32)  # [T, 448]
    K_nope_dequant = (
        K_nope_fp32.reshape(T, _V4_NUM_SCALES, _V4_QUANT_TILE_SIZE)
        * scale_fp32.unsqueeze(-1)
    )
    K_nope_bf16 = K_nope_dequant.reshape(T, _V4_D_NOPE).to(torch.bfloat16)

    # Concat NoPE + RoPE -> full K [T, 512].
    K = torch.cat([K_nope_bf16, K_rope], dim=-1)  # [T, 512] bf16

    # Zero invalid lanes so they contribute nothing to S.
    if invalid.any():
        K = K.clone()
        K[invalid] = 0
        K_nope_bf16 = K_nope_bf16.clone()
        K_nope_bf16[invalid] = 0

    # Q · K^T in fp32 (kernel uses bf16 mma → fp32 accum).
    Q_fp32 = Q_bh.float()  # [H_q, 512]
    K_fp32 = K.float()  # [T, 512]
    scores = Q_fp32 @ K_fp32.T  # [H_q, T]

    # Mask invalid columns to -inf so they contribute zero post-softmax.
    if invalid.any():
        scores = scores.masked_fill(invalid.unsqueeze(0), float("-inf"))

    # Online softmax in base-2 (mirrors kernel arithmetic).
    log2e = math.log2(math.e)
    scaled = scores * (softmax_scale * log2e)  # [H_q, T]
    max_b2 = scaled.amax(dim=-1, keepdim=True)
    all_invalid = torch.isneginf(max_b2.squeeze(-1))
    safe_max = torch.where(
        all_invalid.unsqueeze(-1), torch.zeros_like(max_b2), max_b2
    )
    P = torch.exp2(scaled - safe_max)
    if invalid.any():
        P = P * (~invalid).to(P.dtype).unsqueeze(0)
    sum_P = P.sum(dim=-1)
    sum_P_safe = torch.where(sum_P > 0, sum_P, torch.ones_like(sum_P))
    lse_b2 = torch.log2(sum_P_safe) + safe_max.squeeze(-1)
    lse_b2 = torch.where(
        all_invalid, torch.full_like(lse_b2, float("-inf")), lse_b2
    )

    # Folded attn_sink (no-split path).
    sink_scale = None
    if attn_sink is not None:
        sink_b2 = attn_sink.float() * log2e
        new_lse_b2 = torch.where(
            torch.isneginf(lse_b2),
            sink_b2,
            lse_b2 + torch.log2(1 + torch.exp2(sink_b2 - lse_b2)),
        )
        sink_scale = torch.exp2(lse_b2 - new_lse_b2)
        sink_scale = torch.where(
            torch.isneginf(lse_b2), torch.zeros_like(sink_scale), sink_scale
        )
        lse_b2 = new_lse_b2

    # P_norm @ V (V = K_nope_bf16, the first 448 dims).
    P_norm = P / sum_P_safe.unsqueeze(-1)  # [H_q, T]
    out_nope = P_norm @ K_nope_bf16.float()  # [H_q, 448]

    if sink_scale is not None:
        out_nope = out_nope * sink_scale.unsqueeze(-1)

    # Zero-pad to D_v = 512.
    out = torch.zeros((H_q, _V4_D_V), dtype=torch.float32, device=Q_bh.device)
    out[:, :_V4_D_NOPE] = out_nope

    lse_natural = lse_b2 / log2e
    return out.to(torch.bfloat16), lse_natural.to(torch.float32)


def sparse_decode_fwd_reference(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    indices: torch.Tensor,
    softmax_scale: float,
    attn_sink: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Slow PyTorch eager reference for the V4-Flash sparse decode kernel.

    Parameters
    ----------
    q : ``bfloat16 [b, s_q, h_q, d_qk=512]``
    k_cache : ``[num_pages, page_size, 1, 584]`` V4 layout view (uint8 /
        int8 / float8_e4m3fn — all 1 B/elem).
    indices : ``int32 [b, s_q, topk]`` flat-index lookup; ``-1`` = invalid.
    softmax_scale : pre-softmax scaling factor (``1/sqrt(d_qk)``).
    attn_sink : optional ``float32 [h_q]`` per-head log-domain sink.

    Returns
    -------
    out : ``bfloat16 [b, s_q, h_q, d_v=512]`` with ``out[..., 448:512] = 0``.
    lse : ``float32 [b, h_q, s_q]`` (last two axes transposed to match
        FlashMLA's Python wrapper output order).
    """
    assert q.dim() == 4 and q.dtype == torch.bfloat16
    assert k_cache.dim() == 4
    assert indices.dim() == 3 and indices.dtype == torch.int32
    b, s_q, h_q, d_qk = q.shape
    assert d_qk == _V4_D_QK, f"reference expects d_qk={_V4_D_QK}, got {d_qk}"
    assert k_cache.shape[2] == 1, "h_kv must be 1 (MQA)"
    assert k_cache.shape[3] == _V4_BYTES_PER_TOKEN, (
        f"k_cache bytes_per_token={k_cache.shape[3]} != {_V4_BYTES_PER_TOKEN}"
    )
    page_size = k_cache.shape[1]
    if attn_sink is not None:
        assert attn_sink.shape == (h_q,) and attn_sink.dtype == torch.float32

    out = torch.zeros((b, s_q, h_q, _V4_D_V), dtype=torch.bfloat16, device=q.device)
    lse = torch.zeros((b, h_q, s_q), dtype=torch.float32, device=q.device)

    for bi in range(b):
        for si in range(s_q):
            out_bh, lse_bh = _decode_one_position(
                Q_bh=q[bi, si],
                indices_bh=indices[bi, si],
                k_cache_4d=k_cache,
                page_size=page_size,
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
    """Build a self-consistent synthetic V4-Flash sparse-decode call kit.

    Used by the numerical tests + Compute-Sanitizer tests. Builds the
    underlying 2-D ``[num_pages, bytes_per_page_padded]`` buffer with
    the per-page region split, then exposes it as the 4-D
    ``[num_pages, page_size, 1, 584]`` view sglang's radix backend
    would pass.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    num_pages = num_blocks
    page_size = page_block_size

    # Q: bf16 with mild magnitudes (NoPE) + larger magnitudes (RoPE)
    # so the GEMM exercises both halves.
    q_nope = torch.randn(
        (b, s_q, h_q, _V4_D_NOPE), generator=g, dtype=torch.float32
    ).to(torch.bfloat16) * 0.1
    q_rope = torch.randn(
        (b, s_q, h_q, _V4_D_ROPE), generator=g, dtype=torch.float32
    ).to(torch.bfloat16) * 0.5
    q = torch.cat([q_nope, q_rope], dim=-1).contiguous()
    assert q.shape == (b, s_q, h_q, _V4_D_QK)

    # KV cache: synthesise NoPE FP8 + UE8M0 scales + BF16 RoPE in fp32 first,
    # then re-pack into the V4 layout (region A interleaved per token,
    # region B in per-page tail).
    n_total = num_pages * page_size

    nope_fp32 = torch.randn(
        (n_total, _V4_NUM_SCALES, _V4_QUANT_TILE_SIZE),
        generator=g, dtype=torch.float32,
    )
    # UE8M0 scale: pick uint8 byte values that decode to ~0.05..2.
    scale_uint8 = torch.randint(
        125, 132, (n_total, _V4_NUM_SCALES), generator=g, dtype=torch.uint8
    )
    scale_fp32 = torch.exp2(scale_uint8.to(torch.float32) - _UE8M0_BIAS)
    # Inverse-scale before quantising so dequant recovers the value.
    nope_pre_q = nope_fp32 / scale_fp32.unsqueeze(-1).clamp(min=1e-6)
    nope_fp8 = nope_pre_q.to(torch.float8_e4m3fn).reshape(n_total, _V4_D_NOPE)

    rope_bf16 = torch.randn(
        (n_total, _V4_D_ROPE), generator=g, dtype=torch.float32
    ).to(torch.bfloat16) * 0.05

    bppp = _bytes_per_page_padded(page_size)
    region_a_bytes = page_size * _V4_NOPE_ROPE_BYTES_PER_TOKEN
    region_b_bytes = page_size * _V4_PADDED_SCALE_BYTES_PER_TOKEN

    underlying_2d = torch.zeros(num_pages, bppp, dtype=torch.uint8)

    # Region A: NoPE (FP8 bytes) + RoPE (BF16 bytes) per token, interleaved.
    region_a = underlying_2d[:, :region_a_bytes].view(
        num_pages, page_size, _V4_NOPE_ROPE_BYTES_PER_TOKEN
    )
    nope_uint8 = nope_fp8.contiguous().view(torch.uint8).view(
        num_pages, page_size, _V4_D_NOPE
    )
    rope_uint8 = rope_bf16.contiguous().view(torch.uint8).view(
        num_pages, page_size, _V4_D_ROPE * 2
    )
    region_a[:, :, :_V4_D_NOPE] = nope_uint8
    region_a[:, :, _V4_D_NOPE:_V4_NOPE_ROPE_BYTES_PER_TOKEN] = rope_uint8

    # Region B: 7 UE8M0 scales + 1 pad per token.
    region_b = underlying_2d[
        :, region_a_bytes : region_a_bytes + region_b_bytes
    ].view(num_pages, page_size, _V4_PADDED_SCALE_BYTES_PER_TOKEN)
    region_b[:, :, :_V4_NUM_SCALES] = scale_uint8.view(
        num_pages, page_size, _V4_NUM_SCALES
    )
    # region_b[:, :, _V4_NUM_SCALES:] left at 0 (pad).

    # Build the 4-D view sglang would pass.
    kv = underlying_2d[:, : page_size * _V4_BYTES_PER_TOKEN].view(
        num_pages, page_size, 1, _V4_BYTES_PER_TOKEN
    )

    # Indices: pick valid_ratio fraction valid token positions, rest = -1.
    max_flat = num_pages * page_size
    indices = torch.randint(
        0, max_flat, (b, s_q, topk), generator=g, dtype=torch.int32
    )
    if valid_ratio < 1.0:
        mask = torch.rand((b, s_q, topk), generator=g) > valid_ratio
        indices[mask] = -1

    softmax_scale = 1.0 / math.sqrt(_V4_D_QK)

    if device != "cpu":
        q = q.to(device)
        kv = kv.to(device)
        indices = indices.to(device)

    return {
        "q": q,
        "k_cache": kv,
        "block_table": None,
        "cache_seqlens": None,
        "head_dim_v": _V4_D_V,
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
