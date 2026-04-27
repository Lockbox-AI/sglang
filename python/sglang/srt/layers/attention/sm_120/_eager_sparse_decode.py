"""Eager BF16 reference for sm_120 FP8 sparse-decode (Phase 5 / T5.1 diagnostic).

This module is the **correctness oracle** for the K2a TileLang
``tilelang_fp8_sparse_decode`` kernel and (transitively) for the post-hoc
``attn_sink`` fold in ``triton_combine.py``. It is gated behind the
``SGLANG_SM120_EAGER_SPARSE_DECODE`` env-var toggle (see
``sglang.srt.layers.sm120_diagnostic``) and is intended **only** for the
Phase-5 diagnostic GSM8K A/B run; it is significantly slower than the
TileLang kernel and is never used in production.

Algorithm
---------

For each ``(b, s_q)`` slot, with ``H_q`` query heads, ``D_NOPE=448``,
``D_ROPE=64``, ``D_qk=512``, ``topk`` sparse indices into a paged 584-byte
KV cache:

1. Decode the V4-Flash KV layout the same way ``tilelang_fp8_sparse_decode``
   does (``_underlying_2d_buffer`` + region-A/region-B split with
   576-byte per-token NoPE+RoPE stride and 8-byte per-token UE8M0 scale
   stride at the per-page tail), then dequantize NoPE FP8 -> BF16 with the
   per-64-element UE8M0 scale (``2 ** (uint8 - 127)``).
2. Gather ``[topk, D_NOPE]`` and ``[topk, D_ROPE]`` BF16 KV tiles via the
   sparse indices. Invalid indices (-1) are clamped to 0 and masked to
   ``-inf`` in QK before softmax.
3. ``QK = Q[..., :448] @ K_nope^T + Q[..., 448:512] @ K_rope^T`` (BF16
   GEMM, FP32 accum).
4. Apply ``attn_sink`` IN-KERNEL via the standard "extra zero-value token
   with logit ``attn_sink[h]``" formulation: the row of QK is augmented
   with one extra column whose logit is ``attn_sink[h]`` and whose value
   contribution is zero. Softmax is then taken over (topk + 1) positions.
   This is FlashMLA's combine.cu:101-112 semantics in mathematically
   exact form, applied AT THE PER-(b, s_q) ROW LEVEL rather than as a
   post-hoc combine fold.
5. ``Out[..., :448] = softmax(QK) @ K_nope`` (only the topk-position
   slice; the sink contributes zero by construction). ``Out[..., 448:]``
   is zero-padded (matches K2a kernel contract).
6. ``LSE`` is computed in natural-log space, including the sink term.

The output contract matches ``flash_mla.flash_mla_with_kvcache``:
``out`` is ``bfloat16 [b, s_q, h_q, d_v=512]``, ``lse`` is
``float32 [b, h_q, s_q]`` (note the (h_q, s_q) transpose vs. the K2a
kernel's [b, s_q, h_q] return — the K2a wrapper post-transposes; the
caller in ``sm_120/__init__.py`` (``tilelang_fp8_sparse_decode_sm120``)
already routes through the combine which does the transpose). For the
eager-replacement path we mimic the K2a kernel's pre-combine return
shape (``[b, s_q, h_q, d_v]`` BF16 output, ``[b, s_q, h_q]`` FP32 LSE)
so the caller's combine code (which we want to test indirectly) runs
verbatim. When the eager toggle is on, attn_sink is folded HERE and we
must pass ``attn_sink=None`` to the combine to avoid double-folding.

This module reuses the layout-extraction logic already proven correct
by the T2A.4 internal pass (region-A/region-B split, UE8M0 dequant,
RoPE half plumbing).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

# Re-import the V4-Flash byte-layout constants and the underlying-2D helper
# from the K2a kernel module so we stay in lockstep with its layout
# decoding. See `sm_120/tilelang_sparse_decode.py` module docstring for
# layout source-of-truth notes.
from sglang.srt.layers.attention.sm_120.tilelang_sparse_decode import (
    _UE8M0_BIAS,
    _V4_BYTES_PER_TOKEN,
    _V4_D_NOPE,
    _V4_D_QK,
    _V4_D_ROPE,
    _V4_D_V,
    _V4_NOPE_ROPE_BYTES_PER_TOKEN,
    _V4_NUM_SCALES,
    _V4_PADDED_SCALE_BYTES_PER_TOKEN,
    _V4_QUANT_TILE_SIZE,
    _underlying_2d_buffer,
)


def _decode_kv_v4(
    kv_cache: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode the V4-Flash 4-D ``[num_pages, page_size, 1, 584]`` cache view
    into ``(kv_fp8, kv_rope_bf16, kv_scales_u8)`` 3-D tensors of shape
    ``[num_pages, page_size, D_NOPE | D_ROPE | NUM_SCALES]``.

    Mirrors ``tilelang_sparse_decode.tilelang_fp8_sparse_decode`` Steps 1-3
    in its body. Kept as a sibling helper rather than duck-imported to make
    the eager reference's layout decoding self-evident in code review.
    """
    assert kv_cache.dim() == 4
    num_pages, page_size, h_kv, bpt = kv_cache.shape
    assert h_kv == 1, f"h_kv must be 1, got {h_kv}"
    assert (
        bpt == _V4_BYTES_PER_TOKEN
    ), f"per-token logical record must be {_V4_BYTES_PER_TOKEN} B, got {bpt}"

    underlying_2d = _underlying_2d_buffer(kv_cache)  # [num_pages, bppp]

    region_a_bytes = page_size * _V4_NOPE_ROPE_BYTES_PER_TOKEN
    region_b_bytes = page_size * _V4_PADDED_SCALE_BYTES_PER_TOKEN
    bppp = underlying_2d.shape[1]
    assert bppp >= region_a_bytes + region_b_bytes

    region_a = underlying_2d[:, :region_a_bytes].view(
        num_pages, page_size, _V4_NOPE_ROPE_BYTES_PER_TOKEN
    )
    kv_fp8 = region_a[:, :, :_V4_D_NOPE].view(torch.float8_e4m3fn)
    rope_slice = region_a[:, :, _V4_D_NOPE:_V4_NOPE_ROPE_BYTES_PER_TOKEN]
    try:
        kv_rope = rope_slice.view(torch.bfloat16)
    except RuntimeError:
        kv_rope = rope_slice.contiguous().view(torch.bfloat16)

    region_b = underlying_2d[:, region_a_bytes : region_a_bytes + region_b_bytes].view(
        num_pages, page_size, _V4_PADDED_SCALE_BYTES_PER_TOKEN
    )
    if region_b.dtype != torch.uint8:
        region_b = region_b.view(torch.uint8)
    kv_scales = region_b[:, :, :_V4_NUM_SCALES]

    return kv_fp8, kv_rope, kv_scales


def _dequant_nope_to_bf16(
    kv_fp8: torch.Tensor, kv_scales: torch.Tensor
) -> torch.Tensor:
    """Dequantize FP8 NoPE tensor + per-64-element UE8M0 scales -> BF16.

    Per ``index_buf_accessor_v4.py`` and the K2a kernel:
    ``nope_bf16[..., d] = nope_fp8[..., d] * 2 ** (scale_u8[d // 64] - 127)``.
    """
    num_pages, page_size, d_nope = kv_fp8.shape
    assert d_nope == _V4_D_NOPE
    assert kv_scales.shape == (num_pages, page_size, _V4_NUM_SCALES)
    nope_f32 = kv_fp8.float()  # [num_pages, page_size, 448]
    scales_f32 = torch.exp2(
        kv_scales.float() - _UE8M0_BIAS
    )  # [num_pages, page_size, 7]
    scales_expanded = scales_f32.repeat_interleave(_V4_QUANT_TILE_SIZE, dim=-1)
    return (nope_f32 * scales_expanded).to(torch.bfloat16)


def eager_fp8_sparse_decode(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: Optional[torch.Tensor],
    seq_lens: Optional[torch.Tensor],
    indices: torch.Tensor,
    sm_scale: float,
    attn_sink: Optional[torch.Tensor] = None,
    *,
    is_fp8_kvcache: bool = True,
    h_kv: int = 1,
    d_v: int = 512,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Eager BF16 reference for ``tilelang_fp8_sparse_decode``.

    Drop-in signature replacement, same return contract:
    ``out`` is ``bfloat16 [B, S_q, H_q, d_v]`` and
    ``lse`` is ``float32 [B, S_q, H_q]`` (natural-log).

    When ``attn_sink`` is supplied, it is folded IN-KERNEL via the
    "extra zero-value token with logit ``attn_sink[h]``" formulation
    (mathematically exact form of FlashMLA combine.cu:101-112). When
    None, the sink contributes nothing.
    """
    assert q.dim() == 4 and q.dtype == torch.bfloat16
    B, S_q, H_q, D_qk = q.shape
    assert D_qk == _V4_D_QK, f"D_qk must be {_V4_D_QK}, got {D_qk}"
    assert is_fp8_kvcache is True
    assert h_kv == 1
    assert d_v == _V4_D_V
    assert indices.dim() == 3 and indices.shape[:2] == (B, S_q)
    assert indices.dtype == torch.int32

    _ = block_table
    _ = seq_lens

    num_pages, page_size, _h, _bpt = kv_cache.shape
    topk = indices.shape[-1]

    # Memory-bounded path. Production prefill shapes can have B*S_q up
    # to ~8K with topk=2K, which would be a multi-GB per-rank working
    # set if we materialized the full ``[B, S_q, topk, D]`` slice. We
    # chunk over the (B*S_q) axis with CHUNK=32 so the per-chunk
    # working set is bounded at a few hundred MB even in the worst case.
    kv_fp8, kv_rope, kv_scales = _decode_kv_v4(kv_cache)
    flat_fp8 = kv_fp8.reshape(num_pages * page_size, _V4_D_NOPE)
    flat_rope = kv_rope.reshape(num_pages * page_size, _V4_D_ROPE)
    flat_scales = kv_scales.reshape(num_pages * page_size, _V4_NUM_SCALES)

    safe_idx = indices.clamp(min=0).long().view(B * S_q, topk)
    valid_mask = (indices >= 0).view(B * S_q, topk)

    q_nope_flat = q[..., :_V4_D_NOPE].reshape(B * S_q, H_q, _V4_D_NOPE)
    q_rope_flat = q[..., _V4_D_NOPE:_V4_D_QK].reshape(B * S_q, H_q, _V4_D_ROPE)

    out_flat = torch.zeros(B * S_q, H_q, d_v, dtype=torch.bfloat16, device=q.device)
    lse_flat = torch.zeros(B * S_q, H_q, dtype=torch.float32, device=q.device)

    BS = B * S_q
    CHUNK = 32 if BS > 32 else BS

    for start in range(0, BS, CHUNK):
        end = min(BS, start + CHUNK)
        chunk = end - start
        chunk_idx = safe_idx[start:end].reshape(-1)  # [chunk*topk]
        chunk_mask = valid_mask[start:end]  # [chunk, topk]

        k_fp8_c = flat_fp8.index_select(0, chunk_idx)
        k_rope_c = flat_rope.index_select(0, chunk_idx).view(chunk, topk, _V4_D_ROPE)
        k_scales_c = flat_scales.index_select(0, chunk_idx)

        nope_f32 = k_fp8_c.float()
        scales_f32 = torch.exp2(k_scales_c.float() - _UE8M0_BIAS)
        scales_expanded = scales_f32.repeat_interleave(_V4_QUANT_TILE_SIZE, dim=-1)
        k_nope_bf16 = (nope_f32 * scales_expanded).to(torch.bfloat16)
        del nope_f32, scales_expanded, scales_f32
        k_nope_c = k_nope_bf16.view(chunk, topk, _V4_D_NOPE)

        q_nope_c = q_nope_flat[start:end]
        q_rope_c = q_rope_flat[start:end]

        qk_nope = torch.einsum("bHd,btd->bHt", q_nope_c.float(), k_nope_c.float())
        qk_rope = torch.einsum("bHd,btd->bHt", q_rope_c.float(), k_rope_c.float())
        qk = (qk_nope + qk_rope) * float(sm_scale)
        del qk_nope, qk_rope

        mask = chunk_mask.unsqueeze(1).expand(chunk, H_q, topk)
        qk = qk.masked_fill(~mask, float("-inf"))

        if attn_sink is not None:
            assert attn_sink.shape == (H_q,) and attn_sink.dtype == torch.float32
            sink_logits = attn_sink.view(1, H_q, 1).expand(chunk, H_q, 1).to(qk.dtype)
            qk_aug = torch.cat([qk, sink_logits], dim=-1)
            m = qk_aug.amax(dim=-1, keepdim=True)
            m = torch.where(torch.isinf(m) & (m < 0), torch.zeros_like(m), m)
            e_aug = torch.exp(qk_aug - m)
            sumexp = e_aug.sum(dim=-1, keepdim=True).clamp_min(
                torch.finfo(torch.float32).tiny
            )
            weights = e_aug[..., :topk] / sumexp
            lse_chunk = m.squeeze(-1) + torch.log(sumexp.squeeze(-1))
            del qk_aug, e_aug
        else:
            m = qk.amax(dim=-1, keepdim=True)
            m = torch.where(torch.isinf(m) & (m < 0), torch.zeros_like(m), m)
            e = torch.exp(qk - m)
            sumexp = e.sum(dim=-1, keepdim=True).clamp_min(
                torch.finfo(torch.float32).tiny
            )
            weights = e / sumexp
            lse_chunk = m.squeeze(-1) + torch.log(sumexp.squeeze(-1))
            del e

        out_nope_c = torch.einsum("bHt,btd->bHd", weights, k_nope_c.float())
        del weights, k_nope_c, k_rope_c, k_fp8_c, k_scales_c, k_nope_bf16

        out_flat[start:end, :, :_V4_D_NOPE] = out_nope_c.to(torch.bfloat16)
        lse_flat[start:end] = lse_chunk

    out = out_flat.view(B, S_q, H_q, d_v)
    lse = lse_flat.view(B, S_q, H_q)
    return out, lse


__all__ = ["eager_fp8_sparse_decode"]
