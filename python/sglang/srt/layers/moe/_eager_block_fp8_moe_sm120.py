"""Eager BF16 reference for sm_120 fused-MoE block-FP8 W8A8 (Phase 5 / T5.1).

Drop-in replacement for ``fused_moe(use_fp8_w8a8=True, block_shape=...)``
on sm_120 for diagnostic A/B testing. Gated by ``SGLANG_SM120_EAGER_MOE_FP8``
(see ``sglang.srt.layers.sm120_diagnostic``).

Algorithm (mirrors the eager reference in
``test_sm120_fp8_block_moe.py::_eager_block_fp8_moe_reference`` but extended
to the full V4-Flash MoE topology including the down-projection):

1. Block-dequantize ``w1`` ``[E, 2N, K]`` and ``w2`` ``[E, K, N]`` from
   FP8 + per-block FP32 scales to BF16 once at call time. Ephemeral
   dequantized buffers; freed when this function returns.
2. Per-token, top-k expert dispatch: each token sends activations to
   its top-k experts (typical V4-Flash: top-k = 8 of 256 experts).
3. Per expert ``e``:
     a. ``inter = a @ w1_bf16[e].t()``  -> ``[m_e, 2N]``
     b. ``act = SiluAndMul(inter)``      -> ``[m_e, N]`` (gate * up)
     c. ``out_e = act @ w2_bf16[e].t()`` -> ``[m_e, K]``
4. Scatter-and-weighted-sum back to ``[B, K]`` per the topk weights.

This is **the same math** as the production path; it just bypasses every
FP8-specific kernel. If GSM8K under this toggle clears the post-T4.3
0.010 baseline, the residual gap is dominated by the Triton MoE FP8 kernel
(Phase-5 §5 suspect S3); if not, S3 is not the dominant residual.

Caveats
-------

- Dequantizing 256 experts with N=K=5120 in BF16 costs
  ``256 * 2 * 5120 * 5120 * 2 + 256 * 5120 * 5120 * 2 = ~40 GB``
  if done in one shot. We keep a per-expert dequant scope (compute one
  expert's BF16 weights at a time, run, free) to bound peak memory at a
  few GB per TP shard.
- The eager reference is ~50-100x slower than the Triton fused-MoE; we
  accept this for diagnostic correctness.
"""

from __future__ import annotations

from typing import List, Optional

import torch

from sglang.srt.layers.quantization._eager_block_fp8_sm120 import (
    _block_dequantize_weight_to_bf16,
)


def _silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    """SwiGLU activation: ``silu(gate) * up`` where x is ``[..., 2N]``
    interleaved as ``[gate || up]`` along the last dim.
    """
    gate, up = x.chunk(2, dim=-1)
    return torch.nn.functional.silu(gate) * up


def eager_block_fp8_moe_sm120(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    block_shape: List[int],
    *,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    b1: Optional[torch.Tensor] = None,
    b2: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Eager per-expert BF16 MoE with on-the-fly block-FP8 dequant.

    Parameters
    ----------
    hidden_states
        ``[B, K]`` BF16 input activations (B = total prefill+decode tokens
        on this TP shard).
    w1
        ``[E, 2N, K]`` FP8 e4m3fn weight (gate || up concatenated along
        rows).
    w2
        ``[E, K, N]`` FP8 e4m3fn weight (down projection).
    topk_weights
        ``[B, top_k]`` FP32 routing probabilities (already softmax-normed).
    topk_ids
        ``[B, top_k]`` int64 expert indices.
    w1_scale, w2_scale
        Per-block FP32 scales for w1 / w2 (shape ``[E, ceil(2N / block_n),
        ceil(K / block_k)]`` and ``[E, ceil(K / block_n), ceil(N / block_k)]``
        respectively).
    block_shape
        ``[block_n, block_k]``.
    a1_scale, a2_scale
        Accepted for API parity, ignored (input quantization is bypassed).
    b1, b2
        Per-expert biases (currently expected to be None for V4-Flash MoE).

    Returns
    -------
    out
        ``[B, K]`` BF16 output (same dtype as ``hidden_states``).
    """
    assert (
        hidden_states.dim() == 2
    ), f"hidden_states must be 2-D [B, K], got {tuple(hidden_states.shape)}"
    assert w1.dim() == 3, f"w1 must be 3-D [E, 2N, K], got {tuple(w1.shape)}"
    assert w2.dim() == 3, f"w2 must be 3-D [E, K, N], got {tuple(w2.shape)}"
    E, two_n, K = w1.shape
    assert hidden_states.shape[1] == K
    assert two_n % 2 == 0
    N = two_n // 2
    assert w2.shape == (
        E,
        K,
        N,
    ), f"w2 shape {tuple(w2.shape)} must equal [{E}, {K}, {N}]"
    assert topk_weights.dim() == 2 and topk_weights.shape[0] == hidden_states.shape[0]
    assert topk_ids.shape == topk_weights.shape
    assert b1 is None and b2 is None, (
        "eager_block_fp8_moe_sm120: per-expert bias not currently supported "
        "(V4-Flash MoE has no MoE bias)"
    )
    _ = a1_scale
    _ = a2_scale

    B, _ = hidden_states.shape
    top_k = topk_weights.shape[1]

    out = torch.zeros(B, K, dtype=hidden_states.dtype, device=hidden_states.device)

    a_bf16 = hidden_states.to(torch.bfloat16)

    # Per-expert eager loop. Memory-bounded: only one expert's BF16
    # weights live in HBM at a time.
    for e in range(E):
        token_mask = topk_ids == e  # [B, top_k]
        if not token_mask.any():
            continue
        rows, slot = torch.where(token_mask)  # both [n_e]
        a_e = a_bf16[rows]  # [n_e, K]
        weight_e = topk_weights[rows, slot].to(torch.bfloat16)  # [n_e]

        w1_e_bf16 = _block_dequantize_weight_to_bf16(
            w1[e], w1_scale[e], block_shape
        )  # [2N, K]
        inter = a_e @ w1_e_bf16.t()  # [n_e, 2N]
        del w1_e_bf16
        act = _silu_and_mul(inter)  # [n_e, N]
        del inter

        w2_e_bf16 = _block_dequantize_weight_to_bf16(
            w2[e], w2_scale[e], block_shape
        )  # [K, N]
        contrib = act @ w2_e_bf16.t()  # [n_e, K]
        del w2_e_bf16, act

        contrib = contrib * weight_e.unsqueeze(-1)
        out.index_add_(0, rows, contrib.to(out.dtype))

    return out


__all__ = ["eager_block_fp8_moe_sm120"]
