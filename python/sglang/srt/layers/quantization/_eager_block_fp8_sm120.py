"""Eager BF16 reference for sm_120 W8A8 block FP8 linear (Phase 5 / T5.1).

Drop-in replacement for ``triton_w8a8_block_fp8_linear`` (the
T4.3-dispatched-to backend on sm_120) for diagnostic A/B testing. Gated
by ``SGLANG_SM120_EAGER_W8A8_BLOCK`` (see ``sglang.srt.layers.sm120_diagnostic``).

Algorithm
---------

Block-FP8 linear is ``out = input @ weight^T + bias`` where ``weight`` is
stored in FP8 e4m3fn with per-block (``[block_n, block_k]``) FP32 scales
in ``weight_scale``. The standard production path quantizes ``input`` to
FP8 with per-token-group scales, then runs an FP8 GEMM. The eager
reference instead:

1. Block-dequantizes ``weight`` to BF16 in full (``[N, K]`` BF16 buffer).
2. Casts ``input`` to BF16 (it already is in production V4-Flash).
3. Runs a standard BF16 ``@ weight^T`` GEMM.
4. Adds bias and returns.

This is dramatically slower than the Triton kernel (BF16 matmul over a
weight whose K is e.g. 5120 + an ephemeral dequant tensor) but is the
correct mathematical reference for "what the W8A8 block path *should*
compute given these FP8 weights and these BF16 inputs". If GSM8K under
this toggle moves the needle relative to the Triton-W8A8 baseline, the
Triton kernel has a per-call numerical drift on sm_120 (Phase-5 §5
suspect S2).

The dequantization step matches the producer convention used in
``test_sm120_fp8_block_w8a8.py`` (block-quant tile-by-tile, scale_pow2
per block).
"""

from __future__ import annotations

from typing import List, Optional

import torch


def _block_dequantize_weight_to_bf16(
    weight_fp8: torch.Tensor,
    weight_scale: torch.Tensor,
    block_size: List[int],
) -> torch.Tensor:
    """Block-dequantize an FP8 e4m3fn weight + per-block FP32 scales -> BF16.

    Layout convention (matches sglang's
    ``per_token_group_quant_fp8`` / ``w8a8_block_fp8_matmul`` producer):

      ``weight_fp8`` : ``[N, K]`` FP8 e4m3fn.
      ``weight_scale`` : ``[ceil(N / block_n), ceil(K / block_k)]`` FP32,
      where ``block_size = [block_n, block_k]``. Tile (i, j) in the
      weight is multiplied by ``weight_scale[i, j]``.
    """
    assert weight_fp8.dim() == 2, f"weight must be 2D, got {weight_fp8.shape}"
    assert weight_scale.dim() == 2, f"scale must be 2D, got {weight_scale.shape}"
    block_n, block_k = block_size
    N, K = weight_fp8.shape
    n_blk_n = (N + block_n - 1) // block_n
    n_blk_k = (K + block_k - 1) // block_k
    assert weight_scale.shape == (n_blk_n, n_blk_k), (
        f"weight_scale shape {weight_scale.shape} must equal "
        f"({n_blk_n}, {n_blk_k}) for weight {weight_fp8.shape} block "
        f"{block_size}"
    )

    w_f32 = weight_fp8.float()
    scale_per_n = weight_scale.repeat_interleave(block_n, dim=0)[:N]
    scale_per_nk = scale_per_n.repeat_interleave(block_k, dim=1)[:, :K]
    return (w_f32 * scale_per_nk.to(w_f32.dtype)).to(torch.bfloat16)


def eager_w8a8_block_fp8_linear_sm120(
    input: torch.Tensor,
    weight: torch.Tensor,
    block_size: List[int],
    weight_scale: torch.Tensor,
    input_scale: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """BF16 reference for ``triton_w8a8_block_fp8_linear``."""
    assert input_scale is None, (
        "eager_w8a8_block_fp8_linear_sm120 expects input_scale=None "
        "(matches production V4-Flash; input is BF16 and gets quantized "
        "internally by the Triton kernel)."
    )
    input_2d = input.view(-1, input.shape[-1])
    output_shape = [*input.shape[:-1], weight.shape[0]]

    w_bf16 = _block_dequantize_weight_to_bf16(weight, weight_scale, block_size)
    out = (input_2d.to(torch.bfloat16) @ w_bf16.t()).to(input_2d.dtype)
    if bias is not None:
        out = out + bias
    return out.view(*output_shape)


__all__ = [
    "eager_w8a8_block_fp8_linear_sm120",
    "_block_dequantize_weight_to_bf16",
]
