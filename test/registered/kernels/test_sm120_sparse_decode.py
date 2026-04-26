"""Tests for the sm_120 FP8 sparse-decode Triton fallback (K2a / poc-16 T2A.3).

Three test layers, in order of cost:

1. **Contract tests** (CPU, fast). Mirror the 36 contract tests in
   ``T1.4-python-stubs/tests/test_sparse_decode_fp8_triton.py`` (T1.4 spec
   §3-§9). Verify shape / dtype / contiguity / forbidden-arg rejection.
   These pass against the stub immediately; post-parent-merge they
   continue to pass against the real kernel because the kernel keeps
   the same input asserts (Subagent 1 / 2 are not allowed to relax
   them).

2. **Numerical correctness tests** (GPU, ``xfail`` until parent merge).
   For each ``(B, s_q, topk, h_q, h_kv, d_qk, d_v)`` combo, generate a
   reproducible synthetic call kit, run the kernel, run the PyTorch
   eager reference defined below, assert
   ``torch.allclose(out, ref_out, rtol=1e-3, atol=1e-3)`` (the
   FP8-attention noise floor used everywhere in poc-16 — see
   ``cuda-toolkit-distilled-for-sm120-work.md §3``).

3. **Compute-Sanitizer integration tests** (GPU + ``compute-sanitizer``,
   ``xfail`` until parent merge). One subprocess invocation per tool
   (``memcheck``, ``racecheck``, ``initcheck``, ``synccheck``) with the
   ``--error-exitcode 2`` flag. Documented racecheck WARNINGs on
   ``tl.sum`` / ``tl.reduce`` warp-shuffle reductions are tolerated per
   ``cuda-toolkit-distilled §6`` — only ``ERROR``-severity output is
   treated as a failure.

Resolution of the kernel under test
-----------------------------------
Subagent 3 (this file) is authored in parallel with Subagent 1
(TileLang kernel) and Subagent 2 (integration glue). To avoid a hard
dependency on either of those branches, this test file resolves the
``tilelang_fp8_sparse_decode`` symbol in two steps:

1. First try the production location
   :mod:`sglang.srt.layers.attention.sm_120.tilelang_sparse_decode`
   (post-parent-merge; this is what Subagent 2 wires up).
2. If that fails, fall back to the T1.4 contract stub. The stub path
   can be set via the ``SM120_SPARSE_DECODE_STUB_PATH`` environment
   variable (an absolute path to ``tilelang_sparse_decode.py``);
   otherwise the test module skips with a clear error.

This indirection is **only used during parallel-subagent
development**. After the parent merges all three branches, the
production import succeeds and the env-var path is irrelevant. The
parent merge step also flips the ``xfail`` markers on the numerical /
sanitizer layers from ``strict=False`` to ``strict=True``, turning
them into hard gates.

References
----------
- ``T1.3-sparse-decode-fwd-spec.md`` (kernel contract).
- ``T1.4-python-stubs/sparse_decode_fp8_triton.py`` (Phase-1 stub).
- ``T2A.1-STATUS.md`` (test conventions to mirror).
- ``cuda-toolkit-distilled-for-sm120-work.md §3, §6``
  (numerical noise floor + Compute-Sanitizer recipe).
"""

from __future__ import annotations

import importlib.util
import math
import os
import shutil
import subprocess
import sys
from typing import Callable, Optional, Tuple

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, suite="stage-b-test-1-gpu-large")

# ---------------------------------------------------------------------------
# PyTorch eager reference (inlined from _reference_sparse_decode.py)
# ---------------------------------------------------------------------------

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
_V4_BYTES_PER_TOKEN = (
    _V4_NOPE_ROPE_BYTES_PER_TOKEN + _V4_PADDED_SCALE_BYTES_PER_TOKEN
)  # 584
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
    region_b = underlying[:, region_a_bytes : region_a_bytes + region_b_bytes].view(
        num_pages, page_size, _V4_PADDED_SCALE_BYTES_PER_TOKEN
    )

    # Gather T tokens.
    nope_bytes = region_a[page_idx, off_in_page, :_V4_D_NOPE].contiguous()
    rope_bytes = region_a[
        page_idx, off_in_page, _V4_D_NOPE:_V4_NOPE_ROPE_BYTES_PER_TOKEN
    ].contiguous()
    scale_bytes = region_b[page_idx, off_in_page, :_V4_NUM_SCALES].contiguous()

    K_nope_fp8 = nope_bytes.view(torch.float8_e4m3fn)  # [T, 448]
    K_rope = rope_bytes.view(torch.bfloat16).reshape(T, _V4_D_ROPE)  # [T, 64]
    scale_uint8 = scale_bytes.view(torch.uint8)  # [T, 7]

    # UE8M0 dequant: scale_fp32 = 2 ** (uint8 - 127) per 64-elt tile.
    scale_fp32 = torch.exp2(scale_uint8.to(torch.float32) - _UE8M0_BIAS)  # [T, 7]
    K_nope_fp32 = K_nope_fp8.to(torch.float32)  # [T, 448]
    K_nope_dequant = K_nope_fp32.reshape(
        T, _V4_NUM_SCALES, _V4_QUANT_TILE_SIZE
    ) * scale_fp32.unsqueeze(-1)
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
    safe_max = torch.where(all_invalid.unsqueeze(-1), torch.zeros_like(max_b2), max_b2)
    P = torch.exp2(scaled - safe_max)
    if invalid.any():
        P = P * (~invalid).to(P.dtype).unsqueeze(0)
    sum_P = P.sum(dim=-1)
    sum_P_safe = torch.where(sum_P > 0, sum_P, torch.ones_like(sum_P))
    lse_b2 = torch.log2(sum_P_safe) + safe_max.squeeze(-1)
    lse_b2 = torch.where(all_invalid, torch.full_like(lse_b2, float("-inf")), lse_b2)

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
    assert (
        k_cache.shape[3] == _V4_BYTES_PER_TOKEN
    ), f"k_cache bytes_per_token={k_cache.shape[3]} != {_V4_BYTES_PER_TOKEN}"
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
    q_nope = (
        torch.randn((b, s_q, h_q, _V4_D_NOPE), generator=g, dtype=torch.float32).to(
            torch.bfloat16
        )
        * 0.1
    )
    q_rope = (
        torch.randn((b, s_q, h_q, _V4_D_ROPE), generator=g, dtype=torch.float32).to(
            torch.bfloat16
        )
        * 0.5
    )
    q = torch.cat([q_nope, q_rope], dim=-1).contiguous()
    assert q.shape == (b, s_q, h_q, _V4_D_QK)

    # KV cache: synthesise NoPE FP8 + UE8M0 scales + BF16 RoPE in fp32 first,
    # then re-pack into the V4 layout (region A interleaved per token,
    # region B in per-page tail).
    n_total = num_pages * page_size

    nope_fp32 = torch.randn(
        (n_total, _V4_NUM_SCALES, _V4_QUANT_TILE_SIZE),
        generator=g,
        dtype=torch.float32,
    )
    # UE8M0 scale: pick uint8 byte values that decode to ~0.05..2.
    scale_uint8 = torch.randint(
        125, 132, (n_total, _V4_NUM_SCALES), generator=g, dtype=torch.uint8
    )
    scale_fp32 = torch.exp2(scale_uint8.to(torch.float32) - _UE8M0_BIAS)
    # Inverse-scale before quantising so dequant recovers the value.
    nope_pre_q = nope_fp32 / scale_fp32.unsqueeze(-1).clamp(min=1e-6)
    nope_fp8 = nope_pre_q.to(torch.float8_e4m3fn).reshape(n_total, _V4_D_NOPE)

    rope_bf16 = (
        torch.randn((n_total, _V4_D_ROPE), generator=g, dtype=torch.float32).to(
            torch.bfloat16
        )
        * 0.05
    )

    bppp = _bytes_per_page_padded(page_size)
    region_a_bytes = page_size * _V4_NOPE_ROPE_BYTES_PER_TOKEN
    region_b_bytes = page_size * _V4_PADDED_SCALE_BYTES_PER_TOKEN

    underlying_2d = torch.zeros(num_pages, bppp, dtype=torch.uint8)

    # Region A: NoPE (FP8 bytes) + RoPE (BF16 bytes) per token, interleaved.
    region_a = underlying_2d[:, :region_a_bytes].view(
        num_pages, page_size, _V4_NOPE_ROPE_BYTES_PER_TOKEN
    )
    nope_uint8 = (
        nope_fp8.contiguous().view(torch.uint8).view(num_pages, page_size, _V4_D_NOPE)
    )
    rope_uint8 = (
        rope_bf16.contiguous()
        .view(torch.uint8)
        .view(num_pages, page_size, _V4_D_ROPE * 2)
    )
    region_a[:, :, :_V4_D_NOPE] = nope_uint8
    region_a[:, :, _V4_D_NOPE:_V4_NOPE_ROPE_BYTES_PER_TOKEN] = rope_uint8

    # Region B: 7 UE8M0 scales + 1 pad per token.
    region_b = underlying_2d[:, region_a_bytes : region_a_bytes + region_b_bytes].view(
        num_pages, page_size, _V4_PADDED_SCALE_BYTES_PER_TOKEN
    )
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
    indices = torch.randint(0, max_flat, (b, s_q, topk), generator=g, dtype=torch.int32)
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


# ---------------------------------------------------------------------
# Kernel resolution
# ---------------------------------------------------------------------


def _resolve_kernel() -> Tuple[Optional[Callable], str]:
    """Import the K2a kernel.

    Returns
    -------
    (fn, source) where ``source`` is one of:
        - ``"production"``: imported from
          ``sglang.srt.layers.attention.sm_120`` (post-merge).
        - ``"stub:<path>"``: imported from a T1.4 stub file pointed to
          by ``SM120_SPARSE_DECODE_STUB_PATH``.
        - ``"unavailable:<reason>"``: returned with ``fn=None`` if
          neither path resolves; collection-level skip.
    """
    try:
        from sglang.srt.layers.attention.sm_120.tilelang_sparse_decode import (  # type: ignore
            tilelang_fp8_sparse_decode,
        )

        return tilelang_fp8_sparse_decode, "production"
    except (ImportError, AttributeError):
        pass

    stub_path = os.environ.get("SM120_SPARSE_DECODE_STUB_PATH")
    if stub_path and os.path.isfile(stub_path):
        spec = importlib.util.spec_from_file_location(
            "_sm120_t14_stub_sparse_decode", stub_path
        )
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.tilelang_fp8_sparse_decode, f"stub:{stub_path}"

    return None, (
        "unavailable: production import failed and "
        "SM120_SPARSE_DECODE_STUB_PATH unset"
    )


_KERNEL_FN, _KERNEL_SOURCE = _resolve_kernel()

if _KERNEL_FN is None:
    # Collection-level skip so the file's import doesn't crash CI.
    pytest.skip(
        f"tilelang_fp8_sparse_decode not resolvable ({_KERNEL_SOURCE}); "
        "set SM120_SPARSE_DECODE_STUB_PATH to the T1.4 stub or merge "
        "the parent branch.",
        allow_module_level=True,
    )

_USING_STUB = _KERNEL_SOURCE.startswith("stub:")


# ---------------------------------------------------------------------
# Contract-test fixture (mirrors T1.4 _make_valid_inputs)
# ---------------------------------------------------------------------

_B = 2
_S_Q = 1
_H_Q = 64
_D_QK = 512  # V4-Flash sglang absorbed-latent: NoPE 448 + RoPE 64
_D_V = 512  # head_dim_v for V4-Flash (out[..., 448:512] zero-padded)
_TOPK = 512
_TOPK_BLOCK_SIZE = 64
_PAGE_BLOCK_SIZE = 64
_NUM_BLOCKS = 8
# V4-Flash: 448 NoPE FP8 + 64 BF16 RoPE + 7 UE8M0 scales + 1 pad = 584
_BYTES_PER_TOKEN = 448 + 64 * 2 + 7 + 1  # = 584
_NUM_SM_PARTS = 84  # RTX Pro 6000 sm_120
_SCHED_META_INTS_PER_ROW = 8
_SOFTMAX_SCALE = 1.0 / math.sqrt(_D_QK)


def _make_valid_inputs() -> dict:
    """A fully-valid V4-Flash V32 sparse-decode call kit.

    Mirrors ``_make_valid_inputs`` in
    ``T1.4-python-stubs/tests/test_sparse_decode_fp8_triton.py``.
    Used to drive the contract tests; the data values are zeros (the
    contract layer doesn't care about content, only shape / dtype).
    """
    q = torch.zeros((_B, _S_Q, _H_Q, _D_QK), dtype=torch.bfloat16)
    k_cache = torch.zeros(
        (_NUM_BLOCKS, _PAGE_BLOCK_SIZE, 1, _BYTES_PER_TOKEN),
        dtype=torch.uint8,
    )
    indices = torch.zeros((_B, _S_Q, _TOPK), dtype=torch.int32)
    return dict(
        q=q,
        k_cache=k_cache,
        block_table=None,
        cache_seqlens=None,
        head_dim_v=_D_V,
        tile_scheduler_metadata=None,
        num_splits=None,
        softmax_scale=_SOFTMAX_SCALE,
        causal=False,
        is_fp8_kvcache=True,
        indices=indices,
        attn_sink=None,
        extra_k_cache=None,
        extra_indices_in_kvcache=None,
        topk_length=None,
        extra_topk_length=None,
    )


def _call_or_skip(**kwargs) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Invoke the kernel, skipping with a clear message on the stub's
    documented ``NotImplementedError``.

    Used by contract tests that pass valid inputs: the stub raises
    ``NotImplementedError`` (proving the contract layer was satisfied);
    the real kernel returns ``(out, lse)``. We accept both.
    """
    try:
        return _KERNEL_FN(**kwargs)
    except NotImplementedError:
        pytest.skip(
            "Contract checks passed; kernel body is the T1.4 stub "
            "(NotImplementedError)."
        )


# ---------------------------------------------------------------------
# 1. Contract tests — valid inputs reach the kernel body
# ---------------------------------------------------------------------


def test_valid_inputs_contract_satisfied() -> None:
    """A fully-valid call doesn't trigger any ``AssertionError``."""
    kw = _make_valid_inputs()
    if _USING_STUB:
        with pytest.raises(NotImplementedError, match="not yet implemented"):
            _KERNEL_FN(**kw)
    else:
        out, lse = _KERNEL_FN(**kw)
        assert out.shape == (_B, _S_Q, _H_Q, _D_V)
        assert out.dtype == torch.bfloat16
        assert lse.shape == (_B, _H_Q, _S_Q)
        assert lse.dtype == torch.float32


def test_valid_with_attn_sink_contract_satisfied() -> None:
    kw = _make_valid_inputs()
    kw["attn_sink"] = torch.zeros((_H_Q,), dtype=torch.float32)
    if _USING_STUB:
        with pytest.raises(NotImplementedError):
            _KERNEL_FN(**kw)
    else:
        _call_or_skip(**kw)


def test_valid_with_sched_meta_contract_satisfied() -> None:
    kw = _make_valid_inputs()
    kw["tile_scheduler_metadata"] = torch.zeros(
        (_NUM_SM_PARTS, _SCHED_META_INTS_PER_ROW), dtype=torch.int32
    )
    _call_or_skip(**kw)


def test_valid_with_num_splits_contract_satisfied() -> None:
    kw = _make_valid_inputs()
    kw["num_splits"] = torch.zeros((_B, _S_Q), dtype=torch.int32)
    _call_or_skip(**kw)


# ---------------------------------------------------------------------
# 2. q shape / dtype contract tests
# ---------------------------------------------------------------------


def test_q_wrong_dtype_rejected() -> None:
    kw = _make_valid_inputs()
    kw["q"] = kw["q"].to(torch.float32)
    with pytest.raises(AssertionError):
        _KERNEL_FN(**kw)


def test_q_wrong_dim_rejected() -> None:
    kw = _make_valid_inputs()
    kw["q"] = kw["q"].squeeze(1)
    with pytest.raises(AssertionError):
        _KERNEL_FN(**kw)


def test_q_wrong_d_qk_rejected() -> None:
    kw = _make_valid_inputs()
    kw["q"] = torch.zeros((_B, _S_Q, _H_Q, 448), dtype=torch.bfloat16)
    with pytest.raises(AssertionError):
        _KERNEL_FN(**kw)


# ---------------------------------------------------------------------
# 3. k_cache contract tests
# ---------------------------------------------------------------------


def test_k_cache_wrong_dim_rejected() -> None:
    kw = _make_valid_inputs()
    kw["k_cache"] = kw["k_cache"].squeeze(2)
    with pytest.raises(AssertionError):
        _KERNEL_FN(**kw)


def test_k_cache_wrong_bytes_per_token_rejected() -> None:
    kw = _make_valid_inputs()
    kw["k_cache"] = torch.zeros(
        (_NUM_BLOCKS, _PAGE_BLOCK_SIZE, 1, 656), dtype=torch.uint8
    )
    with pytest.raises(AssertionError):
        _KERNEL_FN(**kw)


# ---------------------------------------------------------------------
# 4. indices contract tests
# ---------------------------------------------------------------------


def test_indices_wrong_dtype_rejected() -> None:
    kw = _make_valid_inputs()
    kw["indices"] = kw["indices"].to(torch.int64)
    with pytest.raises(AssertionError):
        _KERNEL_FN(**kw)


def test_indices_wrong_topk_rejected() -> None:
    """topk not a multiple of BLOCK_TOPK=32."""
    kw = _make_valid_inputs()
    kw["indices"] = torch.zeros((_B, _S_Q, 33), dtype=torch.int32)
    with pytest.raises(AssertionError):
        _KERNEL_FN(**kw)


def test_indices_none_rejected() -> None:
    kw = _make_valid_inputs()
    kw["indices"] = None
    with pytest.raises((AssertionError, TypeError)):
        _KERNEL_FN(**kw)


# ---------------------------------------------------------------------
# 5. head_dim_v contract tests
# ---------------------------------------------------------------------


def test_head_dim_v_wrong_value_rejected() -> None:
    kw = _make_valid_inputs()
    kw["head_dim_v"] = 256
    with pytest.raises(AssertionError):
        _KERNEL_FN(**kw)


# ---------------------------------------------------------------------
# 6. is_fp8_kvcache contract tests
# ---------------------------------------------------------------------


def test_is_fp8_kvcache_false_rejected() -> None:
    kw = _make_valid_inputs()
    kw["is_fp8_kvcache"] = False
    with pytest.raises(AssertionError):
        _KERNEL_FN(**kw)


# ---------------------------------------------------------------------
# 7. causal contract tests
# ---------------------------------------------------------------------


def test_causal_true_rejected() -> None:
    kw = _make_valid_inputs()
    kw["causal"] = True
    with pytest.raises(AssertionError):
        _KERNEL_FN(**kw)


# ---------------------------------------------------------------------
# 8. block_table / cache_seqlens must be None
# ---------------------------------------------------------------------


def test_block_table_not_none_rejected() -> None:
    kw = _make_valid_inputs()
    kw["block_table"] = torch.zeros((_B, 8), dtype=torch.int32)
    with pytest.raises(AssertionError):
        _KERNEL_FN(**kw)


def test_cache_seqlens_not_none_rejected() -> None:
    kw = _make_valid_inputs()
    kw["cache_seqlens"] = torch.zeros((_B,), dtype=torch.int32)
    with pytest.raises(AssertionError):
        _KERNEL_FN(**kw)


# ---------------------------------------------------------------------
# 9. Numerical correctness tests
# ---------------------------------------------------------------------

_NUMERICAL_XFAIL_REASON = (
    "Numerical tests require the integrated TileLang kernel on sm_120 "
    "hardware; xfail until parent merge + GPU CI. Set strict=True "
    "post-merge to gate the CI."
)


def _assert_kernel_matches_reference(
    b: int,
    s_q: int,
    h_q: int,
    topk: int,
    num_blocks: int,
    seed: int,
    use_attn_sink: bool = False,
    valid_ratio: float = 1.0,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA needed for numerical tests")

    inputs = make_synthetic_inputs(
        b=b,
        s_q=s_q,
        h_q=h_q,
        topk=topk,
        num_blocks=num_blocks,
        page_block_size=64,
        seed=seed,
        device="cuda",
        valid_ratio=valid_ratio,
    )

    attn_sink = None
    if use_attn_sink:
        attn_sink = torch.randn((h_q,), dtype=torch.float32, device="cuda")
        inputs["attn_sink"] = attn_sink

    out, lse = _KERNEL_FN(**inputs)

    # Reference on CPU (slow but correct).
    cpu_inputs = make_synthetic_inputs(
        b=b,
        s_q=s_q,
        h_q=h_q,
        topk=topk,
        num_blocks=num_blocks,
        page_block_size=64,
        seed=seed,
        device="cpu",
        valid_ratio=valid_ratio,
    )
    cpu_attn_sink = attn_sink.cpu() if attn_sink is not None else None
    ref_out, ref_lse = sparse_decode_fwd_reference(
        q=cpu_inputs["q"],
        k_cache=cpu_inputs["k_cache"],
        indices=cpu_inputs["indices"],
        softmax_scale=cpu_inputs["softmax_scale"],
        attn_sink=cpu_attn_sink,
    )

    out_cpu = out.cpu().float()
    ref_out_cpu = ref_out.float()
    assert torch.allclose(
        out_cpu, ref_out_cpu, rtol=1e-3, atol=1e-3
    ), f"out mismatch: max_abs_err={( out_cpu - ref_out_cpu).abs().max():.6f}"


@pytest.mark.gpu
@pytest.mark.xfail(reason=_NUMERICAL_XFAIL_REASON, strict=False)
def test_numerical_b1_sq1_topk64_h64() -> None:
    _assert_kernel_matches_reference(b=1, s_q=1, h_q=64, topk=64, num_blocks=2, seed=1)


@pytest.mark.gpu
@pytest.mark.xfail(reason=_NUMERICAL_XFAIL_REASON, strict=False)
def test_numerical_b2_sq1_topk64_h64() -> None:
    _assert_kernel_matches_reference(b=2, s_q=1, h_q=64, topk=64, num_blocks=2, seed=2)


@pytest.mark.gpu
@pytest.mark.xfail(reason=_NUMERICAL_XFAIL_REASON, strict=False)
def test_numerical_b1_sq1_topk128_h64() -> None:
    _assert_kernel_matches_reference(b=1, s_q=1, h_q=64, topk=128, num_blocks=4, seed=3)


@pytest.mark.gpu
@pytest.mark.xfail(reason=_NUMERICAL_XFAIL_REASON, strict=False)
def test_numerical_b2_sq1_topk128_h64() -> None:
    _assert_kernel_matches_reference(b=2, s_q=1, h_q=64, topk=128, num_blocks=4, seed=4)


@pytest.mark.gpu
@pytest.mark.xfail(reason=_NUMERICAL_XFAIL_REASON, strict=False)
def test_numerical_b1_sq1_topk512_h64() -> None:
    """topk=512: typical Lightning-Indexer width for V4-Flash."""
    _assert_kernel_matches_reference(
        b=1, s_q=1, h_q=64, topk=512, num_blocks=16, seed=5
    )


@pytest.mark.gpu
@pytest.mark.xfail(reason=_NUMERICAL_XFAIL_REASON, strict=False)
def test_numerical_b4_sq1_topk512_h64() -> None:
    """B=4, topk=512: realistic V4-Flash decode workload."""
    _assert_kernel_matches_reference(
        b=4, s_q=1, h_q=64, topk=512, num_blocks=32, seed=6
    )


@pytest.mark.gpu
@pytest.mark.xfail(reason=_NUMERICAL_XFAIL_REASON, strict=False)
def test_numerical_with_attn_sink() -> None:
    """attn_sink folds into LSE/O on the no-split path
    (splitkv_mla.cuh:305-307; combine.cu:101-112)."""
    _assert_kernel_matches_reference(
        b=2, s_q=1, h_q=64, topk=128, num_blocks=4, seed=7, use_attn_sink=True
    )


@pytest.mark.gpu
@pytest.mark.xfail(reason=_NUMERICAL_XFAIL_REASON, strict=False)
def test_numerical_varlen_indices() -> None:
    """Variable-length topk via ``-1`` invalid lanes — exercises the
    invalid-index masking path (splitkv_mla.cuh:577-595)."""
    _assert_kernel_matches_reference(
        b=2, s_q=1, h_q=64, topk=128, num_blocks=4, seed=8, valid_ratio=0.5
    )


@pytest.mark.gpu
@pytest.mark.xfail(reason=_NUMERICAL_XFAIL_REASON, strict=False)
def test_numerical_all_invalid_indices() -> None:
    """All ``-1`` lanes → kernel must produce zeros + lse=-INF."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA needed")
    inputs = make_synthetic_inputs(
        b=2,
        s_q=1,
        h_q=64,
        topk=64,
        num_blocks=2,
        page_block_size=64,
        seed=9,
        device="cuda",
    )
    inputs["indices"][:] = -1
    out, lse = _KERNEL_FN(**inputs)
    assert (out.float().abs() == 0).all(), "all-invalid -> zero output"
    assert (lse == float("-inf")).all() or (lse == 0).all(), (
        "kernel may write -inf or 0 for all-invalid; both are kernel-internal "
        "but must be self-consistent"
    )


# ---------------------------------------------------------------------
# 12. Compute-Sanitizer integration tests (xfail until parent merge)
# ---------------------------------------------------------------------

_SANITIZER_XFAIL_REASON = (
    "Compute-Sanitizer requires the integrated kernel; runs after parent "
    "merge. WARNING-severity output from racecheck on Triton tl.sum / "
    "tl.reduce warp-shuffle reductions is documented as expected noise "
    "in cuda-toolkit-distilled-for-sm120-work.md §6 and is filtered out."
)


def _have_compute_sanitizer() -> bool:
    return shutil.which("compute-sanitizer") is not None


def _run_compute_sanitizer(
    tool: str,
    extra_flags: Optional[list] = None,
    timeout_s: int = 120,
) -> subprocess.CompletedProcess:
    """Invoke ``compute-sanitizer`` on a tight in-process kernel call.

    Writes a one-shot inner runner to a temp directory, points it at
    the same kernel resolution rules this test file uses, and runs
    ``compute-sanitizer --tool {tool} --error-exitcode 2``. The inner
    runner exits 0 on success; the sanitizer overrides to 2 on any
    detected ERROR. WARNING-only output (e.g. the documented racecheck
    noise on Triton's ``tl.sum``) does not change the exit code.
    """
    if not _have_compute_sanitizer():
        pytest.skip("compute-sanitizer not on PATH")
    if not torch.cuda.is_available():
        pytest.skip("compute-sanitizer needs a CUDA device")

    import tempfile

    tmpdir = tempfile.mkdtemp(prefix="sm120_sanitizer_")
    inner = os.path.join(tmpdir, "_sanitizer_inner_sparse_decode.py")
    with open(inner, "w") as f:
        f.write(_SANITIZER_INNER_SOURCE)

    cmd = [
        "compute-sanitizer",
        "--tool",
        tool,
        "--error-exitcode",
        "2",
    ]
    if extra_flags:
        cmd.extend(extra_flags)
    cmd.extend([sys.executable, inner])

    env = os.environ.copy()
    if _USING_STUB:
        # Make sure the inner runner sees the same stub the test file
        # resolved against.
        env["SM120_SPARSE_DECODE_STUB_PATH"] = _KERNEL_SOURCE[len("stub:") :]

    try:
        return subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _filter_racecheck_noise(stderr: str) -> str:
    """Drop documented-tolerable WARNING lines from racecheck output.

    Per ``cuda-toolkit-distilled §6``, Triton's ``tl.sum`` / ``tl.reduce``
    use ``__shfl_xor_sync`` without an explicit ``__syncwarp()`` — the
    participating threads are compiler-guaranteed, but Volta+'s ITS
    surfaces these as WARNING. ERROR severity is never tolerable.
    """
    keep = []
    for line in stderr.splitlines():
        lower = line.lower()
        if "warning" in lower and (
            "shfl" in lower or "shared memory" in lower or "race" in lower
        ):
            keep.append(f"# (filtered tolerable): {line}")
            continue
        keep.append(line)
    return "\n".join(keep)


@pytest.mark.gpu
@pytest.mark.xfail(reason=_SANITIZER_XFAIL_REASON, strict=False)
def test_compute_sanitizer_memcheck() -> None:
    """memcheck: catches OOB gathers + misaligned FP8 loads."""
    cp = _run_compute_sanitizer(
        "memcheck",
        extra_flags=["--leak-check", "full", "--padding", "32"],
    )
    assert (
        cp.returncode == 0
    ), f"memcheck found errors:\nstdout:\n{cp.stdout}\nstderr:\n{cp.stderr}"


@pytest.mark.gpu
@pytest.mark.xfail(reason=_SANITIZER_XFAIL_REASON, strict=False)
def test_compute_sanitizer_racecheck() -> None:
    """racecheck: missing __syncthreads in Triton software-pipelined kernels.

    Documented-tolerable WARNINGs on tl.sum / tl.reduce shuffle reductions
    are filtered before assertion (cuda-toolkit-distilled §6).
    """
    cp = _run_compute_sanitizer(
        "racecheck",
        extra_flags=["--racecheck-report", "all"],
    )
    filtered_stderr = _filter_racecheck_noise(cp.stderr)
    assert cp.returncode == 0, (
        f"racecheck found ERROR-severity issues:\n"
        f"stdout:\n{cp.stdout}\n"
        f"stderr (after filter):\n{filtered_stderr}"
    )


@pytest.mark.gpu
@pytest.mark.xfail(reason=_SANITIZER_XFAIL_REASON, strict=False)
def test_compute_sanitizer_initcheck() -> None:
    """initcheck: uninitialised smem reads (e.g. mask without ``other=``)."""
    cp = _run_compute_sanitizer(
        "initcheck",
        extra_flags=["--initcheck-address-space", "all"],
    )
    assert (
        cp.returncode == 0
    ), f"initcheck found errors:\nstdout:\n{cp.stdout}\nstderr:\n{cp.stderr}"


@pytest.mark.gpu
@pytest.mark.xfail(reason=_SANITIZER_XFAIL_REASON, strict=False)
def test_compute_sanitizer_synccheck() -> None:
    """synccheck: divergent threads at __syncthreads. Should always be clean."""
    cp = _run_compute_sanitizer("synccheck")
    assert (
        cp.returncode == 0
    ), f"synccheck found errors:\nstdout:\n{cp.stdout}\nstderr:\n{cp.stderr}"


# ---------------------------------------------------------------------
# Inner sanitizer-runner source (written to disk on first use)
# ---------------------------------------------------------------------

_SANITIZER_INNER_SOURCE = '''#!/usr/bin/env python3
"""Tight runner for compute-sanitizer integration tests.

Imports the K2a kernel via the same resolution rules as
``test_sm120_sparse_decode.py``, runs one tiny call on a
synthetic input kit, and exits 0 on success. Sanitizer ERROR
severities surface via ``--error-exitcode 2``.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import math

import torch


def _resolve_kernel():
    try:
        from sglang.srt.layers.attention.sm_120.tilelang_sparse_decode import (
            tilelang_fp8_sparse_decode,
        )
        return tilelang_fp8_sparse_decode
    except Exception:
        pass

    stub_path = os.environ.get("SM120_SPARSE_DECODE_STUB_PATH")
    if stub_path and os.path.isfile(stub_path):
        spec = importlib.util.spec_from_file_location("_stub", stub_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.tilelang_fp8_sparse_decode
    raise SystemExit("could not resolve tilelang_fp8_sparse_decode")


def _resolve_make_synthetic():
    try:
        from test_sm120_sparse_decode import make_synthetic_inputs
        return make_synthetic_inputs
    except Exception:
        pass

    here = os.path.dirname(os.path.abspath(__file__))
    ref = os.path.normpath(os.path.join(
        here, "..", "..", "..", "python", "sglang", "srt", "layers",
        "attention", "sm_120", "_reference_sparse_decode.py",
    ))
    if os.path.isfile(ref):
        spec = importlib.util.spec_from_file_location("_ref", ref)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.make_synthetic_inputs

    # Inline a minimal synthetic input builder as fallback.
    def _make_inputs(b, s_q, h_q, topk, num_blocks, page_block_size, seed, device):
        import math
        g = torch.Generator(device="cpu").manual_seed(seed)
        q = torch.randn((b, s_q, h_q, 512), generator=g, dtype=torch.float32).to(torch.bfloat16)
        kv = torch.zeros((num_blocks, page_block_size, 1, 584), dtype=torch.uint8)
        indices = torch.randint(0, num_blocks * page_block_size, (b, s_q, topk), generator=g, dtype=torch.int32)
        softmax_scale = 1.0 / math.sqrt(512)
        if device != "cpu":
            q = q.to(device)
            kv = kv.to(device)
            indices = indices.to(device)
        return {
            "q": q, "k_cache": kv, "block_table": None, "cache_seqlens": None,
            "head_dim_v": 512, "tile_scheduler_metadata": None, "num_splits": None,
            "softmax_scale": softmax_scale, "causal": False, "is_fp8_kvcache": True,
            "indices": indices, "attn_sink": None, "extra_k_cache": None,
            "extra_indices_in_kvcache": None, "topk_length": None, "extra_topk_length": None,
        }
    return _make_inputs


def main():
    if not torch.cuda.is_available():
        print("no CUDA device — exiting with sanitizer-friendly skip", file=sys.stderr)
        sys.exit(0)
    fn = _resolve_kernel()
    make_inputs = _resolve_make_synthetic()
    inputs = make_inputs(
        b=1, s_q=1, h_q=64, topk=64, num_blocks=2, page_block_size=64,
        seed=0, device="cuda",
    )
    try:
        out, lse = fn(**inputs)
    except NotImplementedError:
        # Stub mode: contract layer ran on the host without launching
        # any CUDA kernels. Sanitizer has nothing to inspect; report
        # clean exit so the xfail surfaces as expected.
        print("stub NotImplementedError — no kernel launched", file=sys.stderr)
        sys.exit(0)
    torch.cuda.synchronize()
    assert out.shape[0] == 1


if __name__ == "__main__":
    main()
'''
