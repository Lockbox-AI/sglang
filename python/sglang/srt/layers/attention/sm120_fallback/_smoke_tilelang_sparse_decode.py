"""Liveness smoke for the sm_120 TileLang FP8 sparse-decode V2 kernel (T2A.3 K2a).

Synthesises the 4-D ``[num_pages, page_size, h_kv=1, 584]`` V4-Flash
input view (with the underlying 2-D ``[num_pages, bytes_per_page_padded]``
buffer's region-A / region-B split) and proves that:

1. ``tilelang_fp8_sparse_decode`` (V2) imports cleanly,
2. its TileLang JIT compiles on sm_120, and
3. running on a tiny synthetic V4-layout input produces tensors of the
   right shape and dtype with no NaN/Inf.

This is **not** a numerical-correctness test. The PyTorch eager
reference at ``_reference_sparse_decode.py`` (also updated to V4) is
the numerical oracle.

Run from the fork's ``python/`` directory::

    python -m sglang.srt.layers.attention.sm120_fallback._smoke_tilelang_sparse_decode
"""

import math
import sys

import torch

from sglang.srt.layers.attention.sm120_fallback.tilelang_sparse_decode import (
    tilelang_fp8_sparse_decode,
)


# V4-Flash layout constants (mirror the kernel module's _V4_* constants).
_V4_D_NOPE = 448
_V4_D_ROPE = 64
_V4_D_QK = _V4_D_NOPE + _V4_D_ROPE  # 512
_V4_D_V = 512
_V4_NUM_SCALES = _V4_D_NOPE // 64  # 7
_V4_SCALE_PAD = 1
_V4_PADDED_SCALE_BYTES_PER_TOKEN = _V4_NUM_SCALES + _V4_SCALE_PAD  # 8
_V4_NOPE_ROPE_BYTES_PER_TOKEN = _V4_D_NOPE + _V4_D_ROPE * 2  # 576
_V4_BYTES_PER_TOKEN = _V4_NOPE_ROPE_BYTES_PER_TOKEN + _V4_PADDED_SCALE_BYTES_PER_TOKEN  # 584


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _bytes_per_page_padded(page_size: int) -> int:
    """Mirror ``DeepSeekV4SingleKVPool.bytes_per_page_padded`` rule.

    Per ``mem_cache/deepseekv4_memory_pool.py:103-105``::

        bytes_per_page_padded = ceil_div(page_size * 584, 576) * 576
    """
    raw = page_size * _V4_BYTES_PER_TOKEN
    return _ceil_div(raw, 576) * 576


def _build_synthetic_inputs(
    B: int = 1,
    s_q: int = 1,
    H_q: int = 64,
    num_pages: int = 4,
    page_size: int = 64,
    topk: int = 32,
    seed: int = 1234,
):
    """Build a minimal V4-layout (q, kv_cache, indices) triple."""
    if not torch.cuda.is_available():
        raise SystemExit("smoke requires CUDA (sm_120)")

    g = torch.Generator(device="cuda").manual_seed(seed)

    # --- q : bf16 [B, s_q, H_q, D_qk=512] ------------------------------
    # Q[..., :448] = NoPE-like (small magnitudes); Q[..., 448:512] = RoPE-like
    # (larger magnitudes, exercising the Q-RoPE × K-RoPE GEMM).
    q_nope = torch.randn(B, s_q, H_q, _V4_D_NOPE, dtype=torch.bfloat16, device="cuda", generator=g) * 0.1
    q_rope = torch.randn(B, s_q, H_q, _V4_D_ROPE, dtype=torch.bfloat16, device="cuda", generator=g) * 0.5
    q = torch.cat([q_nope, q_rope], dim=-1).contiguous()

    # --- kv_cache : 4-D V4-Flash view of an underlying 2-D buffer ------
    bppp = _bytes_per_page_padded(page_size)
    region_a_bytes = page_size * _V4_NOPE_ROPE_BYTES_PER_TOKEN  # NoPE+RoPE
    region_b_bytes = page_size * _V4_PADDED_SCALE_BYTES_PER_TOKEN  # scales+pad
    raw = page_size * _V4_BYTES_PER_TOKEN
    assert region_a_bytes + region_b_bytes == raw, (region_a_bytes, region_b_bytes, raw)

    underlying_2d = torch.zeros(num_pages, bppp, dtype=torch.uint8, device="cuda")

    # Region A: NoPE+RoPE per token, interleaved.
    n_total = num_pages * page_size
    nope_bf16 = torch.randn(n_total, _V4_D_NOPE, dtype=torch.bfloat16, device="cuda", generator=g) * 0.1
    nope_fp8 = nope_bf16.to(torch.float8_e4m3fn)
    rope_bf16 = torch.randn(n_total, _V4_D_ROPE, dtype=torch.bfloat16, device="cuda", generator=g) * 0.05

    region_a = underlying_2d[:, :region_a_bytes].view(num_pages, page_size, _V4_NOPE_ROPE_BYTES_PER_TOKEN)
    # Write FP8 NoPE bytes (token i: bytes [0, 448)).
    region_a[:, :, :_V4_D_NOPE].view(torch.float8_e4m3fn).copy_(
        nope_fp8.view(num_pages, page_size, _V4_D_NOPE)
    )
    # Write BF16 RoPE bytes (token i: bytes [448, 576)).
    region_a[:, :, _V4_D_NOPE:_V4_NOPE_ROPE_BYTES_PER_TOKEN].contiguous().view(
        torch.bfloat16
    ).copy_(rope_bf16.view(num_pages, page_size, _V4_D_ROPE).contiguous().view(torch.bfloat16))
    # Workaround: the rope slice is non-contiguous in dim 2 (stride 576, not 128).
    # Let's just directly assign uint8 bytes to keep things simple.
    rope_uint8 = rope_bf16.contiguous().view(torch.uint8).view(
        num_pages, page_size, _V4_D_ROPE * 2
    )
    region_a[:, :, _V4_D_NOPE:_V4_NOPE_ROPE_BYTES_PER_TOKEN] = rope_uint8

    # Region B: 7 UE8M0 scales + 1 pad per token, in the per-page tail.
    # UE8M0 byte 120 ↔ scale = 2^(120-127) = 2^-7 ≈ 0.0078125 (matches the
    # producer's behavior on N(0,1) input — see RUN-4A-STATUS.md §3).
    # We use a slightly larger scale to exercise the dequant path more.
    scale_uint8 = torch.full(
        (num_pages, page_size, _V4_NUM_SCALES), 130, dtype=torch.uint8, device="cuda"
    )  # 2^(130-127) = 8.0 — modest non-trivial scale
    region_b = underlying_2d[:, region_a_bytes : region_a_bytes + region_b_bytes].view(
        num_pages, page_size, _V4_PADDED_SCALE_BYTES_PER_TOKEN
    )
    region_b[:, :, :_V4_NUM_SCALES] = scale_uint8
    # region_b[:, :, _V4_NUM_SCALES:] left at 0 (the pad byte).

    # Now build the 4-D view that sglang's radix backend would pass.
    kv_cache_4d = underlying_2d[:, : page_size * _V4_BYTES_PER_TOKEN].view(
        num_pages, page_size, 1, _V4_BYTES_PER_TOKEN
    )

    # --- indices : int32 [B, s_q, topk] ---------------------------------
    flat_indices = torch.arange(topk, device="cuda", dtype=torch.int32)
    indices = flat_indices.view(1, 1, topk).expand(B, s_q, topk).contiguous()

    return q, kv_cache_4d, indices


def main() -> int:
    if not torch.cuda.is_available():
        print("smoke FAIL: CUDA not available", file=sys.stderr)
        return 2

    cap = torch.cuda.get_device_capability()
    name = torch.cuda.get_device_name()
    print(f"device: {name} compute={cap[0]}.{cap[1]}")
    if cap != (12, 0):
        print(
            f"smoke WARNING: device capability is {cap}, not (12, 0); "
            "this smoke is designed for sm_120 (RTX Pro 6000 / RTX 5090)."
        )

    B, s_q, H_q = 1, 1, 64
    D_qk = _V4_D_QK  # 512
    D_v = _V4_D_V  # 512
    topk = 32
    page_size = 64
    num_pages = 4
    sm_scale = 1.0 / math.sqrt(D_qk)

    print(
        f"shapes: q=[{B},{s_q},{H_q},{D_qk}] "
        f"kv=[{num_pages},{page_size},1,{_V4_BYTES_PER_TOKEN}] "
        f"indices=[{B},{s_q},{topk}] sm_scale={sm_scale:.6f}"
    )

    q, kv_cache, indices = _build_synthetic_inputs(
        B=B, s_q=s_q, H_q=H_q,
        num_pages=num_pages, page_size=page_size, topk=topk,
    )

    print("calling tilelang_fp8_sparse_decode (V2; will JIT-compile on first run)…")
    out, lse = tilelang_fp8_sparse_decode(
        q=q,
        kv_cache=kv_cache,
        block_table=None,
        seq_lens=None,
        indices=indices,
        sm_scale=sm_scale,
        is_fp8_kvcache=True,
        h_kv=1,
        d_v=D_v,
    )

    # ---- Liveness gates --------------------------------------------------
    assert isinstance(out, torch.Tensor) and isinstance(lse, torch.Tensor), (
        f"unexpected return types: out={type(out)}, lse={type(lse)}"
    )
    assert out.dtype == torch.bfloat16, f"out dtype must be bfloat16, got {out.dtype}"
    assert lse.dtype == torch.float32, f"lse dtype must be float32, got {lse.dtype}"
    assert tuple(out.shape) == (B, s_q, H_q, D_v), (
        f"out shape {tuple(out.shape)} != expected ({B}, {s_q}, {H_q}, {D_v})"
    )
    assert tuple(lse.shape) == (B, s_q, H_q), (
        f"lse shape {tuple(lse.shape)} != expected ({B}, {s_q}, {H_q})"
    )

    out_f32 = out.float()
    n_nan_out = torch.isnan(out_f32).sum().item()
    n_inf_out = torch.isinf(out_f32).sum().item()
    n_nan_lse = torch.isnan(lse).sum().item()
    n_inf_lse = torch.isinf(lse).sum().item()
    # Verify the trailing 64 dims are zero (V2 zero-pad contract).
    out_tail = out_f32[..., _V4_D_NOPE:]
    tail_nonzero = (out_tail.abs() > 0).sum().item()
    print(
        f"out: shape={tuple(out.shape)} dtype={out.dtype} "
        f"min={out_f32.min().item():.4f} max={out_f32.max().item():.4f} "
        f"mean={out_f32.mean().item():.4f}"
    )
    print(
        f"out[..., 0:448] (V·P): "
        f"min={out_f32[..., :_V4_D_NOPE].min().item():.4f} "
        f"max={out_f32[..., :_V4_D_NOPE].max().item():.4f}"
    )
    print(
        f"out[..., 448:512] (zero-pad): "
        f"min={out_tail.min().item():.4f} "
        f"max={out_tail.max().item():.4f} "
        f"nonzero_count={tail_nonzero}"
    )
    print(
        f"lse: shape={tuple(lse.shape)} dtype={lse.dtype} "
        f"min={lse.min().item():.4f} max={lse.max().item():.4f} "
        f"mean={lse.mean().item():.4f}"
    )
    print(
        f"finiteness: out NaN={n_nan_out}, out Inf={n_inf_out}, "
        f"lse NaN={n_nan_lse}, lse Inf={n_inf_lse}"
    )
    assert n_nan_out == 0, f"out has {n_nan_out} NaNs"
    assert n_inf_out == 0, f"out has {n_inf_out} Infs"
    assert n_nan_lse == 0, f"lse has {n_nan_lse} NaNs"
    assert n_inf_lse == 0, f"lse has {n_inf_lse} Infs"
    assert tail_nonzero == 0, (
        f"out[..., 448:512] must be zero (V2 trailing-dim contract); "
        f"found {tail_nonzero} nonzero entries"
    )

    print("smoke ok (V2 layout)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
