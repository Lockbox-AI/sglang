"""Liveness smoke for the sm_120 TileLang FP8 sparse-decode kernel (T2A.3 K2a).

This is **not** a numerical-correctness test (Subagent 3 owns those). All
this script does is prove that:

1. ``tilelang_fp8_sparse_decode`` imports cleanly,
2. its TileLang JIT actually compiles on sm_120, and
3. running it on a tiny synthetic input produces tensors of the right
   shape and dtype with no NaN/Inf.

Run from the fork's ``python/`` directory:

::

    python -m sglang.srt.layers.attention.sm120_fallback._smoke_tilelang_sparse_decode

Exit codes:

- 0 — "smoke ok" printed; kernel compiled and ran.
- non-zero — assertion failure; see traceback for the gate that failed.
"""

import math
import sys

import torch

from sglang.srt.layers.attention.sm120_fallback.tilelang_sparse_decode import (
    tilelang_fp8_sparse_decode,
)


def _build_synthetic_inputs(
    B: int = 1,
    s_q: int = 1,
    H_q: int = 128,
    D_qk: int = 576,
    D_NOPE: int = 512,
    D_ROPE: int = 64,
    num_blocks: int = 4,
    page_block_size: int = 64,
    topk: int = 32,
    seed: int = 1234,
):
    """Build a minimal (q, kv_cache, indices) triple matching the V32 contract."""
    if not torch.cuda.is_available():
        raise SystemExit("smoke requires CUDA (sm_120)")

    g = torch.Generator(device="cuda").manual_seed(seed)

    # --- q : bf16 [B, s_q, H_q, D_qk] -----------------------------------
    q = torch.randn(
        B, s_q, H_q, D_qk, dtype=torch.bfloat16, device="cuda", generator=g
    ) * 0.1

    # --- kv_cache : uint8 [num_blocks, page_block_size, 656] ------------
    # We build the 656-byte packed layout token-by-token using the typed
    # views the kernel expects. This way we can plant *meaningful* (small,
    # bounded) values, not random bit-patterns that often decode to FP8 NaN.
    bytes_per_token = D_NOPE + 4 * (D_NOPE // 128) + 2 * D_ROPE  # 656
    assert bytes_per_token == 656, bytes_per_token
    kv_cache = torch.zeros(
        num_blocks, page_block_size, bytes_per_token,
        dtype=torch.uint8, device="cuda",
    )
    n_total = num_blocks * page_block_size
    kv_2d = kv_cache.view(n_total, bytes_per_token)

    # FP8 NoPE: small bf16 values cast to fp8 to keep dequant-bounded.
    nope_bf16 = torch.randn(
        n_total, D_NOPE, dtype=torch.bfloat16, device="cuda", generator=g
    ) * 0.1
    nope_fp8 = nope_bf16.to(torch.float8_e4m3fn)
    # fp8 stride 1 byte == uint8 stride 1 byte → simple byte copy via .view.
    kv_2d[:, :D_NOPE].view(torch.float8_e4m3fn).copy_(nope_fp8)

    # FP32 scales: positive, modestly above 1 to exercise dequant path.
    scales = torch.full(
        (n_total, D_NOPE // 128), 1.5, dtype=torch.float32, device="cuda"
    )
    kv_2d[:, D_NOPE:D_NOPE + 4 * (D_NOPE // 128)].view(torch.float32).copy_(scales)

    # BF16 RoPE: small bounded values.
    rope = torch.randn(
        n_total, D_ROPE, dtype=torch.bfloat16, device="cuda", generator=g
    ) * 0.05
    kv_2d[:, D_NOPE + 4 * (D_NOPE // 128):].view(torch.bfloat16).copy_(rope)

    # --- indices : int32 [B, s_q, topk] ---------------------------------
    # Pick distinct valid flat indices so the gather hits the populated
    # rows above and we exercise multiple K-rows in one BLOCK_TOPK tile.
    flat_indices = torch.arange(topk, device="cuda", dtype=torch.int32)
    indices = flat_indices.view(1, 1, topk).expand(B, s_q, topk).contiguous()

    return q, kv_cache, indices


def main() -> int:
    if not torch.cuda.is_available():
        print("smoke FAIL: CUDA not available", file=sys.stderr)
        return 2

    cap = torch.cuda.get_device_capability()
    name = torch.cuda.get_device_name()
    print(f"device: {name} compute={cap[0]}.{cap[1]}")
    if cap != (12, 0):
        # Don't hard-fail — TileLang will JIT for whatever sm_* the device
        # actually is — but make it loud so the on-instance step doesn't
        # silently mask a wrong-target run.
        print(
            f"smoke WARNING: device capability is {cap}, not (12, 0); "
            "this smoke is designed for sm_120 (RTX Pro 6000 / RTX 5090)."
        )

    # V4-Pro pre-FP4 shape (H_q=128) per the run brief. The V4-Flash
    # production config (H_q=64) is also verified — see the status doc.
    B, s_q, H_q, D_qk = 1, 1, 128, 576
    D_v = 512
    topk = 32
    sm_scale = 1.0 / math.sqrt(D_qk)

    print(
        f"shapes: q=[{B},{s_q},{H_q},{D_qk}] kv=[4,64,656] indices=[{B},{s_q},{topk}] "
        f"sm_scale={sm_scale:.6f}"
    )

    q, kv_cache, indices = _build_synthetic_inputs(
        B=B, s_q=s_q, H_q=H_q,
        D_qk=D_qk, D_NOPE=512, D_ROPE=64,
        num_blocks=4, page_block_size=64, topk=topk,
    )

    print("calling tilelang_fp8_sparse_decode (will JIT-compile on first run)…")
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
    print(
        f"out: shape={tuple(out.shape)} dtype={out.dtype} "
        f"min={out_f32.min().item():.4f} max={out_f32.max().item():.4f} "
        f"mean={out_f32.mean().item():.4f}"
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

    print("smoke ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
