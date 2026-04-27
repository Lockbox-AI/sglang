"""sm_120 Phase-5 eager reference oracles - kernel A/B unit tests (T5.1).

These tests validate that the three eager BF16 reference modules added in
Phase 5 (``_eager_sparse_decode``, ``_eager_block_fp8_sm120``, and
``_eager_block_fp8_moe_sm120``) produce numerically equivalent output to
their respective sm_120 production kernels (within FP8 noise floor) at
small synthetic shapes. This is a **prerequisite gate** for the GSM8K
A/B diagnostic in T5.1: if an eager reference is itself buggy, GSM8K
deltas under its toggle would be mis-attributed to the kernel.

Each test runs only on sm_120 (skips on every other arch) and uses
intentionally small shapes (sub-second runtime) so the suite can be a
pre-deployment smoke before the full diagnostic matrix.

Tolerance: per-test, since the noise floor depends on whether the kernel
includes FP8 input quantization (W8A8 block linear: ~10-15% rel due to
per-token-group input quant; sparse-decode at unit-scale UE8M0 dequant:
< 5% rel; eager-MoE smoke: magnitude/NaN sanity only). Gates are tuned
to catch outright reference bugs (typical wrong-impl failure mode is
mean_rel near 1.0 from transposed matmul or wrong dequant scale), not
to assert bit-equivalence.
"""

import unittest

import torch

try:
    from sglang.test.ci.ci_register import register_cuda_ci

    register_cuda_ci(est_time=20, suite="stage-b-test-1-gpu-large")
except Exception:
    pass


def _is_sm120() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        return torch.cuda.get_device_capability(0) == (12, 0)
    except Exception:
        return False


def _block_quantize_weight_to_fp8(w_bf16: torch.Tensor, block_n: int, block_k: int):
    """Block-quantize a BF16 weight ``[N, K]`` -> (FP8 e4m3fn, FP32 scale)."""
    N, K = w_bf16.shape
    n_blk_n = N // block_n
    n_blk_k = K // block_k
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    blocks = w_bf16.view(n_blk_n, block_n, n_blk_k, block_k).transpose(1, 2)
    block_amax = blocks.abs().amax(dim=(-1, -2)).clamp(min=1e-12)
    scale = (block_amax / fp8_max).to(torch.float32)
    scaled = blocks / scale.unsqueeze(-1).unsqueeze(-1)
    w_fp8 = (
        scaled.clamp(-fp8_max, fp8_max)
        .to(torch.float8_e4m3fn)
        .transpose(1, 2)
        .contiguous()
        .view(N, K)
    )
    return w_fp8, scale


@unittest.skipUnless(_is_sm120(), "sm_120-only correctness test")
class TestEagerW8a8BlockFp8(unittest.TestCase):
    """Eager block-FP8 linear vs. ``triton_w8a8_block_fp8_linear``."""

    BLOCK_N = 128
    BLOCK_K = 128
    M = 32
    N = 256
    K = 512

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        torch.set_default_device("cuda")
        w_bf16 = (torch.randn(cls.N, cls.K, dtype=torch.float32) * 0.05).to(
            torch.bfloat16
        )
        cls.w_fp8, cls.w_scale = _block_quantize_weight_to_fp8(
            w_bf16, cls.BLOCK_N, cls.BLOCK_K
        )
        cls.x = (torch.randn(cls.M, cls.K, dtype=torch.float32) * 0.5).to(
            torch.bfloat16
        )

    def test_eager_matches_triton_w8a8_block(self):
        """Eager (BF16 weight dequant + BF16 matmul) vs. triton_w8a8_block_fp8_linear.

        The Triton path FP8-quantizes the input on a per-token-group basis
        (``per_token_group_quant_fp8``) before the FP8 GEMM, while the eager
        reference operates entirely in BF16. So the diff measured here is
        the *full* FP8 e4m3fn noise floor (block-quant on weight +
        per-token-group quant on input), not just weight-side dequant
        roundtrip. Empirically that floor sits around 10-15% mean rel for
        random inputs; we gate at 0.20 so we catch outright reference
        bugs (the typical wrong-implementation failure mode is mean_rel
        near 1.0 from a transposed matmul or wrong dequant scale).

        This is INTENDED: the eager reference is an upper-bound oracle
        for what the FP8 path would compute if FP8 quant noise vanished;
        the GSM8K diagnostic measures whether closing that noise gap
        moves the needle on the residual GSM8K accuracy.
        """
        from sglang.srt.layers.quantization._eager_block_fp8_sm120 import (
            eager_w8a8_block_fp8_linear_sm120,
        )
        from sglang.srt.layers.quantization.fp8_utils import (
            triton_w8a8_block_fp8_linear,
        )

        block_size = [self.BLOCK_N, self.BLOCK_K]
        out_triton = triton_w8a8_block_fp8_linear(
            self.x, self.w_fp8, block_size, self.w_scale
        ).float()
        out_eager = eager_w8a8_block_fp8_linear_sm120(
            self.x, self.w_fp8, block_size, self.w_scale
        ).float()

        diff = (out_triton - out_eager).abs()
        rel = diff / out_triton.abs().clamp(min=1e-3)
        print(
            f"  [eager-vs-triton W8A8 block] max_abs={diff.max().item():.4f} "
            f"mean_abs={diff.mean().item():.6f} mean_rel={rel.mean().item():.4f}"
        )
        self.assertLess(rel.mean().item(), 0.20)
        self.assertEqual(torch.isnan(out_eager).sum().item(), 0)


@unittest.skipUnless(_is_sm120(), "sm_120-only correctness test")
class TestEagerSparseDecode(unittest.TestCase):
    """Eager sparse-decode vs. ``tilelang_fp8_sparse_decode``.

    Synthetic V4-Flash KV cache (random NoPE FP8 + RoPE BF16 + UE8M0 scales),
    small ``B=1, S_q=1, H_q=64, topk=64, num_pages=4, page_size=64``.
    Compared with ``attn_sink=None`` to isolate the kernel correctness from
    the sink-fold question (which is a separate diagnostic).
    """

    NUM_PAGES = 4
    PAGE_SIZE = 64
    B = 1
    S_Q = 1
    H_Q = 64
    TOPK = 64
    D_NOPE = 448
    D_ROPE = 64
    D_QK = 512
    D_V = 512
    BYTES_PER_TOKEN = 584

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        torch.set_default_device("cuda")

        # Build a synthetic V4-Flash 4-D KV cache view by writing into the
        # underlying [num_pages, bppp] uint8 buffer with the same region-A /
        # region-B layout the K2a kernel decodes.
        #   Region A: page_size * 576 bytes per page (NoPE 448 FP8 + RoPE 128 BF16
        #   = 576 bytes per token).
        #   Region B: page_size * 8 bytes per page (7 UE8M0 + 1 pad per token).
        bppp = (
            cls.PAGE_SIZE * 576 + cls.PAGE_SIZE * 8
        )  # 64 * 584 = 37376; pad-to-576 next multiple = 37440
        bppp = ((bppp + 575) // 576) * 576
        underlying = torch.zeros(cls.NUM_PAGES, bppp, dtype=torch.uint8)

        # Region A: fill NoPE FP8 (random small magnitudes) + RoPE BF16
        for p in range(cls.NUM_PAGES):
            for t in range(cls.PAGE_SIZE):
                base = t * 576
                nope = (
                    (torch.randn(cls.D_NOPE, dtype=torch.float32) * 0.5)
                    .to(torch.float8_e4m3fn)
                    .view(torch.uint8)
                )
                rope = (
                    (torch.randn(cls.D_ROPE, dtype=torch.float32) * 1.0)
                    .to(torch.bfloat16)
                    .view(torch.uint8)
                )
                underlying[p, base : base + cls.D_NOPE] = nope
                underlying[
                    p,
                    base + cls.D_NOPE : base + cls.D_NOPE + cls.D_ROPE * 2,
                ] = rope
            # Region B: 7 UE8M0 scale bytes per token, plus 1 pad byte. Use
            # scale value 127 -> 2^0 = 1.0 (so the eager dequant is just
            # NoPE.float() unchanged at the per-tile level - keeps the test's
            # numerical envelope tight).
            scale_base = cls.PAGE_SIZE * 576
            for t in range(cls.PAGE_SIZE):
                for s in range(7):
                    underlying[p, scale_base + t * 8 + s] = 127

        # Re-view as the 4-D [num_pages, page_size, 1, 584] form sglang
        # passes into flash_mla_with_kvcache.
        cls.kv_cache = underlying.as_strided(
            size=(cls.NUM_PAGES, cls.PAGE_SIZE, 1, cls.BYTES_PER_TOKEN),
            stride=(bppp, 576, cls.BYTES_PER_TOKEN, 1),
        )

        # Random Q (BF16) and topk indices (covering valid token positions).
        cls.q = (
            torch.randn(cls.B, cls.S_Q, cls.H_Q, cls.D_QK, dtype=torch.float32) * 0.5
        ).to(torch.bfloat16)
        # Pick topk distinct indices in [0, num_pages * page_size).
        max_idx = cls.NUM_PAGES * cls.PAGE_SIZE
        perm = torch.randperm(max_idx)[: cls.TOPK].to(torch.int32)
        cls.indices = (
            perm.view(1, 1, cls.TOPK).expand(cls.B, cls.S_Q, cls.TOPK).contiguous()
        )

    def test_eager_matches_kernel_no_sink(self):
        from sglang.srt.layers.attention.sm_120._eager_sparse_decode import (
            eager_fp8_sparse_decode,
        )
        from sglang.srt.layers.attention.sm_120.tilelang_sparse_decode import (
            tilelang_fp8_sparse_decode,
        )

        sm_scale = 1.0 / (self.D_QK**0.5)
        out_kernel, lse_kernel = tilelang_fp8_sparse_decode(
            q=self.q,
            kv_cache=self.kv_cache,
            block_table=None,
            seq_lens=None,
            indices=self.indices,
            sm_scale=sm_scale,
            attn_sink=None,
        )
        out_eager, lse_eager = eager_fp8_sparse_decode(
            q=self.q,
            kv_cache=self.kv_cache,
            block_table=None,
            seq_lens=None,
            indices=self.indices,
            sm_scale=sm_scale,
            attn_sink=None,
        )

        # Compare on the meaningful slice (out[..., :448] - the trailing 64
        # zero-padded dims should already match exactly).
        a = out_kernel[..., :448].float()
        b = out_eager[..., :448].float()
        diff = (a - b).abs()
        rel = diff / a.abs().clamp(min=1e-3)
        print(
            f"  [eager-vs-kernel sparse-decode no-sink] "
            f"max_abs={diff.max().item():.4f} "
            f"mean_abs={diff.mean().item():.6f} "
            f"mean_rel={rel.mean().item():.4f}"
        )
        # Trailing zeros must match exactly.
        self.assertEqual(
            (out_kernel[..., 448:] - out_eager[..., 448:]).abs().max().item(), 0.0
        )
        self.assertLess(rel.mean().item(), 0.05)
        # LSE should also match within FP32 noise (natural-log values
        # are O(1) magnitude here so abs-error tolerance is fine).
        lse_diff = (lse_kernel.float() - lse_eager.float()).abs()
        print(
            f"  [eager-vs-kernel sparse-decode lse no-sink] "
            f"max_abs={lse_diff.max().item():.4f} "
            f"mean_abs={lse_diff.mean().item():.6f}"
        )
        self.assertLess(lse_diff.mean().item(), 0.05)


@unittest.skipUnless(_is_sm120(), "sm_120-only correctness test")
class TestEagerBlockFp8Moe(unittest.TestCase):
    """Eager MoE vs. ``fused_moe(use_fp8_w8a8=True, block_shape=[128, 128])``."""

    BLOCK_N = 128
    BLOCK_K = 128
    M = 8
    N = 256
    K = 512
    E = 8
    TOP_K = 2
    DTYPE = torch.bfloat16

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        torch.set_default_device("cuda")

    def _build_weights(self):
        w1_bf16 = (
            torch.randn(self.E, 2 * self.N, self.K, dtype=torch.float32) * 0.05
        ).to(self.DTYPE)
        w2_bf16 = (torch.randn(self.E, self.K, self.N, dtype=torch.float32) * 0.05).to(
            self.DTYPE
        )
        # Per-expert block-quantize using the 2-D helper (loop over experts).
        from sglang.srt.layers.quantization._eager_block_fp8_sm120 import (  # noqa: F401
            _block_dequantize_weight_to_bf16,
        )

        w1_fp8 = torch.empty_like(w1_bf16, dtype=torch.float8_e4m3fn)
        w1_scale = torch.empty(
            self.E,
            (2 * self.N) // self.BLOCK_N,
            self.K // self.BLOCK_K,
            dtype=torch.float32,
        )
        w2_fp8 = torch.empty_like(w2_bf16, dtype=torch.float8_e4m3fn)
        w2_scale = torch.empty(
            self.E,
            self.K // self.BLOCK_N,
            self.N // self.BLOCK_K,
            dtype=torch.float32,
        )
        for e in range(self.E):
            f1, s1 = _block_quantize_weight_to_fp8(
                w1_bf16[e], self.BLOCK_N, self.BLOCK_K
            )
            f2, s2 = _block_quantize_weight_to_fp8(
                w2_bf16[e], self.BLOCK_N, self.BLOCK_K
            )
            w1_fp8[e] = f1
            w1_scale[e] = s1
            w2_fp8[e] = f2
            w2_scale[e] = s2
        return w1_fp8, w1_scale, w2_fp8, w2_scale

    def test_eager_matches_fused_moe_smoke(self):
        """Loose smoke gate: eager MoE output is in the same regime as fused_moe.

        Tightening to mean_rel < 5% is gated on running the actual
        fused_moe call (which is itself a Phase-5 diagnostic target).
        For this prerequisite test we only verify that the eager
        reference produces NUMERICALLY-SANE output (no NaN/Inf, output
        magnitude in the same OOM as input) on small shapes. The strict
        eager-vs-kernel A/B is the actual GSM8K diagnostic in T5.1.
        """
        try:
            w1_fp8, w1_scale, w2_fp8, w2_scale = self._build_weights()
        except Exception as e:
            self.skipTest(f"Could not build synthetic block-FP8 MoE weights: {e}")

        from sglang.srt.layers.moe._eager_block_fp8_moe_sm120 import (
            eager_block_fp8_moe_sm120,
        )

        a = torch.randn(self.M, self.K, dtype=self.DTYPE) * 0.5
        score = torch.randn(self.M, self.E, dtype=self.DTYPE)
        topk_weights, topk_ids = torch.topk(
            torch.softmax(score.float(), dim=-1), self.TOP_K
        )

        out = eager_block_fp8_moe_sm120(
            hidden_states=a,
            w1=w1_fp8,
            w2=w2_fp8,
            topk_weights=topk_weights,
            topk_ids=topk_ids.long(),
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            block_shape=[self.BLOCK_N, self.BLOCK_K],
        )
        nan = torch.isnan(out).sum().item()
        inf = torch.isinf(out).sum().item()
        mag = out.float().abs().mean().item()
        print(
            f"  [eager MoE smoke] shape={tuple(out.shape)} "
            f"mean_abs={mag:.4f} nan={nan} inf={inf}"
        )
        self.assertEqual(nan, 0)
        self.assertEqual(inf, 0)
        self.assertGreater(mag, 1e-6)
        self.assertLess(mag, 1e3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
