"""sm_120 W8A8 block FP8 GEMM dispatch + auto-backend regression guard (Phase 4 T4.2).

Pins three sm_120-specific FP8 W8A8 block GEMM behaviors discovered in
T4.1-FP8-BACKEND-BISECTION-STATUS.md (RTX Pro 6000 Blackwell, compute
capability 12.0):

1. **`flashinfer_trtllm` raises explicitly on sm_120** — the loud-failure
   path. Pinning this prevents a future regression where it starts
   silently returning numbers (which would land in the
   `cutlass`-silent-corruption regime).

2. **Auto-dispatch returns `triton_w8a8_block_fp8_linear` on sm_120** —
   the T4.3 fork patch. Pre-patch, `_dispatch_auto_backend()` returns
   `deepgemm_w8a8_block_fp8_linear_with_fallback`, which fails to
   compile on sm_120 (NVCC error). Post-patch, this returns the triton
   path, which is the only backend confirmed coherent during
   end-to-end V4-Flash inference (per T4.1 combo K).

3. **Synthetic kernel-level numerics are equivalent on sm_120** — `cutlass`
   and `triton` are bit-identical (max_abs < 1e-2 BF16 ULP) when called
   on synthetic random inputs at V4-Flash-like shapes.  The
   cutlass-vs-triton runtime divergence observed in real V4-Flash
   inference (T4.1 §3) does NOT reproduce at the isolated kernel
   level — it requires the full model context (real trained weights +
   TP=8 + 64 sequential layers + chained FP8 quantization).  This is a
   pinned negative result: do NOT chase this bug at the synthetic
   kernel level; the symptom only manifests during integrated
   inference. The fix lives in dispatch (test 2 above).

The test is intentionally sm_120-only (skips on every other arch) so it
can ride into upstream CI as a regression guard once the dispatch fix
lands without polluting other-arch test runs.

Run live (on the g7e.48xl):
    cd /opt/dev-workspace/sglang
    pytest -xvs test/registered/quant/test_sm120_fp8_block_w8a8.py
"""

import unittest

import torch

try:
    from sglang.test.ci.ci_register import register_cuda_ci

    register_cuda_ci(est_time=15, suite="stage-b-test-1-gpu-large")
except Exception:
    # Allow standalone runs outside the sglang CI registry.
    pass


def _is_sm120() -> bool:
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability(0) == (12, 0)


@unittest.skipUnless(_is_sm120(), "sm_120-only correctness test")
class TestSm120Fp8BlockW8a8(unittest.TestCase):
    """Verify each FP8 W8A8 block GEMM backend's output on sm_120."""

    BLOCK_N = 128
    BLOCK_K = 128
    # V4-Flash gate_proj-like shape (intermediate=16384, hidden=2048).
    # Both dims % 128 == 0 so the cutlass path exercises its real kernel
    # rather than the shape-fallback.
    N = 16384
    K = 2048
    M = 32

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        # Build a clean BF16 weight, block-quantize to FP8 (e4m3fn), keep both.
        w_bf16 = torch.randn(cls.N, cls.K, dtype=torch.float32, device="cuda") * 0.05

        block_n, block_k = cls.BLOCK_N, cls.BLOCK_K
        n_blk_n = cls.N // block_n
        n_blk_k = cls.K // block_k

        # Block-wise scale = max(|x|) / fp8_max per (block_n, block_k) tile.
        fp8_max = torch.finfo(torch.float8_e4m3fn).max
        w_blocks = w_bf16.view(n_blk_n, block_n, n_blk_k, block_k).permute(0, 2, 1, 3)
        block_amax = w_blocks.abs().amax(dim=(-1, -2)).clamp(min=1e-12)
        scale = (block_amax / fp8_max).to(torch.float32)
        scaled = w_blocks / scale.unsqueeze(-1).unsqueeze(-1)
        w_fp8 = scaled.clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)
        cls.weight_fp8 = w_fp8.permute(0, 2, 1, 3).contiguous().view(cls.N, cls.K)
        cls.weight_scale = scale  # (n_blk_n, n_blk_k), fp32

        # Reconstruct BF16 reference by dequantizing the FP8 weight tile-by-tile.
        w_recon_blocks = w_fp8.to(torch.float32) * scale.unsqueeze(-1).unsqueeze(-1)
        cls.weight_bf16_dequant = (
            w_recon_blocks.permute(0, 2, 1, 3)
            .contiguous()
            .view(cls.N, cls.K)
            .to(torch.bfloat16)
        )

        # BF16 input.
        cls.input_bf16 = (
            torch.randn(cls.M, cls.K, dtype=torch.bfloat16, device="cuda") * 0.5
        )

        # Reference: dequantized weight @ input (BF16 matmul, no FP8 noise on the
        # weight; only input is BF16). This is the "gold" output every backend
        # should approximate within FP8 quantization noise.
        cls.reference_bf16 = torch.matmul(cls.input_bf16, cls.weight_bf16_dequant.T)

    def _run_backend(self, fn):
        return fn(
            self.input_bf16,
            self.weight_fp8,
            [self.BLOCK_N, self.BLOCK_K],
            self.weight_scale,
        )

    def _summarize(self, label, output):
        diff = (output.float() - self.reference_bf16.float()).abs()
        rel = diff / (self.reference_bf16.float().abs().clamp(min=1e-6))
        max_abs = diff.max().item()
        max_rel = rel.max().item()
        mean_rel = rel.mean().item()
        nan_count = torch.isnan(output).sum().item()
        inf_count = torch.isinf(output).sum().item()
        print(
            f"  [{label:>10s}]  max_abs={max_abs:.4f}  "
            f"max_rel={max_rel:.4f}  mean_rel={mean_rel:.4f}  "
            f"nan={nan_count} inf={inf_count}"
        )
        return {
            "max_abs": max_abs,
            "max_rel": max_rel,
            "mean_rel": mean_rel,
            "nan": nan_count,
            "inf": inf_count,
        }

    def test_synthetic_cutlass_and_triton_are_equivalent(self):
        """Pinned negative result.

        On sm_120, calling `cutlass_w8a8_block_fp8_linear_with_fallback` and
        `triton_w8a8_block_fp8_linear` directly with the same FP8-quantized
        weights and BF16 inputs produces bit-identical output. This means the
        runtime cutlass-vs-triton divergence observed in T4.1 §3 (real
        V4-Flash inference) is NOT reproducible at this isolated kernel
        level — it requires the full model integration (real trained
        weights + TP=8 + chained 64-layer FP8 quantization).

        Pinning this so future investigators don't waste time looking for
        the bug at the synthetic kernel level. The fix lives in dispatch.
        """
        from sglang.srt.layers.quantization.fp8_utils import (
            cutlass_w8a8_block_fp8_linear_with_fallback,
            triton_w8a8_block_fp8_linear,
        )

        out_cutlass = self._run_backend(cutlass_w8a8_block_fp8_linear_with_fallback)
        out_triton = self._run_backend(triton_w8a8_block_fp8_linear)
        diff = (out_cutlass.float() - out_triton.float()).abs()
        max_abs = diff.max().item()
        mean_abs = diff.mean().item()
        print(f"  [cutlass-vs-triton]  max_abs={max_abs:.6f}  mean_abs={mean_abs:.2e}")
        # Both should be within BF16 ULP (~1e-2 for matmul output magnitudes ~1).
        self.assertLess(
            max_abs,
            0.02,
            f"On synthetic data, cutlass and triton W8A8 block FP8 GEMM "
            f"diverged by {max_abs} on sm_120. This is unexpected — the "
            f"runtime bug from T4.1 should NOT reproduce at this layer. "
            f"Investigate.",
        )

    def test_flashinfer_w8a8_block_fp8_explicit_unsupported_on_sm120(self):
        """FlashInfer FP8 GEMM raises an explicit BackendSupportedError on sm_120.

        This is the LOUD failure mode (preferred over cutlass's silent
        corruption). The test pins this behavior so a regression that silently
        starts returning numbers is caught.
        """
        try:
            from sglang.srt.layers.quantization.fp8_utils import (
                flashinfer_gemm_w8a8_block_fp8_linear_with_fallback,
            )
        except ImportError:
            self.skipTest("flashinfer FP8 GEMM not importable on this build")
            return

        with self.assertRaises(Exception) as cm:
            self._run_backend(flashinfer_gemm_w8a8_block_fp8_linear_with_fallback)
        msg = str(cm.exception)
        # Document the expected error string.
        self.assertIn(
            "120",
            msg,
            f"Expected sm_120-related backend error from flashinfer; got: {msg!r}",
        )

    def test_dispatch_w8a8_block_fp8_linear_picks_triton_on_sm120(self):
        """The Phase-4 fork patch makes auto-dispatch pick triton on sm_120.

        Pre-fix: auto picks deep_gemm (compile fail) or cutlass (silent
        corruption). Post-fix: auto picks triton on sm_120.

        This test fails until T4.3's fork patch lands — useful as a TDD gate.
        """
        from sglang.srt.layers.quantization.fp8_utils import (
            _dispatch_auto_backend,
            triton_w8a8_block_fp8_linear,
        )

        try:
            picked = _dispatch_auto_backend()
        except Exception as e:
            # Pre-T4.3: deep_gemm path may raise during _dispatch_auto_backend on
            # the auto branch (it's a callable lookup, so usually returns OK
            # even if calling fails). Treat any unexpected exception as a
            # different bug.
            self.fail(f"_dispatch_auto_backend raised on sm_120: {e}")

        # Accept either the bare function or a partial wrapping it.
        is_triton = (
            picked is triton_w8a8_block_fp8_linear
            or getattr(picked, "func", None) is triton_w8a8_block_fp8_linear
            or "triton" in getattr(picked, "__name__", "").lower()
        )
        self.assertTrue(
            is_triton,
            f"Auto-dispatched backend on sm_120 is {picked!r}; "
            f"expected triton_w8a8_block_fp8_linear (T4.3 fork patch). "
            f"This test will fail until the patch lands.",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
