"""sm_120 Phase-5 T5.2 regression-guard: attn_sink fold semantics + diagnostics.

Pins the T5.2-stabilised behaviour of the post-hoc ``attn_sink`` fold inside
``triton_combine_partials_sm120``:

  * Default: fold ON (matches upstream sglang convention; T5.1 empirical
    "doubled accuracy" did not reproduce on a confirmation GSM8K-200 run
    and the trade-off in invalid rate is net-negative for production - see
    ``source-artifacts/.../sm120-fallback/phase-5/T5.2-ATTN-SINK-FOLD-STATUS.md``).
  * ``SGLANG_SM120_DISABLE_ATTN_SINK=1`` bypasses the fold (T5.1 toggle).
  * ``SGLANG_SM120_PROBE_ATTN_SINK=1`` logs lse / sink / scale magnitudes
    at the fold site (T5.2 sink-probe; throttled to 4 emissions per process).

This test is intentionally tolerant of arch (it does not require sm_120;
the combine function is a Python+torch implementation that runs on any
device). A separate live-instance gate covers the full kernel-level
coupling.
"""

import os
import unittest

import torch

try:
    from sglang.test.ci.ci_register import register_cuda_ci

    register_cuda_ci(est_time=10, suite="stage-b-test-1-gpu-large")
except Exception:
    pass


def _build_partials(b: int = 2, s_q: int = 1, h_q: int = 8, d_v: int = 16):
    """Build a single-split partials pair for the combine function."""
    torch.manual_seed(0)
    partials_out = torch.randn(1, b, s_q, h_q, d_v, dtype=torch.float32) * 0.1
    partials_lse = (
        torch.tensor([0.5, 1.0, 2.0, 0.0, 5.0, -1.0, 8.0, 1.5])
        .view(1, 1, 1, h_q)
        .expand(1, b, s_q, h_q)
        .contiguous()
        .float()
    )
    attn_sink = torch.tensor(
        [1.0, -0.5, 0.6, 1.6, 0.2, 0.8, -1.2, 1.0], dtype=torch.float32
    )
    return partials_out, partials_lse, attn_sink


class TestSm120AttnSinkFoldDefault(unittest.TestCase):
    """T5.2 stabilised behaviour: fold ON by default; toggle bypasses."""

    def setUp(self):
        for k in (
            "SGLANG_SM120_DISABLE_ATTN_SINK",
            "SGLANG_SM120_PROBE_ATTN_SINK",
        ):
            os.environ.pop(k, None)
        from sglang.srt.layers import sm120_diagnostic

        sm120_diagnostic._clear_cache()

    def tearDown(self):
        self.setUp()

    def test_default_on_changes_output(self):
        """With no env vars set, combine(sink) DIFFERS from combine(None).

        T5.2 default is fold-ON. The fold rescales output by
        ``exp(lse - new_lse)`` which is in (0, 1] and strictly < 1 when
        the sink is non-degenerate.
        """
        from sglang.srt.layers.attention.sm_120.triton_combine import (
            triton_combine_partials_sm120,
        )

        partials_out, partials_lse, attn_sink = _build_partials()
        out_with, lse_with = triton_combine_partials_sm120(
            partials_out, partials_lse, attn_sink=attn_sink
        )
        out_none, lse_none = triton_combine_partials_sm120(
            partials_out, partials_lse, attn_sink=None
        )
        self.assertFalse(torch.equal(out_with, out_none))
        self.assertFalse(torch.equal(lse_with, lse_none))
        # Sink fold reduces output magnitude (scale in (0, 1]).
        self.assertLess(
            out_with.float().abs().mean().item(),
            out_none.float().abs().mean().item(),
        )

    def test_disable_toggle_bypasses_fold(self):
        """With DISABLE_ATTN_SINK=1, combine(sink) == combine(None).

        The legacy T5.1 diagnostic toggle is preserved; setting it pins
        fold-OFF behaviour for explicit A/B comparison or accuracy
        regression studies.
        """
        os.environ["SGLANG_SM120_DISABLE_ATTN_SINK"] = "1"
        from sglang.srt.layers import sm120_diagnostic
        from sglang.srt.layers.attention.sm_120.triton_combine import (
            triton_combine_partials_sm120,
        )

        sm120_diagnostic._clear_cache()
        partials_out, partials_lse, attn_sink = _build_partials()
        out_with, lse_with = triton_combine_partials_sm120(
            partials_out, partials_lse, attn_sink=attn_sink
        )
        out_none, lse_none = triton_combine_partials_sm120(
            partials_out, partials_lse, attn_sink=None
        )
        self.assertTrue(torch.equal(out_with, out_none))
        self.assertTrue(torch.equal(lse_with, lse_none))


if __name__ == "__main__":
    unittest.main(verbosity=2)
