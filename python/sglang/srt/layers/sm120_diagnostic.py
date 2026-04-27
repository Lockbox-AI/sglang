"""sm_120 Phase-5 diagnostic toggle registry (T5.1).

Phase 5 of the sm_120 V4-Flash work targets the residual GSM8K accuracy gap
that survived the T4.3 cutlass->triton FP8 dispatch fix (post-T4.3:
GSM8K-200 = 0.010, gate > 0.40, residual gap = 40x). The plan
(``source-artifacts/.../sm120-fallback/phase-4/T4.3-PATCH-STATUS.md`` Section
5) lists four candidate suspects for the residual gap, none of which are
individually validated against an oracle:

  S1 - ``attn_sink`` post-hoc combine fold
       (``sm_120/triton_combine.py:159-169`` / FlashMLA combine.cu:101-112)
  S2 - Triton FP8 W8A8 block GEMM numerical drift on sm_120
  S3 - MoE Triton FP8 W8A8 block path numerical drift on sm_120
  S4 - FP8 KV cache write/read

This module exposes four env-var toggles that swap each suspect's kernel for
an eager BF16 reference, so a GSM8K-200 A/B run can attribute the residual
gap to one (or more) of them BEFORE we write any kernel-level fix code.

================================================================
Toggles (all default OFF; each gates an eager BF16 substitution)
================================================================

``SGLANG_SM120_DISABLE_ATTN_SINK`` (S1)
    Set to ``1`` to bypass the attn_sink fold in
    ``sm_120/triton_combine.py``. Output and LSE are emitted as if no sink
    contribution were present (equivalent to ``attn_sink == -inf``).

``SGLANG_SM120_EAGER_SPARSE_DECODE`` (combined S1 + S4 oracle)
    Set to ``1`` to replace ``tilelang_fp8_sparse_decode`` with an eager
    BF16 reference that dequantizes the FP8 NoPE half of the KV cache,
    gathers via the sparse indices, and runs a pure-BF16 SDPA. The
    reference applies ``attn_sink`` IN-KERNEL (via the standard
    "extra zero-value token with logit ``attn_sink[h]``" formulation),
    so its output is a correctness oracle for both K2a TileLang and the
    post-hoc combine fold.

``SGLANG_SM120_EAGER_MOE_FP8`` (S3)
    Set to ``1`` to replace ``fused_moe(use_fp8_w8a8=True, block_shape=...)``
    with an eager per-expert BF16 MoE that dequantizes block-FP8 weights
    on-the-fly. Mirrors the structure of
    ``test_sm120_fp8_block_moe.py::_eager_block_fp8_moe_reference``.

``SGLANG_SM120_EAGER_W8A8_BLOCK`` (S2)
    Set to ``1`` to replace ``triton_w8a8_block_fp8_linear`` with an eager
    BF16 reference (block-dequantize weight, BF16 matmul). This catches
    sm_120-specific Triton autotune drift on the regular W8A8 linears.

================================================================
Wiring
================================================================

Each toggle is read once per process (``functools.lru_cache``) so that the
production hot path pays only a single env-var lookup per process. Tests can
clear the cache with ``_clear_cache()``.

Live activation also requires running on sm_120; on every other arch the
toggles are silent no-ops (the dispatch sites guard ``is_sm120() and
toggle_active()``).
"""

from __future__ import annotations

import functools
import logging
import os

logger = logging.getLogger(__name__)

_LOG_PREFIX = "[sm_120 phase-5 diagnostic]"


def _truthy(s: str | None) -> bool:
    return s is not None and s.strip().lower() in {"1", "true", "yes", "on"}


@functools.lru_cache(maxsize=1)
def disable_attn_sink() -> bool:
    """``True`` iff ``SGLANG_SM120_DISABLE_ATTN_SINK`` is set (S1 toggle)."""
    val = _truthy(os.environ.get("SGLANG_SM120_DISABLE_ATTN_SINK"))
    if val:
        logger.warning(
            "%s SGLANG_SM120_DISABLE_ATTN_SINK=1 -> attn_sink fold BYPASSED in "
            "triton_combine_partials_sm120 (combine.cu:101-112 fold treated as "
            "no-op; equivalent to attn_sink == -inf).",
            _LOG_PREFIX,
        )
    return val


@functools.lru_cache(maxsize=1)
def eager_sparse_decode() -> bool:
    """``True`` iff ``SGLANG_SM120_EAGER_SPARSE_DECODE`` is set (S1+S4 oracle)."""
    val = _truthy(os.environ.get("SGLANG_SM120_EAGER_SPARSE_DECODE"))
    if val:
        logger.warning(
            "%s SGLANG_SM120_EAGER_SPARSE_DECODE=1 -> tilelang_fp8_sparse_decode "
            "replaced with eager BF16 reference (dequant FP8 NoPE -> BF16 SDPA "
            "with in-kernel attn_sink fold). Slower; use for diagnostics only.",
            _LOG_PREFIX,
        )
    return val


@functools.lru_cache(maxsize=1)
def eager_moe_fp8() -> bool:
    """``True`` iff ``SGLANG_SM120_EAGER_MOE_FP8`` is set (S3 toggle)."""
    val = _truthy(os.environ.get("SGLANG_SM120_EAGER_MOE_FP8"))
    if val:
        logger.warning(
            "%s SGLANG_SM120_EAGER_MOE_FP8=1 -> fused_moe(use_fp8_w8a8=True, "
            "block_shape=...) replaced with eager per-expert BF16 MoE. Slower; "
            "use for diagnostics only.",
            _LOG_PREFIX,
        )
    return val


@functools.lru_cache(maxsize=1)
def eager_w8a8_block() -> bool:
    """``True`` iff ``SGLANG_SM120_EAGER_W8A8_BLOCK`` is set (S2 toggle)."""
    val = _truthy(os.environ.get("SGLANG_SM120_EAGER_W8A8_BLOCK"))
    if val:
        logger.warning(
            "%s SGLANG_SM120_EAGER_W8A8_BLOCK=1 -> triton_w8a8_block_fp8_linear "
            "replaced with eager BF16 reference (block-dequant weight, BF16 "
            "matmul). Slower; use for diagnostics only.",
            _LOG_PREFIX,
        )
    return val


def any_active() -> bool:
    """Cheap "is anything on?" probe for boot-time logging."""
    return (
        disable_attn_sink()
        or eager_sparse_decode()
        or eager_moe_fp8()
        or eager_w8a8_block()
    )


def _clear_cache() -> None:
    """Test hook: clear all four cached toggle reads after env mutation."""
    disable_attn_sink.cache_clear()
    eager_sparse_decode.cache_clear()
    eager_moe_fp8.cache_clear()
    eager_w8a8_block.cache_clear()


__all__ = [
    "disable_attn_sink",
    "eager_sparse_decode",
    "eager_moe_fp8",
    "eager_w8a8_block",
    "any_active",
]
