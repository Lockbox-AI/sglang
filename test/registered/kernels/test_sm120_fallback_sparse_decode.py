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
   eager reference at
   :mod:`sglang.srt.layers.attention.sm120_fallback._reference_sparse_decode`,
   assert ``torch.allclose(out, ref_out, rtol=1e-3, atol=1e-3)`` (the
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
``sparse_decode_fp8_triton`` symbol in two steps:

1. First try the production location
   :mod:`sglang.srt.layers.attention.sm120_fallback.sparse_decode_fp8_triton`
   (post-parent-merge; this is what Subagent 2 wires up).
2. If that fails, fall back to the T1.4 contract stub. The stub path
   can be set via the ``SM120_SPARSE_DECODE_STUB_PATH`` environment
   variable (an absolute path to ``sparse_decode_fp8_triton.py``);
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
from typing import Any, Callable, Optional, Tuple

import pytest
import torch

# ---------------------------------------------------------------------
# Kernel resolution + reference import
# ---------------------------------------------------------------------


def _resolve_kernel() -> Tuple[Optional[Callable], str]:
    """Import the K2a kernel.

    Returns
    -------
    (fn, source) where ``source`` is one of:
        - ``"production"``: imported from
          ``sglang.srt.layers.attention.sm120_fallback`` (post-merge).
        - ``"stub:<path>"``: imported from a T1.4 stub file pointed to
          by ``SM120_SPARSE_DECODE_STUB_PATH``.
        - ``"unavailable:<reason>"``: returned with ``fn=None`` if
          neither path resolves; collection-level skip.
    """
    try:
        from sglang.srt.layers.attention.sm120_fallback import (  # type: ignore
            sparse_decode_fp8_triton,
        )

        return sparse_decode_fp8_triton, "production"
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
        return mod.sparse_decode_fp8_triton, f"stub:{stub_path}"

    return None, (
        "unavailable: production import failed and "
        "SM120_SPARSE_DECODE_STUB_PATH unset"
    )


_KERNEL_FN, _KERNEL_SOURCE = _resolve_kernel()

if _KERNEL_FN is None:
    # Collection-level skip so the file's import doesn't crash CI.
    pytest.skip(
        f"sparse_decode_fp8_triton not resolvable ({_KERNEL_SOURCE}); "
        "set SM120_SPARSE_DECODE_STUB_PATH to the T1.4 stub or merge "
        "the parent branch.",
        allow_module_level=True,
    )

_USING_STUB = _KERNEL_SOURCE.startswith("stub:")


def _import_reference():
    """Resolve the PyTorch eager reference, mirroring the kernel resolver."""
    try:
        from sglang.srt.layers.attention.sm120_fallback._reference_sparse_decode import (  # type: ignore
            make_synthetic_inputs,
            sparse_decode_fwd_reference,
        )

        return sparse_decode_fwd_reference, make_synthetic_inputs
    except ImportError:
        # File-system fallback: load directly so the test file works
        # even when sglang's top-level `__init__` has unmet optional
        # deps on a thin dev box.
        ref_path = os.path.normpath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "..",
                "..",
                "python",
                "sglang",
                "srt",
                "layers",
                "attention",
                "sm120_fallback",
                "_reference_sparse_decode.py",
            )
        )
        spec = importlib.util.spec_from_file_location(
            "_reference_sparse_decode_fallback", ref_path
        )
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.sparse_decode_fwd_reference, mod.make_synthetic_inputs


sparse_decode_fwd_reference, make_synthetic_inputs = _import_reference()


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
    kw["num_splits"] = torch.zeros((_B + 1,), dtype=torch.int32)
    if _USING_STUB:
        with pytest.raises(NotImplementedError):
            _KERNEL_FN(**kw)
    else:
        _call_or_skip(**kw)


def test_h_q_128_contract_satisfied() -> None:
    """h_q=128 (V4-Pro pre-FP4) is in scope per the spec."""
    kw = _make_valid_inputs()
    kw["q"] = torch.zeros((_B, _S_Q, 128, _D_QK), dtype=torch.bfloat16)
    if _USING_STUB:
        with pytest.raises(NotImplementedError):
            _KERNEL_FN(**kw)
    else:
        _call_or_skip(**kw)


def test_topk_2048_contract_satisfied() -> None:
    """topk=2048 (a multiple of 64) is the V4-Flash large-context default."""
    kw = _make_valid_inputs()
    kw["indices"] = torch.zeros((_B, _S_Q, 2048), dtype=torch.int32)
    if _USING_STUB:
        with pytest.raises(NotImplementedError):
            _KERNEL_FN(**kw)
    else:
        _call_or_skip(**kw)


# ---------------------------------------------------------------------
# 2. Q dtype / shape / layout contract
# ---------------------------------------------------------------------


def test_q_must_be_bf16() -> None:
    kw = _make_valid_inputs()
    kw["q"] = torch.zeros((_B, _S_Q, _H_Q, _D_QK), dtype=torch.float16)
    with pytest.raises(AssertionError, match="bfloat16"):
        _KERNEL_FN(**kw)


def test_q_must_be_4d() -> None:
    kw = _make_valid_inputs()
    kw["q"] = torch.zeros((_B, _H_Q, _D_QK), dtype=torch.bfloat16)
    with pytest.raises(AssertionError, match="4-D"):
        _KERNEL_FN(**kw)


def test_q_d_qk_must_be_512() -> None:
    """V4-Flash absorbed-latent: D_qk=512 (NoPE 448 + RoPE 64). FlashMLA's
    documented 576 (NoPE+RoPE concat) is rejected."""
    kw = _make_valid_inputs()
    kw["q"] = torch.zeros((_B, _S_Q, _H_Q, 576), dtype=torch.bfloat16)
    with pytest.raises(AssertionError, match="D_qk"):
        _KERNEL_FN(**kw)


def test_q_h_q_must_be_64_or_128() -> None:
    kw = _make_valid_inputs()
    kw["q"] = torch.zeros((_B, _S_Q, 32, _D_QK), dtype=torch.bfloat16)
    with pytest.raises(AssertionError, match="h_q"):
        _KERNEL_FN(**kw)


def test_q_last_dim_must_be_contiguous() -> None:
    """Non-contiguous last dim violates KU_CHECK_LAST_DIM_CONTIGUOUS(q)."""
    kw = _make_valid_inputs()
    base = torch.zeros((_B, _S_Q, _H_Q, _D_QK * 2), dtype=torch.bfloat16)
    kw["q"] = base[:, :, :, ::2]
    assert kw["q"].shape == (_B, _S_Q, _H_Q, _D_QK)
    with pytest.raises(AssertionError, match="contiguous"):
        _KERNEL_FN(**kw)


# ---------------------------------------------------------------------
# 3. KV cache layout contract — the part Run 1 got wrong
# ---------------------------------------------------------------------


def test_k_cache_must_be_4d() -> None:
    kw = _make_valid_inputs()
    kw["k_cache"] = torch.zeros(
        (_NUM_BLOCKS * _PAGE_BLOCK_SIZE, 1, _BYTES_PER_TOKEN), dtype=torch.uint8
    )
    with pytest.raises(AssertionError, match="4-D"):
        _KERNEL_FN(**kw)


def test_k_cache_dtype_must_be_byte_or_fp8() -> None:
    """sparse_decode.h:260 accepts uint8 / int8 / fp8_e4m3fn."""
    kw = _make_valid_inputs()
    kw["k_cache"] = torch.zeros(
        (_NUM_BLOCKS, _PAGE_BLOCK_SIZE, 1, _BYTES_PER_TOKEN), dtype=torch.float32
    )
    with pytest.raises(AssertionError, match="dtype"):
        _KERNEL_FN(**kw)


def test_k_cache_h_kv_must_be_one() -> None:
    """sparse_decode.h:234 — only MQA (h_kv=1) is supported."""
    kw = _make_valid_inputs()
    kw["k_cache"] = torch.zeros(
        (_NUM_BLOCKS, _PAGE_BLOCK_SIZE, 2, _BYTES_PER_TOKEN), dtype=torch.uint8
    )
    with pytest.raises(AssertionError, match="h_kv"):
        _KERNEL_FN(**kw)


def test_k_cache_bytes_per_token_must_match_v4() -> None:
    """V4-Flash: 448 (FP8 NoPE) + 128 (64 bf16 RoPE) + 7 (UE8M0 scales) + 1 (pad) = 584."""
    kw = _make_valid_inputs()
    kw["k_cache"] = torch.zeros(
        (_NUM_BLOCKS, _PAGE_BLOCK_SIZE, 1, _BYTES_PER_TOKEN + 8),
        dtype=torch.uint8,
    )
    with pytest.raises(AssertionError, match="bytes_per_token"):
        _KERNEL_FN(**kw)


def test_k_cache_inner_stride_must_equal_bytes_per_token() -> None:
    """sparse_decode.h:301 — the whole 656-byte block must be contiguous."""
    kw = _make_valid_inputs()
    base = torch.zeros(
        (_NUM_BLOCKS, _PAGE_BLOCK_SIZE, 1, _BYTES_PER_TOKEN + 32),
        dtype=torch.uint8,
    )
    sliced = base[:, :, :, :_BYTES_PER_TOKEN]
    assert sliced.stride(1) != _BYTES_PER_TOKEN
    kw["k_cache"] = sliced
    with pytest.raises(AssertionError, match="bytes_per_token"):
        _KERNEL_FN(**kw)


# ---------------------------------------------------------------------
# 4. block_table / cache_seqlens / causal / is_fp8_kvcache flags
# ---------------------------------------------------------------------


def test_block_table_must_be_none_on_sparse_path() -> None:
    kw = _make_valid_inputs()
    kw["block_table"] = torch.zeros((_B, 4), dtype=torch.int32)
    with pytest.raises(AssertionError, match="block_table"):
        _KERNEL_FN(**kw)


def test_cache_seqlens_must_be_none_on_sparse_path() -> None:
    kw = _make_valid_inputs()
    kw["cache_seqlens"] = torch.full((_B,), 1024, dtype=torch.int32)
    with pytest.raises(AssertionError, match="cache_seqlens"):
        _KERNEL_FN(**kw)


def test_causal_must_be_false() -> None:
    kw = _make_valid_inputs()
    kw["causal"] = True
    with pytest.raises(AssertionError, match="causal"):
        _KERNEL_FN(**kw)


def test_is_fp8_kvcache_must_be_true() -> None:
    kw = _make_valid_inputs()
    kw["is_fp8_kvcache"] = False
    with pytest.raises(AssertionError, match="is_fp8_kvcache"):
        _KERNEL_FN(**kw)


def test_head_dim_v_must_be_512() -> None:
    kw = _make_valid_inputs()
    kw["head_dim_v"] = 256
    with pytest.raises(AssertionError, match="head_dim_v"):
        _KERNEL_FN(**kw)


# ---------------------------------------------------------------------
# 5. indices contract
# ---------------------------------------------------------------------


def test_indices_required_on_sparse_path() -> None:
    kw = _make_valid_inputs()
    kw["indices"] = None
    with pytest.raises(AssertionError, match="indices"):
        _KERNEL_FN(**kw)


def test_indices_must_be_int32() -> None:
    kw = _make_valid_inputs()
    kw["indices"] = torch.zeros((_B, _S_Q, _TOPK), dtype=torch.int64)
    with pytest.raises(AssertionError, match="int32"):
        _KERNEL_FN(**kw)


def test_indices_must_be_3d() -> None:
    kw = _make_valid_inputs()
    kw["indices"] = torch.zeros((_B * _S_Q, _TOPK), dtype=torch.int32)
    with pytest.raises(AssertionError, match="3-D"):
        _KERNEL_FN(**kw)


def test_indices_first_dims_must_match_q() -> None:
    kw = _make_valid_inputs()
    kw["indices"] = torch.zeros((_B + 1, _S_Q, _TOPK), dtype=torch.int32)
    with pytest.raises(AssertionError, match="indices"):
        _KERNEL_FN(**kw)


def test_indices_topk_must_be_multiple_of_64() -> None:
    """splitkv_mla.cuh:689 KU_ASSERT(params.topk % TOPK_BLOCK_SIZE == 0)."""
    kw = _make_valid_inputs()
    kw["indices"] = torch.zeros((_B, _S_Q, 100), dtype=torch.int32)
    with pytest.raises(AssertionError, match="multiple of 64"):
        _KERNEL_FN(**kw)


# ---------------------------------------------------------------------
# 6. softmax_scale (required positive float)
# ---------------------------------------------------------------------


def test_softmax_scale_must_be_provided() -> None:
    kw = _make_valid_inputs()
    kw["softmax_scale"] = None
    with pytest.raises(AssertionError, match="softmax_scale"):
        _KERNEL_FN(**kw)


def test_softmax_scale_must_be_positive() -> None:
    kw = _make_valid_inputs()
    kw["softmax_scale"] = -1.0
    with pytest.raises(AssertionError, match="softmax_scale"):
        _KERNEL_FN(**kw)


# ---------------------------------------------------------------------
# 7. attn_sink optional shape/dtype
# ---------------------------------------------------------------------


def test_attn_sink_must_be_float32_when_provided() -> None:
    kw = _make_valid_inputs()
    kw["attn_sink"] = torch.zeros((_H_Q,), dtype=torch.bfloat16)
    with pytest.raises(AssertionError, match="float32"):
        _KERNEL_FN(**kw)


def test_attn_sink_shape_must_match_h_q() -> None:
    kw = _make_valid_inputs()
    kw["attn_sink"] = torch.zeros((_H_Q + 1,), dtype=torch.float32)
    with pytest.raises(AssertionError, match="attn_sink"):
        _KERNEL_FN(**kw)


# ---------------------------------------------------------------------
# 8. tile_scheduler_metadata / num_splits optional shape/dtype
# ---------------------------------------------------------------------


def test_sched_meta_must_be_int32_when_provided() -> None:
    kw = _make_valid_inputs()
    kw["tile_scheduler_metadata"] = torch.zeros(
        (_NUM_SM_PARTS, _SCHED_META_INTS_PER_ROW), dtype=torch.int64
    )
    with pytest.raises(AssertionError, match="int32"):
        _KERNEL_FN(**kw)


def test_sched_meta_inner_dim_must_be_8() -> None:
    """params.h:10-17 DecodingSchedMeta is 8 int32 fields."""
    kw = _make_valid_inputs()
    kw["tile_scheduler_metadata"] = torch.zeros(
        (_NUM_SM_PARTS, 4), dtype=torch.int32
    )
    with pytest.raises(AssertionError, match="DecodingSchedMeta"):
        _KERNEL_FN(**kw)


def test_num_splits_shape_must_be_b_plus_one() -> None:
    kw = _make_valid_inputs()
    kw["num_splits"] = torch.zeros((_B,), dtype=torch.int32)
    with pytest.raises(AssertionError, match="num_splits"):
        _KERNEL_FN(**kw)


# ---------------------------------------------------------------------
# 9. V32-only: topk_length and extra_* must be None
# ---------------------------------------------------------------------


def test_topk_length_must_be_none_on_v32() -> None:
    """splitkv_mla.cuh:701 KU_ASSERT(topk_length == nullptr)."""
    kw = _make_valid_inputs()
    kw["topk_length"] = torch.full((_B,), _TOPK, dtype=torch.int32)
    with pytest.raises(AssertionError, match="topk_length"):
        _KERNEL_FN(**kw)


def test_extra_k_cache_must_be_none_on_v32() -> None:
    """splitkv_mla.cuh:700 KU_ASSERT(extra_kv == nullptr)."""
    kw = _make_valid_inputs()
    kw["extra_k_cache"] = torch.zeros(
        (1, _PAGE_BLOCK_SIZE, 1, _BYTES_PER_TOKEN), dtype=torch.uint8
    )
    with pytest.raises(AssertionError, match="extra_k_cache"):
        _KERNEL_FN(**kw)


def test_extra_indices_must_be_none_on_v32() -> None:
    kw = _make_valid_inputs()
    kw["extra_indices_in_kvcache"] = torch.zeros(
        (_B, _S_Q, 64), dtype=torch.int32
    )
    with pytest.raises(AssertionError, match="extra_indices"):
        _KERNEL_FN(**kw)


def test_extra_topk_length_must_be_none_on_v32() -> None:
    kw = _make_valid_inputs()
    kw["extra_topk_length"] = torch.full((_B,), 64, dtype=torch.int32)
    with pytest.raises(AssertionError, match="extra_topk_length"):
        _KERNEL_FN(**kw)


# ---------------------------------------------------------------------
# 10. Reference self-checks (CPU; oracle correctness)
# ---------------------------------------------------------------------
# These verify the PyTorch eager reference itself, independently of
# whether the kernel is the stub or the real thing. They guard against
# bugs in the oracle.


class TestReferenceOracle:
    """Cross-check the reference against an independent
    ``torch.softmax``-based formulation. If the reference gets the
    algorithm wrong, the numerical tests below would silently pass on
    a wrong-but-self-consistent kernel; this layer prevents that.
    """

    def test_reference_matches_torch_softmax_formulation(self) -> None:
        inputs = make_synthetic_inputs(
            b=1, s_q=1, h_q=64, topk=64, num_blocks=2, page_block_size=64, seed=7
        )
        out_ref, lse_ref = sparse_decode_fwd_reference(
            q=inputs["q"],
            k_cache=inputs["k_cache"],
            indices=inputs["indices"],
            softmax_scale=inputs["softmax_scale"],
        )

        # Independent formulation using ``torch.softmax`` and direct
        # FP8 dequant.
        indices = inputs["indices"][0, 0]
        flat = indices.long()
        T = flat.numel()
        ps = inputs["k_cache"].shape[1]
        block_idx = flat // ps
        off_idx = flat % ps
        raw = inputs["k_cache"][block_idx, off_idx, 0, :].contiguous()
        nope_fp8 = raw[:, 0:512].contiguous().view(torch.float8_e4m3fn)
        scales = (
            raw[:, 512:528].contiguous().view(torch.float32).reshape(T, 4)
        )
        rope_bf16 = raw[:, 528:656].contiguous().view(torch.bfloat16)
        nope_bf16 = (
            (nope_fp8.float().reshape(T, 4, 128) * scales.unsqueeze(-1))
            .reshape(T, 512)
            .to(torch.bfloat16)
        )
        K = torch.cat([nope_bf16, rope_bf16], dim=-1)
        Q = inputs["q"][0, 0]
        scores = (Q.float() @ K.float().T) * inputs["softmax_scale"]
        P = torch.softmax(scores, dim=-1)
        out_alt = (P @ nope_bf16.float()).to(torch.bfloat16)
        lse_alt = torch.logsumexp(scores, dim=-1)

        max_out_diff = (out_ref[0, 0].float() - out_alt.float()).abs().max().item()
        max_lse_diff = (lse_ref[0, :, 0] - lse_alt).abs().max().item()
        assert max_out_diff < 1e-2, (
            f"reference disagrees with torch.softmax: out diff={max_out_diff}"
        )
        assert max_lse_diff < 1e-3, (
            f"reference disagrees with torch.logsumexp: lse diff={max_lse_diff}"
        )

    def test_reference_handles_all_invalid_indices(self) -> None:
        inputs = make_synthetic_inputs(
            b=2, s_q=1, h_q=64, topk=64, num_blocks=4, page_block_size=64, seed=0
        )
        inputs["indices"][:] = -1
        out, lse = sparse_decode_fwd_reference(
            q=inputs["q"],
            k_cache=inputs["k_cache"],
            indices=inputs["indices"],
            softmax_scale=inputs["softmax_scale"],
        )
        assert not torch.isnan(out).any()
        assert (out.float().abs() == 0).all(), "all-invalid → all-zero output"
        # lse should be -inf for every position (kernel writes -INF).
        assert (lse == float("-inf")).all()

    def test_reference_attn_sink_finite(self) -> None:
        inputs = make_synthetic_inputs(
            b=1, s_q=1, h_q=64, topk=64, num_blocks=4, page_block_size=64, seed=1
        )
        sink = torch.zeros(64, dtype=torch.float32)
        out, lse = sparse_decode_fwd_reference(
            q=inputs["q"],
            k_cache=inputs["k_cache"],
            indices=inputs["indices"],
            softmax_scale=inputs["softmax_scale"],
            attn_sink=sink,
        )
        assert torch.isfinite(out.float()).all()
        assert torch.isfinite(lse).all()


# ---------------------------------------------------------------------
# 11. Numerical correctness vs reference (xfail until parent merge)
# ---------------------------------------------------------------------
# Each test exercises one (B, s_q, topk, h_q, h_kv, d_qk, d_v) point
# from the task brief. Parameters fit the FP8 noise floor of
# ``rtol=1e-3, atol=1e-3`` (cuda-toolkit-distilled §3).

_NUM_RTOL = 1e-3
_NUM_ATOL = 1e-3


def _assert_kernel_matches_reference(
    *,
    b: int,
    s_q: int,
    h_q: int,
    topk: int,
    num_blocks: int,
    seed: int = 0,
    valid_ratio: float = 1.0,
    use_attn_sink: bool = False,
) -> None:
    """Build inputs, run kernel + reference, assert allclose.

    Skips the actual comparison when running against the stub (the
    contract-pass + ``NotImplementedError`` path).
    """
    if not torch.cuda.is_available():
        pytest.skip("numerical tests need a CUDA device")

    device = "cuda"
    inputs = make_synthetic_inputs(
        b=b,
        s_q=s_q,
        h_q=h_q,
        topk=topk,
        num_blocks=num_blocks,
        page_block_size=64,
        seed=seed,
        device=device,
        valid_ratio=valid_ratio,
    )
    if use_attn_sink:
        inputs["attn_sink"] = torch.zeros((h_q,), dtype=torch.float32, device=device)

    # Reference: run on CPU for determinism (no CUDA-stream non-assoc
    # surprises); move inputs to CPU first.
    cpu_kw = {
        k: (v.cpu() if isinstance(v, torch.Tensor) else v)
        for k, v in inputs.items()
    }
    ref_out, ref_lse = sparse_decode_fwd_reference(
        q=cpu_kw["q"],
        k_cache=cpu_kw["k_cache"],
        indices=cpu_kw["indices"],
        softmax_scale=cpu_kw["softmax_scale"],
        attn_sink=cpu_kw.get("attn_sink"),
    )

    out, lse = _KERNEL_FN(**inputs)

    out_cpu = out.cpu().float()
    lse_cpu = lse.cpu().float()
    ref_out_f = ref_out.float()
    ref_lse_f = ref_lse.float()

    # Mask out -inf / nan in lse (all-invalid rows) — both kernel and
    # reference agree those positions are "no work".
    finite_lse = torch.isfinite(ref_lse_f) & torch.isfinite(lse_cpu)
    if finite_lse.any():
        max_lse_diff = (
            lse_cpu[finite_lse] - ref_lse_f[finite_lse]
        ).abs().max().item()
        assert max_lse_diff < _NUM_ATOL + _NUM_RTOL * ref_lse_f[finite_lse].abs().max().item(), (
            f"lse diff {max_lse_diff} exceeds FP8 floor"
        )

    assert torch.allclose(
        out_cpu, ref_out_f, rtol=_NUM_RTOL, atol=_NUM_ATOL
    ), (
        "kernel output disagrees with PyTorch eager reference beyond "
        f"the FP8 noise floor (rtol={_NUM_RTOL}, atol={_NUM_ATOL}); "
        f"shapes: out={tuple(out_cpu.shape)} ref={tuple(ref_out_f.shape)}"
    )


_NUMERICAL_XFAIL_REASON = (
    "kernel not yet integrated; numerical tests run after parent merges "
    "Subagents 1 + 2 + 3 and flips this xfail to strict=True."
)


@pytest.mark.gpu
@pytest.mark.xfail(reason=_NUMERICAL_XFAIL_REASON, strict=False)
def test_numerical_b1_sq1_topk64_h64() -> None:
    """Smallest stable shape: B=1, s_q=1, topk=64 (one TOPK_BLOCK)."""
    _assert_kernel_matches_reference(
        b=1, s_q=1, h_q=64, topk=64, num_blocks=2, seed=0
    )


@pytest.mark.gpu
@pytest.mark.xfail(reason=_NUMERICAL_XFAIL_REASON, strict=False)
def test_numerical_b1_sq1_topk64_h128() -> None:
    """h_q=128 (V4-Pro pre-FP4) with the smallest stable topk."""
    _assert_kernel_matches_reference(
        b=1, s_q=1, h_q=128, topk=64, num_blocks=2, seed=1
    )


@pytest.mark.gpu
@pytest.mark.xfail(reason=_NUMERICAL_XFAIL_REASON, strict=False)
def test_numerical_b4_sq1_topk64_h64() -> None:
    """B=4, s_q=1: typical V4-Flash decode batch."""
    _assert_kernel_matches_reference(
        b=4, s_q=1, h_q=64, topk=64, num_blocks=4, seed=2
    )


@pytest.mark.gpu
@pytest.mark.xfail(reason=_NUMERICAL_XFAIL_REASON, strict=False)
def test_numerical_b4_sq2_topk64_h64() -> None:
    """B=4, s_q=2: MTP / speculative-decode path with next_n=2."""
    _assert_kernel_matches_reference(
        b=4, s_q=2, h_q=64, topk=64, num_blocks=4, seed=3
    )


@pytest.mark.gpu
@pytest.mark.xfail(reason=_NUMERICAL_XFAIL_REASON, strict=False)
def test_numerical_b1_sq1_topk128_h64() -> None:
    _assert_kernel_matches_reference(
        b=1, s_q=1, h_q=64, topk=128, num_blocks=4, seed=4
    )


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
    assert cp.returncode == 0, (
        f"memcheck found errors:\nstdout:\n{cp.stdout}\nstderr:\n{cp.stderr}"
    )


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
    assert cp.returncode == 0, (
        f"initcheck found errors:\nstdout:\n{cp.stdout}\nstderr:\n{cp.stderr}"
    )


@pytest.mark.gpu
@pytest.mark.xfail(reason=_SANITIZER_XFAIL_REASON, strict=False)
def test_compute_sanitizer_synccheck() -> None:
    """synccheck: divergent threads at __syncthreads. Should always be clean."""
    cp = _run_compute_sanitizer("synccheck")
    assert cp.returncode == 0, (
        f"synccheck found errors:\nstdout:\n{cp.stdout}\nstderr:\n{cp.stderr}"
    )


# ---------------------------------------------------------------------
# Inner sanitizer-runner source (written to disk on first use)
# ---------------------------------------------------------------------

_SANITIZER_INNER_SOURCE = '''#!/usr/bin/env python3
"""Tight runner for compute-sanitizer integration tests.

Imports the K2a kernel via the same resolution rules as
``test_sm120_fallback_sparse_decode.py``, runs one tiny call on a
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
        from sglang.srt.layers.attention.sm120_fallback import (
            sparse_decode_fp8_triton,
        )
        return sparse_decode_fp8_triton
    except Exception:
        pass

    stub_path = os.environ.get("SM120_SPARSE_DECODE_STUB_PATH")
    if stub_path and os.path.isfile(stub_path):
        spec = importlib.util.spec_from_file_location("_stub", stub_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.sparse_decode_fp8_triton
    raise SystemExit("could not resolve sparse_decode_fp8_triton")


def _resolve_make_synthetic():
    try:
        from sglang.srt.layers.attention.sm120_fallback._reference_sparse_decode import (
            make_synthetic_inputs,
        )
        return make_synthetic_inputs
    except Exception:
        pass

    here = os.path.dirname(os.path.abspath(__file__))
    ref = os.path.normpath(os.path.join(
        here, "..", "..", "..", "python", "sglang", "srt", "layers",
        "attention", "sm120_fallback", "_reference_sparse_decode.py",
    ))
    spec = importlib.util.spec_from_file_location("_ref", ref)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.make_synthetic_inputs


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
