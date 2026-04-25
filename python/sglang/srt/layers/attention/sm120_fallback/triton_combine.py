"""sm_120 Triton fallback for FlashMLA's split-KV combine kernel.

FlashMLA's `sparse_decode_fwd` is a 2-kernel pipeline on sm_90 / sm_100:

  1. Per-SM-part decode kernel emits ``(o_accum, lse_accum)`` partials over
     contiguous ranges of `topk_blocks` — one (or more) partial per
     ``(b, s_q)`` slot, indexed by the FlashMLA scheduler's ``num_splits``
     prefix-sum.
  2. Combine kernel (``csrc/smxx/decode/combine/combine.cu``) merges the
     partials into the final ``(out, lse)`` using an online-log-sum-exp
     reduction in base-2 log space, optionally folding ``attn_sink`` into
     the merged ``global_lse`` (combine.cu:101-112).

T1.3 §8.3 / §11 item 4 originally claimed the combine kernel was portable
to sm_120 unchanged. In practice it ships only as part of the FlashMLA
package whose top-level entry point hard-asserts
``"Unsupported architecture for sparse decode fwd"`` at
``csrc/api/sparse_decode.h:380`` for arch != sm_90a / sm_100f. Because the
combine kernel is launched from inside ``sparse_attn_decode_interface``
(``sparse_decode.h:467-490``) — i.e. *after* the arch check — and is not
exposed as a separate Python entry point in the published wheel, we lose
access to it on sm_120 anyway. Hence: a Triton fallback (per
T2A.3-STATUS.md "Bridges" → "FlashMLA `combine.cu`" row, path (b)).

Initial T2A.3 ship — no split-KV
================================

Subagent 1's TileLang sparse-decode kernel (T2A.3 K2a) emits a single-shot
``(out, lse)`` per ``(b, s_q)``: no split-KV, ``num_splits = 1`` everywhere.
At ``num_splits == 1`` the combine math collapses to::

    out_bf16   = partials_out[..., 0].to(bfloat16)
    lse_f32    = partials_lse[..., 0]    # already in natural-log

so the "kernel" reduces to a pass-through with dtype coercion and a layout
transpose for ``lse`` (the upstream contract is ``[b, h_q, s_q]`` but the
partial buffer is more naturally ``[b, s_q, h_q]``; we transpose to match
``flash_mla.flash_mla_with_kvcache``'s return layout — see T1.3 §5).

The shim signature and the partials layout are deliberately authored to
match the eventual real-combine semantics, so a Phase-4 follow-up that
turns split-KV back on (Subagent 1 emits multi-split partials, Triton
combine kernel does the actual log-sum-exp reduce) drops in without
restructuring callers.

Planned post-split-KV semantics
===============================

When Subagent 1's kernel grows a real split-KV partition (one program per
``(b, s_q, sm_part)``), the partials emitted are:

  - ``partials_out``: ``float32 [num_splits, b, s_q, h_q, d_v=512]``
  - ``partials_lse``: ``float32 [num_splits, b, s_q, h_q]`` (natural-log)

with the convention that ``partials_*[i]`` is the partial output of the
``i``-th sm_part. ``partials_lse[i, ...] == -inf`` flags an unwritten
slot (the FlashMLA scheduler does not emit a fixed ``num_splits`` per
``(b, s_q)``; we follow the same "lse == -inf means skip" convention as
``combine.cu:54-58``).

The combine reduction:

  m       = max(partials_lse, dim=0)                      # [b, s_q, h_q]
  scales  = exp2((partials_lse - m) * log2(e))            # [num_splits, b, s_q, h_q]
  sum_p   = scales.sum(dim=0)                             # [b, s_q, h_q]
  out     = (partials_out * scales[..., None]).sum(0) / sum_p[..., None]
  lse     = m + log(sum_p)                                # natural-log

with optional ``attn_sink``: ``lse = lse + log1p(exp(attn_sink - lse))``
(combine.cu:101-112). The Phase-4 follow-up will Triton-jit this and add
``attn_sink`` plumbing.
"""

from __future__ import annotations

from typing import Tuple

import torch


def triton_combine_partials_sm120(
    partials_out: torch.Tensor,
    partials_lse: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Combine per-split-KV partial decode outputs into final ``(out, lse)``.

    For the initial T2A.3 ship Subagent 1's TileLang kernel emits a single
    partial (``num_splits == 1``); this implementation is therefore a
    pass-through with dtype + layout coercion. See module docstring for
    the planned post-split-KV semantics.

    Parameters
    ----------
    partials_out
        ``float32 [num_splits, b, s_q, h_q, d_v=512]`` per-split-KV partial
        attention output. ``num_splits == 1`` for the initial ship; any
        value is accepted but only ``num_splits == 1`` is tested.
        The leading axis is the sm_part / split-KV axis (matches the
        natural emission order of the FlashMLA decode partial kernel —
        T1.3 §9.1, ``o_accum_ptr``).
    partials_lse
        ``float32 [num_splits, b, s_q, h_q]`` per-split-KV partial log-sum-exp
        in natural log. Same leading-axis convention as ``partials_out``.

    Returns
    -------
    out
        ``bfloat16 [b, s_q, h_q, d_v]``. Matches the upstream
        ``flash_mla.flash_mla_with_kvcache`` return contract (T1.3 §5).
    lse
        ``float32 [b, h_q, s_q]``. Note the ``(h_q, s_q)`` transpose:
        FlashMLA's Python wrapper transposes the C++ ``[b, s_q, h_q]``
        return into ``[b, h_q, s_q]`` (``flash_mla_interface.py:173``);
        we adopt that final layout here so callers don't need to know
        which combine implementation ran.
    """
    if not isinstance(partials_out, torch.Tensor) or not isinstance(
        partials_lse, torch.Tensor
    ):
        raise TypeError("partials_out and partials_lse must be torch.Tensor")
    if partials_out.dim() != 5:
        raise ValueError(
            f"partials_out must be 5-D [num_splits, b, s_q, h_q, d_v], "
            f"got dim={partials_out.dim()} (shape={tuple(partials_out.shape)})"
        )
    if partials_lse.dim() != 4:
        raise ValueError(
            f"partials_lse must be 4-D [num_splits, b, s_q, h_q], "
            f"got dim={partials_lse.dim()} (shape={tuple(partials_lse.shape)})"
        )
    if partials_out.shape[:4] != partials_lse.shape:
        raise ValueError(
            "partials_out and partials_lse must agree on the leading "
            "[num_splits, b, s_q, h_q] axes; got "
            f"partials_out.shape[:4]={tuple(partials_out.shape[:4])}, "
            f"partials_lse.shape={tuple(partials_lse.shape)}"
        )

    num_splits = partials_out.shape[0]

    if num_splits == 1:
        # Pass-through fast path.  Subagent 1's single-shot kernel hits this
        # branch on every call for the initial T2A.3 ship.  No reduction
        # math is needed; the partial *is* the final answer.
        out_f32 = partials_out[0]                  # [b, s_q, h_q, d_v] fp32
        lse_f32 = partials_lse[0]                  # [b, s_q, h_q]      fp32
        out_bf16 = out_f32.to(torch.bfloat16)
        # Match flash_mla_interface.py:173 ([b, h_q, s_q]).
        lse_bhs = lse_f32.transpose(-1, -2).contiguous()
        return out_bf16, lse_bhs

    # ------------------------------------------------------------------
    # num_splits > 1: planned post-split-KV reduction (NOT YET WIRED).
    #
    # Subagent 1's initial kernel emits num_splits == 1, so this branch
    # is unreachable for the T2A.3 ship.  We still author a correct
    # PyTorch reference here so the integration tests (Subagent 3) can
    # exercise the API shape, and so the follow-up that turns split-KV
    # back on can replace the body with a Triton kernel without any
    # caller-visible change.
    # ------------------------------------------------------------------
    finite_mask = torch.isfinite(partials_lse)
    safe_lse = torch.where(
        finite_mask, partials_lse, torch.full_like(partials_lse, float("-inf"))
    )
    m, _ = safe_lse.max(dim=0, keepdim=True)        # [1, b, s_q, h_q]
    m_finite = torch.where(
        torch.isinf(m), torch.zeros_like(m), m
    )
    scales = torch.exp(safe_lse - m_finite)         # [num_splits, b, s_q, h_q]
    sum_p = scales.sum(dim=0)                       # [b, s_q, h_q]
    out_f32 = (partials_out * scales.unsqueeze(-1)).sum(dim=0) / sum_p.unsqueeze(
        -1
    ).clamp_min(torch.finfo(torch.float32).tiny)
    lse_f32 = m_finite.squeeze(0) + torch.log(
        sum_p.clamp_min(torch.finfo(torch.float32).tiny)
    )
    out_bf16 = out_f32.to(torch.bfloat16)
    lse_bhs = lse_f32.transpose(-1, -2).contiguous()
    return out_bf16, lse_bhs


__all__ = ["triton_combine_partials_sm120"]
