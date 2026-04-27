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

import logging
from typing import Tuple

import torch

logger = logging.getLogger(__name__)


_PROBE_COMBINE_CALLS = 0
_PROBE_COMBINE_LIMIT = 4


def _probe_log_combine_fold(
    *,
    lse_f32: torch.Tensor,
    sink: torch.Tensor,
    new_lse: torch.Tensor,
    scale: torch.Tensor,
) -> None:
    """Log fold-site magnitudes for the first few combine calls.

    Phase-5 / T5.2 diagnostic. Throttled per process via
    ``_PROBE_COMBINE_LIMIT``. Output: per-head stats for ``lse`` (the
    LSE returned by K2a, which is ``log(sum exp(qk * sm_scale))``),
    ``sink`` (the trained per-head sink logit), the fold's LSE delta,
    and the fold's output scaling factor. Used to reconcile the
    apparent contradiction between the small mathematical fold
    contribution and the empirical 2x accuracy lift when the fold is
    disabled (T5.1).
    """
    global _PROBE_COMBINE_CALLS
    if _PROBE_COMBINE_CALLS >= _PROBE_COMBINE_LIMIT:
        return
    _PROBE_COMBINE_CALLS += 1
    try:
        lse_flat = lse_f32.detach().flatten()
        sink_flat = sink.detach().flatten()
        scale_flat = scale.detach().flatten()
        nl_flat = new_lse.detach().flatten()
        delta_flat = (nl_flat - lse_flat).flatten()
        logger.warning(
            "[sm_120 phase-5 T5.2 fold-probe %d/%d] "
            "lse: shape=%s mean=%.4f abs_mean=%.4f min=%.4f max=%.4f | "
            "sink: shape=%s abs_mean=%.4f min=%.4f max=%.4f | "
            "delta_lse: abs_mean=%.6f max=%.6f | "
            "scale: abs_mean=%.6f min=%.6f max=%.6f",
            _PROBE_COMBINE_CALLS,
            _PROBE_COMBINE_LIMIT,
            tuple(lse_f32.shape),
            float(lse_flat.mean().item()),
            float(lse_flat.abs().mean().item()),
            float(lse_flat.min().item()),
            float(lse_flat.max().item()),
            tuple(sink.shape),
            float(sink_flat.abs().mean().item()),
            float(sink_flat.min().item()),
            float(sink_flat.max().item()),
            float(delta_flat.abs().mean().item()),
            float(delta_flat.abs().max().item()),
            float(scale_flat.abs().mean().item()),
            float(scale_flat.min().item()),
            float(scale_flat.max().item()),
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("[sm_120 phase-5 T5.2 fold-probe] log failed: %r", e)


def triton_combine_partials_sm120(
    partials_out: torch.Tensor,
    partials_lse: torch.Tensor,
    attn_sink: torch.Tensor = None,
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

    # Multi-split log-sum-exp reduction (matches FlashMLA combine.cu).
    # We always run this path now; for num_splits == 1 it reduces to a
    # straight pass-through (sum_p == 1, m_finite == lse), which then
    # composes cleanly with the optional attn_sink fold below.
    finite_mask = torch.isfinite(partials_lse)
    safe_lse = torch.where(
        finite_mask, partials_lse, torch.full_like(partials_lse, float("-inf"))
    )
    m, _ = safe_lse.max(dim=0, keepdim=True)  # [1, b, s_q, h_q]
    m_finite = torch.where(torch.isinf(m), torch.zeros_like(m), m)
    scales = torch.exp(safe_lse - m_finite)  # [num_splits, b, s_q, h_q]
    sum_p = scales.sum(dim=0)  # [b, s_q, h_q]
    out_f32 = (partials_out * scales.unsqueeze(-1)).sum(dim=0) / sum_p.unsqueeze(
        -1
    ).clamp_min(torch.finfo(torch.float32).tiny)
    lse_f32 = m_finite.squeeze(0) + torch.log(
        sum_p.clamp_min(torch.finfo(torch.float32).tiny)
    )

    # Fold attn_sink (per-head log-domain bias) — matches FlashMLA
    # combine.cu:101-112. The sink is a virtual "extra token" with
    # log-weight attn_sink[h] and zero value contribution; folding it in
    # rescales the output by exp(lse - new_lse) and updates the lse.
    #
    # Phase-5 / T5.1 diagnostic: ``SGLANG_SM120_DISABLE_ATTN_SINK=1``
    # bypasses the fold here. Used to A/B the sink contribution to the
    # post-T4.3 GSM8K residual gap; default OFF in production.
    from sglang.srt.layers.sm120_diagnostic import disable_attn_sink, probe_attn_sink

    if attn_sink is not None and not disable_attn_sink():
        sink = attn_sink.to(torch.float32)
        # Broadcast sink [h_q] over [b, s_q, h_q]
        new_lse = lse_f32 + torch.nn.functional.softplus(sink - lse_f32)
        scale = torch.exp(lse_f32 - new_lse).unsqueeze(-1)  # [b, s_q, h_q, 1]

        # Phase-5 / T5.2 fold-site probe. When SGLANG_SM120_PROBE_ATTN_SINK=1,
        # log lse / sink / scale magnitudes for the first few combine calls
        # so we can see whether the fold's mathematical contribution actually
        # matches the empirical 2x accuracy lift observed when the fold is
        # disabled.
        if probe_attn_sink():
            _probe_log_combine_fold(
                lse_f32=lse_f32, sink=sink, new_lse=new_lse, scale=scale
            )

        out_f32 = out_f32 * scale
        lse_f32 = new_lse

    out_bf16 = out_f32.to(torch.bfloat16)
    lse_bhs = lse_f32.transpose(-1, -2).contiguous()
    return out_bf16, lse_bhs


__all__ = ["triton_combine_partials_sm120"]
