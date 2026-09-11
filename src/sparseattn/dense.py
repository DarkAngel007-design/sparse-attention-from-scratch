"""Manual dense attention -- the correctness reference for everything else.

Deliverable 1.1: matmul + mask + softmax, written out by hand.
F.scaled_dot_product_attention is deliberately never called here.

Deliverable 1.4 lives here too: `masked_softmax` is where a fully-masked query
row turns into NaN, and where we stop it from doing so.
"""

from __future__ import annotations

import math
from typing import Literal, Optional

import torch

DeadRowPolicy = Literal["zero", "self", "raise", "naive"]


def masked_softmax(
    scores: torch.Tensor,
    mask: Optional[torch.Tensor],
    policy: DeadRowPolicy = "zero",
) -> torch.Tensor:
    """Softmax over the last dim, honouring `mask` (True == attendable).

    The NaN problem (deliverable 1.4)
    --------------------------------
    The textbook recipe is `scores.masked_fill(~mask, -inf)` then softmax.
    If an entire row of `mask` is False, every logit in that row is -inf, so
    softmax computes `exp(-inf - max)` where `max` is itself -inf.  That is
    `exp(-inf + inf)` = `exp(nan)` = nan.  The row is nan, and because nan
    poisons every arithmetic op it touches, the whole output tensor follows.

    Fixing this *after* the softmax (`torch.nan_to_num`) repairs the forward
    pass and leaves the backward pass broken: the gradient of softmax is still
    evaluated at the nan point, so nan flows back into Q and K and the model
    silently stops training.  The fix has to happen before the softmax.

    Strategy here: find the dead rows, let them attend to everything (which
    makes their logits finite, so softmax is well defined and differentiable),
    then multiply their output weights by zero.  Forward is exact, backward is
    clean, and no nan is ever constructed.
    """
    if mask is None:
        return torch.softmax(scores, dim=-1)

    if policy == "naive":
        # Deliberately broken: kept so scripts/demo_nan.py can show the failure.
        return torch.softmax(scores.masked_fill(~mask, float("-inf")), dim=-1)

    # (..., Nq, 1) -- True where this query has nothing it is allowed to see.
    dead = ~mask.any(dim=-1, keepdim=True)

    if policy == "raise" and bool(dead.any()):
        n = int(dead.sum())
        raise ValueError(
            f"{n} query row(s) have an entirely-masked attendable set. "
            "Use policy='zero' or 'self', or widen the sparsity pattern."
        )

    if policy == "self":
        # Let a dead query attend to itself.  Only meaningful when the query and
        # key sequences are the same length and aligned (self-attention).
        nq, nk = scores.shape[-2], scores.shape[-1]
        if nq != nk:
            raise ValueError("policy='self' needs square attention (Nq == Nk).")
        eye = torch.eye(nq, dtype=torch.bool, device=scores.device)
        mask = mask | (dead & eye)
        dead = ~mask.any(dim=-1, keepdim=True)  # now empty

    # Dead rows attend to everything -> finite logits -> finite softmax.
    safe_mask = mask | dead
    scores = scores.masked_fill(~safe_mask, float("-inf"))
    attn = torch.softmax(scores, dim=-1)

    if policy == "zero":
        # Zero the weights of dead rows.  Their output is the zero vector and
        # their gradient contribution is exactly zero -- no nan anywhere.
        attn = attn.masked_fill(dead, 0.0)

    return attn


def dense_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    policy: DeadRowPolicy = "zero",
    return_weights: bool = False,
):
    """Dense scaled dot-product attention, written out by hand (1.1).

    q, k, v : (B, H, N, D)
    mask    : broadcastable to (B, H, Nq, Nk), True == attendable
    returns : (B, H, Nq, D)   [, weights (B, H, Nq, Nk) if return_weights]

    The 1/sqrt(D) scale is not cosmetic.  q.k is a sum of D products of roughly
    unit-variance terms, so its variance grows with D; without the scale the
    logits spread out as D grows, softmax saturates onto one key, and the
    gradient through it goes to zero.  Dividing by sqrt(D) holds the logit
    variance at ~1 regardless of head dimension.
    """
    scale = scale if scale is not None else 1.0 / math.sqrt(q.shape[-1])
    scores = (q @ k.transpose(-2, -1)) * scale  # (B, H, Nq, Nk) -- the O(N^2) tensor
    attn = masked_softmax(scores, mask, policy=policy)
    out = attn @ v
    return (out, attn) if return_weights else out


def causal_mask(n: int, device=None) -> torch.Tensor:
    """(N, N) bool, True on and below the diagonal."""
    return torch.ones(n, n, dtype=torch.bool, device=device).tril()
