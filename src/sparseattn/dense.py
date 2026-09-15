"""Manual dense attention -- the correctness reference for everything else.

Deliverable 1.1: matmul + mask + softmax, written out by hand.
F.scaled_dot_product_attention is deliberately never called here.

Deliverable 1.4 lives here too: `masked_softmax` is where a fully-masked query
row turns into NaN, and where we stop it from doing so.

READING ORDER: this file first, then patterns.py, then blocksparse.py.
Full background in docs/EXPLAINER.md Parts 1, 5 and 6.
"""

from __future__ import annotations

import math
from typing import Literal, Optional

import torch

# The four ways to handle a "dead row" -- a query allowed to see nothing at all.
# Literal[...] makes a typo like policy="Zero" a type-checker error rather than a
# silent fallthrough to the default branch.  See masked_softmax for what each does.
DeadRowPolicy = Literal["zero", "self", "raise", "naive"]


def masked_softmax(
    scores: torch.Tensor,                    # (..., Nq, Nk) raw attention logits
    mask: Optional[torch.Tensor],            # broadcastable to scores; True == may attend
    policy: DeadRowPolicy = "zero",
    fallback: Optional[torch.Tensor] = None,  # where each query's OWN position sits
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
    # ---- Fast path: no mask at all -------------------------------------------
    # Used by the harness check that a full non-causal pattern reproduces plain
    # softmax(QK^T)V.  Skips every dead-row branch below, since with no mask no
    # row can be dead.
    if mask is None:
        return torch.softmax(scores, dim=-1)

    # ---- The deliberately broken path ----------------------------------------
    # This is the textbook recipe, kept as a first-class option so demo_nan.py can
    # SHOW the failure and check_dead_rows can assert it still fails.  If this
    # branch ever stopped producing nan, the demo would silently stop
    # demonstrating -- so a test pins it.
    if policy == "naive":
        return torch.softmax(scores.masked_fill(~mask, float("-inf")), dim=-1)

    # ---- Find the dead rows ---------------------------------------------------
    # mask.any(-1) is True if this query may attend to AT LEAST ONE key; negating
    # gives "this query may attend to nothing".  keepdim=True leaves the trailing
    # axis as size 1 so `dead` broadcasts cleanly against (..., Nq, Nk) below.
    # Shape: (..., Nq, 1).
    dead = ~mask.any(dim=-1, keepdim=True)

    # ---- policy="raise": refuse to paper over it ------------------------------
    # Correct default while developing: a dead row almost always means the
    # pattern is wrong, and silently handling it hides the actual bug.
    if policy == "raise" and bool(dead.any()):
        n = int(dead.sum())
        raise ValueError(
            f"{n} query row(s) have an entirely-masked attendable set. "
            "Use policy='zero' or 'self', or widen the sparsity pattern."
        )

    # ---- policy="self": revive dead rows onto their own position --------------
    if policy == "self":
        # Let a dead query attend to itself.  "Itself" is a statement about
        # positions, and the caller is the only one who knows how positions map
        # onto the last axis of `scores`.  For dense attention that axis is the
        # key sequence, so the diagonal is the identity.  For the block-sparse
        # kernel the axis is gathered key slots, whose absolute positions are
        # scattered -- there the identity is meaningless and the caller passes
        # (key_position == query_position) instead.  Defaulting to eye() here
        # was a bug: it silently assumed the dense layout.
        if fallback is None:
            nq, nk = scores.shape[-2], scores.shape[-1]
            if nq != nk:
                # Square is a necessary condition for eye() to mean "own
                # position".  Refuse rather than guess.
                raise ValueError(
                    "policy='self' on a non-square score matrix needs an explicit "
                    "`fallback` mask marking where each query's own position sits.")
            fallback = torch.eye(nq, dtype=torch.bool, device=scores.device)

        # `dead & fallback` = "the own-position slot, but only for rows that need
        # reviving".  OR-ing it into the mask opens exactly one key per dead row.
        mask = mask | (dead & fallback)

        # Recompute: any row we just revived is no longer dead.
        dead = ~mask.any(dim=-1, keepdim=True)

        # VERIFY THE FIX WORKED.  If the caller's fallback did not cover some
        # dead row, that row is still -inf everywhere and would nan below.  The
        # first version of this function assumed the fix succeeded; it did not,
        # and the failure surfaced 40 frames away.  Check, don't assume.
        if bool(dead.any()):
            raise ValueError(
                "policy='self' could not revive every dead row: the fallback mask "
                "does not cover them. Check that the query's own position is "
                "reachable in the gathered key set.")

    # ---- The actual fix -------------------------------------------------------
    # `mask | dead` broadcasts the (...,Nq,1) dead flag across the whole key axis,
    # so a dead row becomes all-True: it may attend to everything.  Its logits are
    # then finite, so softmax is well defined AND differentiable at that point.
    # This is the line that means no nan is ever constructed.
    safe_mask = mask | dead

    # masked_fill on the LOGITS (not the weights).  Chosen over an additive -inf
    # bias partly because masked_fill's backward zeroes gradients at masked
    # positions -- see docs/EXPLAINER.md Part 6 for why that distinction matters.
    scores = scores.masked_fill(~safe_mask, float("-inf"))

    # torch.softmax subtracts the row max internally, so exp() cannot overflow.
    attn = torch.softmax(scores, dim=-1)

    if policy == "zero":
        # Zero the weights of dead rows.  Their output is the zero vector and
        # their gradient contribution is exactly zero -- no nan anywhere.
        # (Note this runs AFTER the softmax, but it is not a nan repair: there is
        # no nan to repair.  It is choosing what a dead row should output.)
        attn = attn.masked_fill(dead, 0.0)

    return attn


def dense_attention(
    q: torch.Tensor,                         # (B, H, Nq, D)
    k: torch.Tensor,                         # (B, H, Nk, D)
    v: torch.Tensor,                         # (B, H, Nk, D)
    mask: Optional[torch.Tensor] = None,     # broadcastable to (B,H,Nq,Nk), True == attend
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
    # q.shape[-1] is D, the HEAD dimension -- not the model dimension.  With 4
    # heads of 64, this is 64, not 256.  `scale` is overridable so tests can pin
    # it to 1.0 and reason about exact expected values.
    scale = scale if scale is not None else 1.0 / math.sqrt(q.shape[-1])

    # THE O(N^2) LINE.  (B,H,Nq,D) @ (B,H,D,Nk) -> (B,H,Nq,Nk).  At N=8192, B=1,
    # H=8 this is 537M floats = 2.1 GB, and PyTorch will materialise ~3 of them
    # (masked_fill copies, softmax copies).  Everything in blocksparse.py exists
    # to avoid this allocation.
    #
    # Note the scale is applied to the PRODUCT, not to q beforehand.  Identical
    # maths; scaling q would touch B*H*N*D elements instead of B*H*N^2 (128x less
    # work at N=8192). Left as-is because the reference optimises for being
    # obviously correct, not fast.
    scores = (q @ k.transpose(-2, -1)) * scale

    # All masking and all NaN handling live in one function, so the sparse kernel
    # can reuse exactly the same numerics.  If this were inlined here, the two
    # paths could drift.
    attn = masked_softmax(scores, mask, policy=policy)

    # (B,H,Nq,Nk) @ (B,H,Nk,D) -> (B,H,Nq,D).  Each output row is a weighted
    # average of value vectors.  Same shape as q, which is what lets you stack
    # attention layers.
    out = attn @ v

    # Weights are returned only on request: the harness compares outputs, but
    # debugging a wrong output means looking at the weights.  Optional return
    # avoids a second code path that could drift from this one.
    return (out, attn) if return_weights else out


def causal_mask(n: int, device=None) -> torch.Tensor:
    """(N, N) bool, True on and below the diagonal.

    tril = "triangle, lower".  Row i is True in columns 0..i, so query i may see
    keys 0..i and nothing later.  Swap it for triu() and the model reads the
    future, trains to near-zero loss, and is useless.
    """
    return torch.ones(n, n, dtype=torch.bool, device=device).tril()
