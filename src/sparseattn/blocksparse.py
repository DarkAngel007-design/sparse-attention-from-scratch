"""Block-sparse attention by gathering K/V blocks (deliverable 1.2 / 1.5).

The point of this file is that the (Nq x Nk) score matrix is never built.  For
each query block we gather only the K/V blocks its pattern selects, so the
largest tensor allocated is (B, H, NQB, bs, K*bs) instead of (B, H, N, N).
With N = 8192, bs = 64 and K = 6 that is a ~21x reduction in the score tensor.

What this is *not*: a fused kernel.  The gather physically copies K and V, and
for a sliding window the copies overlap (block i-1 is gathered again by query
block i and again by i+1), so the K/V traffic goes up even as the score matrix
goes down.  That tradeoff is the honest limit of doing this in PyTorch, and it
is exactly what a Triton flash-attention kernel removes -- see WRITEUP.md.
"""

from __future__ import annotations

import math
from typing import Optional

import torch

from .dense import DeadRowPolicy, masked_softmax
from .patterns import BlockPattern, to_gather_index


def _pad_to_blocks(x: torch.Tensor, block_size: int) -> tuple[torch.Tensor, int]:
    """(B, H, N, D) -> (B, H, N_pad, D) with N_pad a multiple of block_size."""
    n = x.shape[-2]
    rem = (-n) % block_size
    if rem:
        x = torch.nn.functional.pad(x, (0, 0, 0, rem))
    return x, n


def block_sparse_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    pattern: BlockPattern,
    scale: Optional[float] = None,
    policy: DeadRowPolicy = "zero",
    q_chunk: Optional[int] = None,
) -> torch.Tensor:
    """q, k, v: (B, H, N, D) -> (B, H, N, D).

    `q_chunk` caps how many query blocks are processed at once.  Peak memory is
    linear in it, so it is the knob that decides whether N = 8192 fits.
    """
    B, H, N, D = q.shape
    if k.shape[-2] != N or v.shape[-2] != N:
        raise ValueError("this implementation assumes self-attention (Nq == Nk)")
    if pattern.n_heads not in (1, H):
        raise ValueError(f"pattern has {pattern.n_heads} heads, tensors have {H}")

    bs = pattern.block_size
    scale = scale if scale is not None else 1.0 / math.sqrt(D)
    dev = q.device

    qp, _ = _pad_to_blocks(q, bs)
    kp, _ = _pad_to_blocks(k, bs)
    vp, _ = _pad_to_blocks(v, bs)
    n_pad = qp.shape[-2]
    nqb = n_pad // bs

    idx, valid = to_gather_index(pattern)         # (Hp, NQB, K)
    idx, valid = idx.to(dev), valid.to(dev)
    Hp, _, K = idx.shape
    if Hp == 1 and H > 1:
        idx = idx.expand(H, -1, -1)
        valid = valid.expand(H, -1, -1)

    kb = kp.view(B, H, nqb, bs, D)
    vb = vp.view(B, H, nqb, bs, D)
    qb = qp.view(B, H, nqb, bs, D)

    # Absolute key position of every gathered slot: (H, NQB, K*bs).
    within = torch.arange(bs, device=dev)
    kpos = (idx.unsqueeze(-1) * bs + within).reshape(H, nqb, K * bs)
    valid_tok = valid.repeat_interleave(bs, dim=-1)                 # (H, NQB, K*bs)
    valid_tok = valid_tok & (kpos < N)          # drop the right-edge padding keys
    qpos_all = torch.arange(n_pad, device=dev).view(nqb, bs)

    hh = torch.arange(H, device=dev).view(H, 1, 1).expand(H, nqb, K)

    step = q_chunk or nqb
    outs = []
    for s in range(0, nqb, step):
        e = min(s + step, nqb)
        ci, ch = idx[:, s:e], hh[:, s:e]                            # (H, c, K)
        kg = kb[:, ch, ci]                                          # (B,H,c,K,bs,D)
        vg = vb[:, ch, ci]
        c = e - s
        kg = kg.reshape(B, H, c, K * bs, D)
        vg = vg.reshape(B, H, c, K * bs, D)

        scores = (qb[:, :, s:e] @ kg.transpose(-2, -1)) * scale      # (B,H,c,bs,K*bs)

        m = valid_tok[:, s:e].unsqueeze(0).unsqueeze(3)             # (1,H,c,1,K*bs)
        m = m.expand(1, H, c, bs, K * bs)
        qp_c = qpos_all[s:e].view(1, 1, c, bs, 1)
        kp_c = kpos[:, s:e].view(1, H, c, 1, K * bs)
        if pattern.causal:
            m = m & (kp_c <= qp_c)

        # Where does each query's own position land among the gathered slots?
        # The kernel's last axis is gathered key slots, not key positions, so
        # policy='self' cannot find the diagonal on its own.  Each absolute key
        # position appears in at most one slot (to_gather_index emits a
        # permutation), so this is unambiguous.
        fb = (kp_c == qp_c) & (kp_c < N) if policy == "self" else None

        attn = masked_softmax(scores, m, policy=policy, fallback=fb)
        outs.append(attn @ vg)                                      # (B,H,c,bs,D)

    out = torch.cat(outs, dim=2).reshape(B, H, n_pad, D)
    return out[:, :, :N]


def peak_score_elements(pattern: BlockPattern, seq_len: int, batch: int, heads: int,
                        q_chunk: Optional[int] = None) -> int:
    """Elements in the largest score tensor -- the analytic version of 1.5."""
    bs = pattern.block_size
    nqb = (seq_len + bs - 1) // bs
    _, valid = to_gather_index(pattern)
    K = valid.shape[-1]
    c = min(q_chunk or nqb, nqb)
    return batch * heads * c * bs * K * bs
