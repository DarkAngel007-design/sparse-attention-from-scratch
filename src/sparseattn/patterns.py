"""Block-level sparsity patterns (deliverable 1.2).

Everything is expressed as a *block mask* of shape (H, NQB, NKB), where entry
[h, i, j] says "for head h, query block i is allowed to look at key block j".
Working at block granularity rather than token granularity is the whole trick:
it is what lets the kernel gather contiguous slabs of K/V instead of touching
the N x N score matrix.

Two consumers:
  * `to_dense_mask`   -- expands to a token-level (H, Nq, Nk) bool mask, used to
                         drive the dense reference in the correctness harness.
  * `to_gather_index` -- compacts to (H, NQB, K) block indices + a validity mask,
                         used by the block-sparse kernel.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class BlockPattern:
    """A sparsity pattern, already resolved to a concrete block mask."""

    name: str
    block_mask: torch.Tensor  # (H, NQB, NKB) bool
    block_size: int
    causal: bool

    @property
    def n_heads(self) -> int:
        return self.block_mask.shape[0]

    @property
    def density(self) -> float:
        """Fraction of blocks kept.  Not the same as token-level density once
        causal masking trims the diagonal blocks -- see `token_density`."""
        return float(self.block_mask.float().mean())

    def token_density(self, seq_len: int) -> float:
        m = to_dense_mask(self, seq_len, seq_len)
        return float(m.float().mean())


def _empty(n_heads: int, nqb: int, nkb: int, device) -> torch.Tensor:
    return torch.zeros(n_heads, nqb, nkb, dtype=torch.bool, device=device)


def sliding_window(
    seq_len: int,
    block_size: int,
    window_blocks: int = 1,
    causal: bool = True,
    n_heads: int = 1,
    include_self_block: bool = True,
    device=None,
) -> BlockPattern:
    """Local attention: query block i sees blocks [i - w, i + w].

    `include_self_block=False` drops the diagonal block.  That is not a silly
    option -- it is the shape of a real off-by-one, and under causal masking it
    leaves query block 0 with nothing to attend to at all.  scripts/demo_nan.py
    uses it to produce the NaN of deliverable 1.4 without hand-crafting a mask.
    """
    nqb = nkb = (seq_len + block_size - 1) // block_size
    m = _empty(n_heads, nqb, nkb, device)
    i = torch.arange(nqb, device=device)[:, None]
    j = torch.arange(nkb, device=device)[None, :]
    local = (j - i).abs() <= window_blocks
    if not include_self_block:
        local &= i != j
    if causal:
        local &= j <= i
    m[:] = local
    return BlockPattern("sliding_window", m, block_size, causal)


def bigbird(
    seq_len: int,
    block_size: int,
    window_blocks: int = 1,
    n_global: int = 1,
    n_random: int = 2,
    causal: bool = True,
    n_heads: int = 1,
    per_head_random: bool = True,
    seed: int = 0,
    device=None,
) -> BlockPattern:
    """BigBird-style: local window + global blocks + random blocks.

    Global blocks are the first `n_global` blocks and are wired both ways: every
    query block may read them, and they may read every key block.  That
    two-way wiring is what makes global tokens disproportionately important --
    they are the only path by which information crosses the sequence in a
    bounded number of hops.

    Random blocks are drawn once per (head, query block) with a fixed seed, so
    the pattern is reproducible across runs and across the reference and kernel
    paths.  Under `causal=True` the draw is restricted to j <= i; a random block
    in the future would be entirely masked out later and would just waste a
    gather slot.
    """
    nqb = nkb = (seq_len + block_size - 1) // block_size
    m = _empty(n_heads, nqb, nkb, device)
    i = torch.arange(nqb, device=device)[:, None]
    j = torch.arange(nkb, device=device)[None, :]

    base = (j - i).abs() <= window_blocks
    if n_global > 0:
        base |= j < n_global                      # everyone reads the global prefix
        base |= i < n_global                      # the global prefix reads everyone
    if causal:
        base &= j <= i
    m[:] = base

    if n_random > 0:
        g = torch.Generator(device="cpu").manual_seed(seed)
        heads = n_heads if per_head_random else 1
        for h in range(heads):
            for qi in range(nqb):
                hi = nkb if not causal else qi + 1  # candidates j in [0, hi)
                taken = m[h, qi, :hi]
                free = (~taken).nonzero(as_tuple=True)[0]
                if free.numel() == 0:
                    continue
                k = min(n_random, free.numel())
                pick = free[torch.randperm(free.numel(), generator=g)[:k].to(free.device)]
                m[h, qi, pick] = True
        if not per_head_random:
            m[:] = m[0:1]

    return BlockPattern("bigbird", m, block_size, causal)


def dilated(
    seq_len: int,
    block_size: int,
    n_taps: int = 4,
    causal: bool = True,
    n_heads: int = 1,
    device=None,
) -> BlockPattern:
    """Stretch pattern: exponentially dilated lookback (i-1, i-2, i-4, i-8, ...).

    Logarithmic reach per layer at constant cost per query, but the coverage is
    ragged -- most key blocks are never reachable in one hop.
    """
    nqb = nkb = (seq_len + block_size - 1) // block_size
    m = _empty(n_heads, nqb, nkb, device)
    i = torch.arange(nqb, device=device)[:, None]
    m[:] = torch.eye(nqb, dtype=torch.bool, device=device)
    for t in range(n_taps):
        off = 2 ** t
        sel = (i - off) == torch.arange(nkb, device=device)[None, :]
        m |= sel
    if causal:
        m &= (torch.arange(nkb, device=device)[None, :] <= i)
    return BlockPattern("dilated", m, block_size, causal)


def dense_pattern(seq_len: int, block_size: int, causal: bool = True,
                  n_heads: int = 1, device=None) -> BlockPattern:
    """Every block attends to every (allowed) block -- the sparsity-free control.

    Useful as a sanity check: the block-sparse kernel run on this pattern must
    reproduce dense attention exactly.
    """
    nqb = nkb = (seq_len + block_size - 1) // block_size
    m = torch.ones(n_heads, nqb, nkb, dtype=torch.bool, device=device)
    if causal:
        i = torch.arange(nqb, device=device)[:, None]
        j = torch.arange(nkb, device=device)[None, :]
        m &= (j <= i)
    return BlockPattern("dense", m, block_size, causal)


def to_dense_mask(
    pattern: BlockPattern,
    seq_len_q: int,
    seq_len_k: int,
    device=None,
) -> torch.Tensor:
    """Expand a block mask to a token mask (H, Nq, Nk), True == attendable.

    Two things happen here, and conflating them is a common bug:
      1. the block mask is repeated out to token granularity, and
      2. *within* a kept diagonal block, causal masking still applies per token.
    A pattern that keeps block (i, i) does not mean every token in block i sees
    every token in block i.
    """
    bs = pattern.block_size
    bm = pattern.block_mask
    if device is not None:
        bm = bm.to(device)
    dev = bm.device

    qb = torch.arange(seq_len_q, device=dev) // bs
    kb = torch.arange(seq_len_k, device=dev) // bs
    mask = bm[:, qb][:, :, kb]  # (H, Nq, Nk)

    if pattern.causal:
        qpos = torch.arange(seq_len_q, device=dev)[:, None]
        kpos = torch.arange(seq_len_k, device=dev)[None, :]
        mask = mask & (kpos <= qpos)
    return mask


def to_gather_index(pattern: BlockPattern):
    """Compact a block mask into (idx, valid) for the gather kernel.

    idx   : (H, NQB, K) long  -- key-block indices to gather
    valid : (H, NQB, K) bool  -- which of those slots are real

    K is the largest number of key blocks any query block selects.  Rows that
    select fewer are padded, and the padding *must* carry valid=False.  Padding
    with a repeated real index and no validity mask is the bug this signature
    exists to prevent: the duplicated block would enter the softmax twice, its
    weight would be counted twice, and the output would be wrong in a way that
    is still finite, still smooth, and still trains -- just to the wrong answer.
    """
    bm = pattern.block_mask                       # (H, NQB, NKB)
    counts = bm.sum(dim=-1)                       # (H, NQB)
    K = int(counts.max())
    if K == 0:
        raise ValueError("pattern selects no blocks at all")

    H, NQB, NKB = bm.shape
    order = torch.argsort(bm.int(), dim=-1, descending=True, stable=True)
    idx = order[..., :K].contiguous()             # selected indices first
    rank = torch.arange(K, device=bm.device).view(1, 1, K)
    valid = rank < counts.unsqueeze(-1)
    return idx, valid
