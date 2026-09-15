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

One object, two consumers, is the load-bearing design decision of this repo: the
thing being tested and the thing testing it come from the same source, so the
harness cannot drift from the kernel.  See docs/EXPLAINER.md Part 3.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class BlockPattern:
    """A sparsity pattern, already resolved to a concrete block mask.

    frozen=True makes it immutable.  A pattern is shared between the reference
    path and the kernel path; if one could mutate it, they would silently be
    comparing two different things.
    """

    name: str
    block_mask: torch.Tensor  # (H, NQB, NKB) bool -- may query block i read key block j
    block_size: int           # tokens per block
    causal: bool              # whether per-token causal masking is applied on top

    @property
    def n_heads(self) -> int:
        # 1 means "same pattern for every head"; the kernel broadcasts it.
        return self.block_mask.shape[0]

    @property
    def density(self) -> float:
        """Fraction of blocks kept.  Not the same as token-level density once
        causal masking trims the diagonal blocks -- see `token_density`."""
        return float(self.block_mask.float().mean())

    def token_density(self, seq_len: int) -> float:
        # The honest number.  Quoting `density` as if it were this OVERSTATES
        # your sparsity, because a kept diagonal block is only half-used under
        # causality.
        m = to_dense_mask(self, seq_len, seq_len)
        return float(m.float().mean())


def _empty(n_heads: int, nqb: int, nkb: int, device) -> torch.Tensor:
    return torch.zeros(n_heads, nqb, nkb, dtype=torch.bool, device=device)


def sliding_window(
    seq_len: int,
    block_size: int,
    window_blocks: int = 1,        # w: how many blocks either side
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

    With w=1 and causal, each query block selects exactly {i-1, i} => K=2,
    CONSTANT IN N.  That constant is why memory is O(N) and not O(N^2).
    """
    # ceil division: the last block may be partially filled, and the kernel pads.
    nqb = nkb = (seq_len + block_size - 1) // block_size
    m = _empty(n_heads, nqb, nkb, device)

    # Build an (NQB, NKB) grid of block-index differences using broadcasting:
    # i is a column vector (NQB,1), j is a row vector (1,NKB), so (j - i) is the
    # full (NQB, NKB) matrix of offsets.  This is the standard idiom -- see
    # docs/EXPLAINER.md Part 0 on `None` indexing.
    i = torch.arange(nqb, device=device)[:, None]
    j = torch.arange(nkb, device=device)[None, :]

    local = (j - i).abs() <= window_blocks      # |offset| <= w -> within the band
    if not include_self_block:
        local &= i != j                         # drop the diagonal (the bug shape)
    if causal:
        local &= j <= i                         # no looking forward
    m[:] = local                                # same pattern for every head
    return BlockPattern("sliding_window", m, block_size, causal)


def bigbird(
    seq_len: int,
    block_size: int,
    window_blocks: int = 1,
    n_global: int = 1,             # how many leading blocks act as hubs
    n_random: int = 2,             # extra random blocks per (head, query block)
    causal: bool = True,
    n_heads: int = 1,
    per_head_random: bool = True,  # independent random draw per head (stretch goal)
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

    base = (j - i).abs() <= window_blocks          # 1. LOCAL: the sliding window
    if n_global > 0:
        # 2. GLOBAL: the two lines that make a hub.  Both directions are needed.
        # With only the first, a global block is readable but cannot aggregate;
        # with only the second it aggregates but nobody can read it.  Together
        # they put every token within 2 hops of every other, regardless of N.
        base |= j < n_global                      # everyone reads the global prefix
        base |= i < n_global                      # the global prefix reads everyone
    if causal:
        base &= j <= i
    m[:] = base

    if n_random > 0:
        # 3. RANDOM: small-world edges.  SEEDED, because the reference path and
        # the kernel path must see the IDENTICAL pattern or the correctness
        # harness is meaningless.  Two independent draws would guarantee a
        # mismatch.  device="cpu" keeps the draw reproducible across backends.
        g = torch.Generator(device="cpu").manual_seed(seed)
        heads = n_heads if per_head_random else 1
        # A Python loop, O(H * NQB).  Runs ONCE at pattern construction, never in
        # a forward pass, so it never appears in the benchmark's timed region.
        for h in range(heads):
            for qi in range(nqb):
                # Candidate range: under causality only blocks 0..qi are legal.
                # Drawing from the future would waste a gather slot on something
                # the causal mask deletes anyway -- and would inflate K for free.
                hi = nkb if not causal else qi + 1
                taken = m[h, qi, :hi]
                free = (~taken).nonzero(as_tuple=True)[0]   # blocks not already in
                if free.numel() == 0:
                    continue                                # nothing left to add
                k = min(n_random, free.numel())             # may be short near i=0
                pick = free[torch.randperm(free.numel(), generator=g)[:k].to(free.device)]
                m[h, qi, pick] = True
        if not per_head_random:
            m[:] = m[0:1]                          # copy head 0's draw to all heads

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
    ragged -- most key blocks are never reachable in one hop, and WHICH ones are
    reachable depends on index arithmetic rather than on relevance.  Good for
    extending span, bad for retrieving a specific fact.
    """
    nqb = nkb = (seq_len + block_size - 1) // block_size
    m = _empty(n_heads, nqb, nkb, device)
    i = torch.arange(nqb, device=device)[:, None]
    m[:] = torch.eye(nqb, dtype=torch.bool, device=device)   # always keep own block
    for t in range(n_taps):
        off = 2 ** t                                          # 1, 2, 4, 8, ...
        sel = (i - off) == torch.arange(nkb, device=device)[None, :]
        m |= sel
    if causal:
        m &= (torch.arange(nkb, device=device)[None, :] <= i)
    return BlockPattern("dilated", m, block_size, causal)


def dense_pattern(seq_len: int, block_size: int, causal: bool = True,
                  n_heads: int = 1, device=None) -> BlockPattern:
    """Every block attends to every (allowed) block -- the sparsity-free control.

    Useful as a sanity check: the block-sparse kernel run on this pattern must
    reproduce dense attention exactly.  A systematically wrong gather would still
    agree with a mask built the same wrong way, but it could not agree with plain
    softmax(QK^T)V -- which is what check_dense_pattern_is_plain_dense tests.
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

    Drop step 2 and your reference silently lets every query see the rest of its
    own block -- i.e. the future.  The model trains, scores suspiciously well,
    and is useless.
    """
    bs = pattern.block_size
    bm = pattern.block_mask
    if device is not None:
        bm = bm.to(device)
    dev = bm.device

    # Which block does each TOKEN belong to?  Integer division: tokens 0..bs-1
    # are block 0, bs..2bs-1 are block 1, and so on.
    qb = torch.arange(seq_len_q, device=dev) // bs      # (Nq,) values in [0, NQB)
    kb = torch.arange(seq_len_k, device=dev) // bs      # (Nk,) values in [0, NKB)

    # Advanced indexing with REPETITION: bm[:, qb] picks block-row qb[t] for each
    # token t, so block 0's row appears bs times consecutively.  Then [:, :, kb]
    # does the same to columns.  Net effect: "zoom" the (NQB,NKB) grid up to
    # (Nq, Nk) without any arithmetic.
    mask = bm[:, qb][:, :, kb]  # (H, Nq, Nk)

    # STEP 2 -- the per-token causality that block granularity cannot express.
    if pattern.causal:
        qpos = torch.arange(seq_len_q, device=dev)[:, None]   # (Nq,1)
        kpos = torch.arange(seq_len_k, device=dev)[None, :]   # (1,Nk)
        mask = mask & (kpos <= qpos)                          # broadcast -> (H,Nq,Nk)
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
    counts = bm.sum(dim=-1)                       # (H, NQB) how many blocks each row wants
    K = int(counts.max())                         # fixed gather width for the batch
    if K == 0:
        raise ValueError("pattern selects no blocks at all")

    H, NQB, NKB = bm.shape

    # argsort on the boolean-as-int puts all the 1s (selected) before the 0s, so
    # the first counts[h,i] entries of each row are exactly the selected indices.
    #
    # stable=True keeps ties in their original order, so those selected indices
    # come out ASCENDING.  That makes `idx` deterministic and lets
    # check_gather_index_no_duplicates compare it against a sorted nonzero().
    # Set stable=False and the output stays correct but that test breaks.
    order = torch.argsort(bm.int(), dim=-1, descending=True, stable=True)
    idx = order[..., :K].contiguous()             # selected indices first

    # valid[h,i,t] = (t < counts[h,i]).  A plain rank comparison, broadcast.
    rank = torch.arange(K, device=bm.device).view(1, 1, K)
    valid = rank < counts.unsqueeze(-1)

    # NOTE: the padded slots of `idx` still hold REAL block indices (whatever
    # argsort happened to put there).  They are neutralised by `valid`, not by
    # being meaningless.  The kernel MUST AND `valid` into its attention mask.
    return idx, valid
