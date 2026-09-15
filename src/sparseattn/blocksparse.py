"""Block-sparse attention by gathering K/V blocks (deliverable 1.2 / 1.5).

The point of this file is that the (Nq x Nk) score matrix is never built.  For
each query block we gather only the K/V blocks its pattern selects, so the
largest tensor allocated is (B, H, NQB, bs, K*bs) instead of (B, H, N, N).

K -- the number of key blocks a query block selects -- is constant in N for
these patterns, which is exactly why the score tensor is O(N) rather than
O(N^2).  Measured at bs = 64: sliding window K = 2 and BigBird K = 5 at every
sequence length tested.  At N = 8192 that is a 64x and 25.6x smaller score
tensor respectively.

What this is *not*: a fused kernel.  The gather physically copies K and V, and
for a sliding window the copies overlap (block i-1 is gathered again by query
block i and again by i+1), so the K/V traffic goes up even as the score matrix
goes down.  That tradeoff is the honest limit of doing this in PyTorch, and it
is exactly what a Triton flash-attention kernel removes -- see WRITEUP.md.

Walkthrough with shapes at every step: docs/EXPLAINER.md Part 4.
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
    # (-n) % bs is the "distance up to the next multiple" idiom.  Python's % on a
    # negative returns a NON-NEGATIVE result (unlike C), which is exactly what we
    # want: n=320,bs=64 -> 0;  n=200,bs=32 -> 24.
    rem = (-n) % block_size
    if rem:
        # F.pad's pad tuple runs from the LAST dim backwards:
        # (left_D, right_D, left_N, right_N).  So (0,0,0,rem) means "add rem
        # zeros at the end of the sequence axis, touch nothing in D".
        x = torch.nn.functional.pad(x, (0, 0, 0, rem))
    return x, n


def block_sparse_attention(
    q: torch.Tensor,                      # (B, H, N, D)
    k: torch.Tensor,                      # (B, H, N, D)
    v: torch.Tensor,                      # (B, H, N, D)
    pattern: BlockPattern,
    scale: Optional[float] = None,
    policy: DeadRowPolicy = "zero",
    q_chunk: Optional[int] = None,        # query blocks processed per iteration
) -> torch.Tensor:
    """q, k, v: (B, H, N, D) -> (B, H, N, D).

    `q_chunk` caps how many query blocks are processed at once.  Peak memory is
    linear in it, so it is the knob that decides whether N = 8192 fits.  It has
    NO effect on the numerics -- check_chunk_invariance asserts bit-identical
    output across chunk sizes.  Memory is a free parameter; correctness is not.
    """
    B, H, N, D = q.shape

    # ---- Guards: fail loudly, naming the assumption ---------------------------
    # A cross-attention caller gets a sentence here instead of a shape error 40
    # frames deep inside the gather.
    if k.shape[-2] != N or v.shape[-2] != N:
        raise ValueError("this implementation assumes self-attention (Nq == Nk)")
    if pattern.n_heads not in (1, H):
        raise ValueError(f"pattern has {pattern.n_heads} heads, tensors have {H}")

    bs = pattern.block_size
    scale = scale if scale is not None else 1.0 / math.sqrt(D)
    dev = q.device

    # ---- Step 1: pad the sequence up to a whole number of blocks --------------
    # Zero padding is safe because those positions are removed from the mask
    # below (kpos < N) and sliced off the output at the end.
    qp, _ = _pad_to_blocks(q, bs)
    kp, _ = _pad_to_blocks(k, bs)
    vp, _ = _pad_to_blocks(v, bs)
    n_pad = qp.shape[-2]
    nqb = n_pad // bs                     # number of blocks after padding

    # ---- Step 2: which key blocks does each query block want? -----------------
    idx, valid = to_gather_index(pattern)         # (Hp, NQB, K)
    idx, valid = idx.to(dev), valid.to(dev)
    Hp, _, K = idx.shape
    if Hp == 1 and H > 1:
        # A shared (single-head) pattern costs nothing: expand is a VIEW, no copy.
        # This is what makes `n_heads=1` the cheap common case while still
        # supporting per-head patterns (the stretch goal).
        idx = idx.expand(H, -1, -1)
        valid = valid.expand(H, -1, -1)

    # ---- Step 3: reinterpret the sequence axis as (blocks, tokens-in-block) ---
    # Pure reshape, no data movement.  kb[b,h,j] is now key block j as a (bs,D)
    # slab, which is the unit the gather moves around.
    kb = kp.view(B, H, nqb, bs, D)
    vb = vp.view(B, H, nqb, bs, D)
    qb = qp.view(B, H, nqb, bs, D)

    # ---- Step 4: reconstruct absolute positions of every gathered slot --------
    # After gathering, the last axis is "slot 0..K*bs-1", which says nothing
    # about WHERE in the sequence those keys came from.  Causal masking needs
    # that, so compute it now: block j starts at token j*bs, so token t inside it
    # is at j*bs + t.
    within = torch.arange(bs, device=dev)                       # (bs,)
    # idx.unsqueeze(-1) is (H,NQB,K,1); + within broadcasts to (H,NQB,K,bs);
    # reshape flattens the K blocks of bs tokens into one K*bs axis.
    kpos = (idx.unsqueeze(-1) * bs + within).reshape(H, nqb, K * bs)

    # repeat_interleave (NOT repeat): expand each block's validity flag to all bs
    # of its tokens.  [True,False] with bs=2 -> [T,T,F,F].  Plain .repeat() would
    # give [T,F,T,F] -- silently wrong masks.
    valid_tok = valid.repeat_interleave(bs, dim=-1)             # (H, NQB, K*bs)
    # Drop the right-edge zero padding: those key positions exist in the tensor
    # but are not real tokens.  Only matters when N % bs != 0 -- which is exactly
    # why the harness tests N=320/bs=64 and N=200/bs=32.
    valid_tok = valid_tok & (kpos < N)
    qpos_all = torch.arange(n_pad, device=dev).view(nqb, bs)    # (NQB, bs)

    # Head index broadcast to the shape of idx, so advanced indexing below can
    # pair (head, block) elementwise and let each head gather its own blocks.
    hh = torch.arange(H, device=dev).view(H, 1, 1).expand(H, nqb, K)

    # ---- Step 5: the main loop over query-block chunks ------------------------
    step = q_chunk or nqb                 # None => one iteration over everything
    outs = []
    for s in range(0, nqb, step):
        e = min(s + step, nqb)
        c = e - s                         # query blocks in THIS chunk

        # THE GATHER.  Two index tensors of shape (H,c,K) applied to dims 1 and 2
        # of kb (B,H,NQB,bs,D).  PyTorch pairs them elementwise and inserts the
        # index dims where they were, giving (B,H,c,K,bs,D): for each head, for
        # each query block in the chunk, the K key blocks it selected.
        ci, ch = idx[:, s:e], hh[:, s:e]                        # (H, c, K)
        kg = kb[:, ch, ci]                                      # (B,H,c,K,bs,D)
        vg = vb[:, ch, ci]
        # Flatten "K blocks of bs tokens" into one K*bs key axis.  THIS RESHAPE
        # IS A COPY -- the gather result is not contiguous.  That copy is the
        # honest cost of doing block-sparse attention in PyTorch, and at high
        # sparsity it dominates the memory (67 MB of gathered K/V against 34 MB
        # of scores at N=8192).  A fused Triton kernel never makes it.
        kg = kg.reshape(B, H, c, K * bs, D)
        vg = vg.reshape(B, H, c, K * bs, D)

        # Scores for this chunk only.  (B,H,c,bs,D) @ (B,H,c,D,K*bs).
        # NEVER (N, N) -- that is the entire point of the file.
        scores = (qb[:, :, s:e] @ kg.transpose(-2, -1)) * scale  # (B,H,c,bs,K*bs)

        # ---- Build the mask, axis by axis ------------------------------------
        # Every axis must line up with scores' (B,H,c,bs,K*bs).  Getting this
        # wrong is the most likely bug in the file: the original code used
        # .unsqueeze(1).unsqueeze(0) here, producing (1,H,1,c,K*bs) with c and bs
        # swapped, and failed with "expanded size (64) must match existing size
        # (5) at non-singleton dimension 3".
        m = valid_tok[:, s:e].unsqueeze(0).unsqueeze(3)         # (1,H,c,1,K*bs)
        m = m.expand(1, H, c, bs, K * bs)

        qp_c = qpos_all[s:e].view(1, 1, c, bs, 1)               # (1,1,c,bs,1)
        kp_c = kpos[:, s:e].view(1, H, c, 1, K * bs)            # (1,H,c,1,K*bs)
        if pattern.causal:
            # Broadcasting (bs,1) against (1,K*bs) produces the full (bs,K*bs)
            # causal comparison for every query block at once -- the entire
            # causal mask, without ever building (N,N).
            m = m & (kp_c <= qp_c)

        # Where does each query's own position land among the gathered slots?
        # The kernel's last axis is gathered key slots, not key positions, so
        # policy='self' cannot find the diagonal on its own.  Each absolute key
        # position appears in at most one slot (to_gather_index emits a
        # permutation), so this is unambiguous.
        fb = (kp_c == qp_c) & (kp_c < N) if policy == "self" else None

        # Same masked_softmax as the dense path -- one implementation of the
        # numerics, so the two paths cannot disagree about masking or NaN.
        attn = masked_softmax(scores, m, policy=policy, fallback=fb)
        outs.append(attn @ vg)                                  # (B,H,c,bs,D)

    # ---- Step 6: stitch the chunks back into a sequence and unpad -------------
    out = torch.cat(outs, dim=2).reshape(B, H, n_pad, D)
    return out[:, :, :N]                  # drop the padding added in step 1


def peak_score_elements(pattern: BlockPattern, seq_len: int, batch: int, heads: int,
                        q_chunk: Optional[int] = None) -> int:
    """Elements in the largest score tensor -- the analytic version of 1.5.

    Reported alongside the MEASURED peak memory so the O(N) vs O(N^2) claim can
    be checked two independent ways.  Theory and measurement agreeing is much
    stronger evidence than either alone -- and it does not depend on the memory
    probe, which on MPS is only a 2 kHz sample.
    """
    bs = pattern.block_size
    nqb = (seq_len + bs - 1) // bs
    _, valid = to_gather_index(pattern)
    K = valid.shape[-1]
    c = min(q_chunk or nqb, nqb)
    # B * H * (query blocks in a chunk) * (tokens per block) * (gathered keys).
    # Note K*bs is constant in seq_len for these patterns => linear in N.
    return batch * heads * c * bs * K * bs
