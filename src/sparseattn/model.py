"""A 2-layer character-level GPT whose attention backend is swappable
(deliverable 1.6).

Kept deliberately small and plain.  The only thing that varies between the runs
being compared is the sparsity pattern handed to `CausalSelfAttention`; the
parameter count, the initialisation, the data order and the optimiser are
identical, so a difference in loss is attributable to the pattern and nothing
else.

Architecture background (residuals, pre-norm, why an MLP at all):
docs/EXPLAINER.md Part 9.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocksparse import block_sparse_attention
from .dense import dense_attention
from .patterns import BlockPattern, to_dense_mask


@dataclass
class GPTConfig:
    vocab_size: int = 65    # 65 distinct characters in TinyShakespeare
    context: int = 256      # max sequence length; fixes the positional table size
    n_layer: int = 2        # as the brief specifies (~10 min on a T4)
    n_head: int = 4         # parallel attention mechanisms
    n_embd: int = 128       # model width; head dim = n_embd // n_head = 32
    dropout: float = 0.0    # ZERO ON PURPOSE -- see CharGPT docstring note below


class CausalSelfAttention(nn.Module):
    """One attention sublayer.  `pattern=None` means dense; anything else is
    block-sparse.  This is the ONLY place the experiment's variable enters."""

    def __init__(self, cfg: GPTConfig, pattern: Optional[BlockPattern] = None):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0        # heads must tile the width exactly
        self.n_head, self.n_embd = cfg.n_head, cfg.n_embd

        # ONE linear producing Q, K and V concatenated (3 * n_embd wide), split
        # after.  Mathematically identical to three separate projections; one
        # bigger matmul is faster than three smaller ones.
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        # Output projection: mixes the per-head results back into one vector.
        # Without it, heads would never interact.
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.drop = nn.Dropout(cfg.dropout)
        self.pattern = pattern

        if pattern is None:
            # Precompute the causal mask once instead of per forward pass.
            # register_buffer = "part of the module state, moves with .to(device),
            # but NOT a parameter" (no gradient, no optimiser update).
            # persistent=False keeps it out of state_dict -- it is derivable from
            # config, so saving it would just bloat checkpoints.
            self.register_buffer(
                "causal", torch.ones(cfg.context, cfg.context, dtype=torch.bool).tril(),
                persistent=False)

    def forward(self, x):                          # x: (B, T, C)
        B, T, C = x.shape
        # (B,T,C) -> (B,T,3C) -> three tensors of (B,T,C)
        q, k, v = self.qkv(x).split(C, dim=2)

        # Split the channel axis into heads and move H next to B so that both act
        # as batch dimensions for the attention matmuls:
        # (B,T,C) -> (B,T,H,D) -> (B,H,T,D)
        shape = lambda t: t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q, k, v = shape(q), shape(k), shape(v)

        if self.pattern is None:
            # Slice to [:T,:T] so shorter-than-context batches still work.
            y = dense_attention(q, k, v, mask=self.causal[:T, :T])
        else:
            # policy='self' rather than 'zero': in a decoder, emitting a zero
            # vector for a dead query punches a hole the next layer has to
            # absorb, whereas falling back to self-attention keeps the residual
            # stream meaningful.  With the patterns used here there are no dead
            # rows anyway -- this is belt and braces.
            #
            # This line is also what exposed the torch.eye bug: 'self' used to
            # locate "own position" with an identity matrix, which is only the
            # diagonal in DENSE coordinates.  See dense.py::masked_softmax.
            y = block_sparse_attention(q, k, v, self.pattern, policy="self")

        # Undo the head split: (B,H,T,D) -> (B,T,H,D) -> (B,T,C).
        # .contiguous() is required because transpose leaves the memory strided
        # and .view() refuses non-contiguous input.
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.drop(self.proj(y))


class Block(nn.Module):
    """One transformer block: attention sublayer + MLP sublayer, both residual."""

    def __init__(self, cfg, pattern):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(cfg.n_embd), nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg, pattern)
        # Attention moves information BETWEEN positions; the MLP processes each
        # position independently.  4x expansion is convention.  GELU is a smooth
        # ReLU (differentiable everywhere, which helps optimisation).
        self.mlp = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd), nn.GELU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd), nn.Dropout(cfg.dropout))

    def forward(self, x):
        # PRE-NORM: x + f(ln(x)), not ln(x + f(x)).  The residual path stays
        # clean and un-normalised, which is what makes deep stacks trainable
        # without a learning-rate warmup.  GPT-2 onward use this; the original
        # 2017 paper used post-norm.
        #
        # The `x + ...` residual means each block computes a DELTA.  Without it,
        # gradients vanish through depth; with it there is always a direct path
        # from the loss to every layer.
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class CharGPT(nn.Module):
    def __init__(self, cfg: GPTConfig, pattern: Optional[BlockPattern] = None):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.n_embd)   # id -> vector lookup
        # LEARNED positional embedding.  Needed because attention is
        # permutation-invariant: without it "abc" and "cba" produce identical
        # attention patterns and the model cannot tell them apart.
        self.pos = nn.Embedding(cfg.context, cfg.n_embd)
        self.blocks = nn.ModuleList([Block(cfg, pattern) for _ in range(cfg.n_layer)])
        self.lnf = nn.LayerNorm(cfg.n_embd)                    # final norm
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)  # -> logits
        self.apply(self._init)                                 # recurse over submodules

    @staticmethod
    def _init(m):
        # std=0.02 is the GPT-2 convention.  The absolute value matters less than
        # it being IDENTICAL across the arms being compared -- torch.manual_seed
        # in the training script makes every pattern start from the same weights.
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):          # idx: (B, T) integer token ids
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        # Token meaning + position, summed (not concatenated -- summing keeps the
        # width at n_embd and is what GPT does).
        x = self.tok(idx) + self.pos(pos)
        for b in self.blocks:
            x = b(x)
        logits = self.head(self.lnf(x))            # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # cross_entropy expects (n, classes) and (n,), so flatten batch and
            # time together.  It combines log_softmax + NLL in one numerically
            # stable op -- doing them separately would risk log(0).
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss
