"""A 2-layer character-level GPT whose attention backend is swappable
(deliverable 1.6).

Kept deliberately small and plain.  The only thing that varies between the runs
being compared is the sparsity pattern handed to `CausalSelfAttention`; the
parameter count, the initialisation, the data order and the optimiser are
identical, so a difference in loss is attributable to the pattern and nothing
else.
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
    vocab_size: int = 65
    context: int = 256
    n_layer: int = 2
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.0


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig, pattern: Optional[BlockPattern] = None):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head, self.n_embd = cfg.n_head, cfg.n_embd
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.drop = nn.Dropout(cfg.dropout)
        self.pattern = pattern
        if pattern is None:
            self.register_buffer(
                "causal", torch.ones(cfg.context, cfg.context, dtype=torch.bool).tril(),
                persistent=False)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        shape = lambda t: t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q, k, v = shape(q), shape(k), shape(v)
        if self.pattern is None:
            y = dense_attention(q, k, v, mask=self.causal[:T, :T])
        else:
            # policy='self' rather than 'zero': in a decoder, emitting a zero
            # vector for a dead query punches a hole the next layer has to
            # absorb, whereas falling back to self-attention keeps the residual
            # stream meaningful.  With the patterns used here there are no dead
            # rows anyway -- this is belt and braces.
            y = block_sparse_attention(q, k, v, self.pattern, policy="self")
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.drop(self.proj(y))


class Block(nn.Module):
    def __init__(self, cfg, pattern):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(cfg.n_embd), nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg, pattern)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd), nn.GELU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd), nn.Dropout(cfg.dropout))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class CharGPT(nn.Module):
    def __init__(self, cfg: GPTConfig, pattern: Optional[BlockPattern] = None):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos = nn.Embedding(cfg.context, cfg.n_embd)
        self.blocks = nn.ModuleList([Block(cfg, pattern) for _ in range(cfg.n_layer)])
        self.lnf = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok(idx) + self.pos(pos)
        for b in self.blocks:
            x = b(x)
        logits = self.head(self.lnf(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss
