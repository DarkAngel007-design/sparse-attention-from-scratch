"""Correctness checks (deliverable 1.3, plus the 1.4 cases).

Every check compares the block-sparse kernel against `dense_attention` run with
*the same pattern expanded to a token mask*.  That is the only comparison that
means anything: "matches dense on unmasked positions" is not a statement about
unmasked dense attention, it is a statement that the fast path reproduces
softmax(QK^T + mask)V exactly.

Both `tests/test_correctness.py` and `scripts/check_correctness.py` consume the
cases defined here so the two can never disagree about what passed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator

import torch

from .blocksparse import block_sparse_attention
from .dense import dense_attention
from . import patterns as P

TOL = {torch.float64: 1e-12, torch.float32: 2e-6}


@dataclass
class Result:
    ok: bool
    detail: str


@dataclass
class Check:
    name: str
    run: Callable[[], Result]


def _qkv(B, H, N, D, dtype, scale=1.0, seed=0, device="cpu"):
    g = torch.Generator(device="cpu").manual_seed(seed)
    mk = lambda: (torch.randn(B, H, N, D, generator=g, dtype=torch.float64) * scale).to(
        device=device, dtype=dtype)
    return mk(), mk(), mk()


def _build(name, N, bs, H, causal, seed=0, device="cpu"):
    if name == "sliding_window":
        return P.sliding_window(N, bs, 1, causal=causal, n_heads=H, device=device)
    if name == "bigbird":
        return P.bigbird(N, bs, 1, 1, 2, causal=causal, n_heads=H, seed=seed, device=device)
    if name == "dilated":
        return P.dilated(N, bs, 3, causal=causal, n_heads=H, device=device)
    if name == "dense":
        return P.dense_pattern(N, bs, causal=causal, n_heads=H, device=device)
    raise KeyError(name)


PATTERNS = ["sliding_window", "bigbird", "dilated", "dense"]


def _match(pattern_name, N, bs, dtype, causal, B=2, H=4, D=32, scale=1.0,
           q_chunk=None, device="cpu") -> Result:
    """The core check: kernel output == dense output under the same mask."""
    pat = _build(pattern_name, N, bs, H, causal, device=device)
    q, k, v = _qkv(B, H, N, D, dtype, scale=scale, device=device)
    mask = P.to_dense_mask(pat, N, N).unsqueeze(0)
    ref = dense_attention(q, k, v, mask=mask)
    got = block_sparse_attention(q, k, v, pat, q_chunk=q_chunk)
    err = float((ref - got).abs().max())
    tol = TOL[dtype]
    return Result(err <= tol, f"max|err|={err:.3e} tol={tol:.0e} "
                              f"density={pat.token_density(N):.3f}")


def _iter_match_checks(device="cpu") -> Iterator[Check]:
    for name in PATTERNS:
        for N, bs in [(256, 64), (320, 64), (512, 128), (200, 32)]:
            for dtype in (torch.float64, torch.float32):
                for causal in (True, False):
                    label = (f"match/{name}/N{N}/bs{bs}/"
                             f"{'f64' if dtype == torch.float64 else 'f32'}/"
                             f"{'causal' if causal else 'full'}")
                    yield Check(label, lambda n=name, N=N, bs=bs, d=dtype, c=causal:
                                _match(n, N, bs, d, c, device=device))


def check_dense_pattern_is_plain_dense(device="cpu") -> Result:
    """A full-density non-causal pattern must reproduce unmasked dense attention.

    This is the check that catches a systematically wrong gather: if the kernel
    silently dropped or duplicated blocks, it would still be self-consistent
    against a mask built the same wrong way, but it could not reproduce plain
    softmax(QK^T)V."""
    N, bs, B, H, D = 256, 64, 2, 4, 32
    q, k, v = _qkv(B, H, N, D, torch.float64, device=device)
    pat = P.dense_pattern(N, bs, causal=False, n_heads=H, device=device)
    err = float((dense_attention(q, k, v) - block_sparse_attention(q, k, v, pat)).abs().max())
    return Result(err <= 1e-12, f"max|err|={err:.3e}")


def check_chunk_invariance(device="cpu") -> Result:
    """Peak memory is traded off with q_chunk; the numerics must not move."""
    N, bs, B, H, D = 512, 64, 2, 4, 32
    pat = _build("bigbird", N, bs, H, True, device=device)
    q, k, v = _qkv(B, H, N, D, torch.float64, device=device)
    full = block_sparse_attention(q, k, v, pat)
    worst = 0.0
    for c in (1, 2, 3, 8):
        worst = max(worst, float((full - block_sparse_attention(q, k, v, pat, q_chunk=c)).abs().max()))
    return Result(worst == 0.0, f"max|err| over q_chunk in 1,2,3,8 = {worst:.3e}")


def check_gather_index_no_duplicates(device="cpu") -> Result:
    """Padding slots must be marked invalid, never filled with a repeated block.

    A duplicated key block would enter the softmax twice and be weighted twice.
    The output stays finite and smooth, so nothing downstream complains."""
    from .patterns import to_gather_index
    bad = []
    for name in PATTERNS:
        pat = _build(name, 512, 64, 4, True, device=device)
        idx, valid = to_gather_index(pat)
        counts = pat.block_mask.sum(-1)
        if not torch.equal(valid.sum(-1), counts):
            bad.append(f"{name}: valid count != block count")
        for h in range(idx.shape[0]):
            for i in range(idx.shape[1]):
                sel = idx[h, i][valid[h, i]]
                if sel.numel() != sel.unique().numel():
                    bad.append(f"{name}: duplicate block at head {h} qblock {i}")
                want = pat.block_mask[h, i].nonzero(as_tuple=True)[0]
                if not torch.equal(sel.sort().values, want):
                    bad.append(f"{name}: gathered set != pattern at head {h} qblock {i}")
    return Result(not bad, "; ".join(bad[:3]) if bad else "indices exact, no duplicates")


def check_adversarial_magnitudes(device="cpu") -> Result:
    """Large-magnitude logits (deliverable 1.4's cousin: overflow, not nan).

    q,k are scaled by 30, giving logits around +-5e3.  Two separate claims,
    and it matters that they are separated:

      1. The kernel must agree with the dense reference *at the same precision*.
         That is the correctness claim, and it is asserted.
      2. float32 attention diverges from float64 attention at these magnitudes.
         That is a property of float32, not of this kernel -- both paths drift
         together, by the same amount.  It is reported, not asserted.

    Folding (2) into the pass criterion would make the harness fail for a reason
    that has nothing to do with the code under test.  The number is still worth
    printing: the maximum rounding error on a float32 logit is half an ulp (2^-24
    relative), so at |logit| ~ 5e3 it is ~2.4e-4; softmax weights inherit that as
    relative error, and it lands in the output scaled by |v|.
    """
    N, bs, B, H, D = 256, 64, 1, 2, 32
    pat = _build("bigbird", N, bs, H, True, device=device)
    q, k, v = _qkv(B, H, N, D, torch.float32, scale=30.0, device=device)
    mask = P.to_dense_mask(pat, N, N).unsqueeze(0)

    ref32 = dense_attention(q, k, v, mask=mask)
    got32 = block_sparse_attention(q, k, v, pat)
    ref64 = dense_attention(q.double(), k.double(), v.double(), mask=mask)

    kernel_err = float((ref32 - got32).abs().max())
    precision_gap = float((ref32.double() - ref64).abs().max())
    naive = torch.exp((q @ k.transpose(-2, -1)) / D**0.5)
    logit_max = float((q @ k.transpose(-2, -1)).abs().max()) / D**0.5
    finite = bool(torch.isfinite(got32).all())

    ok = finite and kernel_err <= TOL[torch.float32]
    return Result(ok, f"logits +-{logit_max:.0f} | naive exp() overflowed="
                      f"{bool(torch.isinf(naive).any())} | ours finite={finite} | "
                      f"kernel vs dense (both f32) = {kernel_err:.2e} [asserted] | "
                      f"f32 vs f64 dense = {precision_gap:.2e} [reported: float32's "
                      f"own error, both paths drift together]")


def check_degenerate_equal_scores(device="cpu") -> Result:
    """All-equal logits: weights must be exactly uniform over the allowed set."""
    N, bs, B, H, D = 128, 32, 1, 1, 16
    pat = _build("sliding_window", N, bs, H, True, device=device)
    q = torch.zeros(B, H, N, D, dtype=torch.float64, device=device)
    k = torch.zeros(B, H, N, D, dtype=torch.float64, device=device)
    v = torch.randn(B, H, N, D, dtype=torch.float64, device=device)
    mask = P.to_dense_mask(pat, N, N).unsqueeze(0)
    want = (mask.to(v.dtype) / mask.sum(-1, keepdim=True).to(v.dtype)) @ v
    got = block_sparse_attention(q, k, v, pat)
    err = float((want - got).abs().max())
    return Result(err <= 1e-12, f"max|err| vs exact uniform mean = {err:.3e}")


def check_dead_rows(device="cpu") -> Result:
    """Deliverable 1.4 as an assertion rather than a demo."""
    N, bs, B, H, D = 256, 32, 1, 2, 16
    pat = P.sliding_window(N, bs, 1, causal=True, n_heads=H,
                           include_self_block=False, device=device)
    q, k, v = _qkv(B, H, N, D, torch.float32, device=device)
    msgs = []
    ok = True
    o = block_sparse_attention(q, k, v, pat, policy="zero")
    ok &= bool(torch.isfinite(o).all()) and bool((o[:, :, :bs] == 0).all())
    msgs.append(f"zero: finite={bool(torch.isfinite(o).all())}, block0 zeroed=True")
    o = block_sparse_attention(q, k, v, pat, policy="self")
    ok &= bool(torch.isfinite(o).all()) and not bool((o[:, :, :bs] == 0).all())
    msgs.append("self: finite=True, block0 non-zero")
    try:
        block_sparse_attention(q, k, v, pat, policy="raise")
        ok = False
        msgs.append("raise: DID NOT RAISE")
    except ValueError:
        msgs.append("raise: raised")
    naive = block_sparse_attention(q, k, v, pat, policy="naive")
    ok &= bool(torch.isnan(naive).any())
    msgs.append(f"naive: produces nan={bool(torch.isnan(naive).any())}")
    return Result(ok, "; ".join(msgs))


def check_backward_matches(device="cpu") -> Result:
    """Gradients, not just outputs.  A kernel can be right forward and wrong back."""
    N, bs, B, H, D = 256, 64, 1, 2, 16
    pat = _build("bigbird", N, bs, H, True, device=device)
    mask = P.to_dense_mask(pat, N, N).unsqueeze(0)
    base = _qkv(B, H, N, D, torch.float64, device=device)
    grads = []
    for fn in (lambda a, b, c: dense_attention(a, b, c, mask=mask),
               lambda a, b, c: block_sparse_attention(a, b, c, pat)):
        t = [x.clone().requires_grad_(True) for x in base]
        out = fn(*t)
        (out * torch.linspace(0.1, 1.0, D, dtype=torch.float64, device=device)).sum().backward()
        grads.append([x.grad for x in t])
    err = max(float((a - b).abs().max()) for a, b in zip(*grads))
    return Result(err <= 1e-11, f"max|dQ,dK,dV err| = {err:.3e}")


def check_per_head_patterns(device="cpu") -> Result:
    """Per-head random blocks must actually reach the kernel, per head."""
    N, bs, B, H, D = 512, 64, 1, 4, 16
    pat = P.bigbird(N, bs, 1, 1, 2, causal=True, n_heads=H, per_head_random=True,
                    seed=3, device=device)
    distinct = len({tuple(pat.block_mask[h].flatten().tolist()) for h in range(H)})
    q, k, v = _qkv(B, H, N, D, torch.float64, device=device)
    mask = P.to_dense_mask(pat, N, N).unsqueeze(0)
    err = float((dense_attention(q, k, v, mask=mask)
                 - block_sparse_attention(q, k, v, pat)).abs().max())
    return Result(distinct > 1 and err <= 1e-12,
                  f"{distinct}/{H} heads have distinct patterns, max|err|={err:.3e}")


def all_checks(device="cpu") -> list[Check]:
    checks = list(_iter_match_checks(device))
    checks += [
        Check("dense_pattern_is_plain_dense", lambda: check_dense_pattern_is_plain_dense(device)),
        Check("chunk_invariance", lambda: check_chunk_invariance(device)),
        Check("gather_index_no_duplicates", lambda: check_gather_index_no_duplicates(device)),
        Check("adversarial_magnitudes", lambda: check_adversarial_magnitudes(device)),
        Check("degenerate_equal_scores", lambda: check_degenerate_equal_scores(device)),
        Check("dead_rows_1.4", lambda: check_dead_rows(device)),
        Check("backward_matches_dense", lambda: check_backward_matches(device)),
        Check("per_head_patterns", lambda: check_per_head_patterns(device)),
    ]
    return checks
