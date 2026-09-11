"""Deliverable 1.4: where the NaN comes from, and why the obvious fix is wrong.

Run:  python scripts/demo_nan.py
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import torch
from sparseattn import dense_attention, patterns as P
from sparseattn.blocksparse import block_sparse_attention


def rule(t):
    print(f"\n{'='*72}\n{t}\n{'='*72}")


torch.manual_seed(0)
B, H, N, D = 1, 2, 8, 4
q, k, v = (torch.randn(B, H, N, D) for _ in range(3))

rule("1. The failure, in four lines")
mask = torch.ones(B, H, N, N, dtype=torch.bool).tril()
mask[0, 0, 3, :] = False          # query 3 of head 0 is allowed to see nothing
scores = (q @ k.transpose(-2, -1)) / D**0.5
naive = torch.softmax(scores.masked_fill(~mask, float("-inf")), dim=-1)
print("row 3 logits :", scores.masked_fill(~mask, float('-inf'))[0, 0, 3].tolist())
print("row 3 softmax:", naive[0, 0, 3].tolist())
print("\nsoftmax subtracts the row max for stability: exp(x_i - max).")
print("Here every x_i is -inf, so max is -inf, and -inf - (-inf) = nan.")
print("exp(nan) = nan. The row is nan before any division happens.")
print(f"\nnan in the whole output tensor? {bool(torch.isnan(naive @ v).any())}")

rule("2. Why patching the nan afterwards is not a fix")
print("""The obvious repair is torch.nan_to_num.  Whether it actually works turns
out to depend on two choices made elsewhere in the file: how the mask was
applied, and where the patch is inserted.  Measured, not asserted:""")

def variant(additive: bool, patch: str):
    qg = q.clone().requires_grad_(True)
    vg = v.clone().requires_grad_(True)
    s_ = (qg @ k.transpose(-2, -1)) / D**0.5
    if additive:                        # what HuggingFace et al. actually do
        s_ = s_ + torch.zeros_like(s_).masked_fill(~mask, float("-inf"))
    else:
        s_ = s_.masked_fill(~mask, float("-inf"))
    a_ = torch.softmax(s_, dim=-1)
    out_ = torch.nan_to_num(a_) @ vg if patch == "attn" else torch.nan_to_num(a_ @ vg)
    f = bool(torch.isnan(out_).any())
    out_.sum().backward()
    return f, bool(torch.isnan(qg.grad).any()), bool(torch.isnan(vg.grad).any())

print(f"\n{'mask applied via':18s} {'patched at':12s}  forward   dQ      dV")
print("-" * 58)
for add in (False, True):
    for patch in ("attn", "out"):
        f, gq, gv = variant(add, patch)
        style = "additive -inf" if add else "masked_fill"
        fmt = lambda b: "nan" if b else "clean"
        print(f"{style:18s} {patch:12s}  {fmt(f):8s} {fmt(gq):7s} {fmt(gv):6s}")

qg = q.clone().requires_grad_(True); vg = v.clone().requires_grad_(True)
o = dense_attention(qg, k, vg, mask=mask, policy="zero"); o.sum().backward()
print(f"{'ours (pre-softmax)':18s} {'n/a':12s}  "
      f"{'clean' if not torch.isnan(o).any() else 'nan':8s} "
      f"{'clean' if not torch.isnan(qg.grad).any() else 'nan':7s} "
      f"{'clean' if not torch.isnan(vg.grad).any() else 'nan':6s}")

print("""
Read the forward column first: it is clean in every row.  That is the trap.  The
nan is gone from the values you can print, and the gradient is poisoned anyway.

Three of the four combinations corrupt a gradient.  The one that survives does so
by accident: masked_fill's backward zeroes the gradient at masked positions, which
happens to kill the nan on its way to Q.  Swap masked_fill for an additive -inf
bias -- which is what most production attention code uses, because it composes
with other biases -- and that accident disappears.  Nobody writing the nan_to_num
line is thinking about which of these four squares they are standing in.

Repairing before the softmax removes the dependency entirely.  There is no nan to
patch because no nan is ever constructed: the dead rows are given a finite set of
logits so softmax is well defined, and their weights are zeroed afterwards, which
also makes their gradient contribution exactly zero.""")

rule("3. How it arises for real: a block-boundary off-by-one")
pat = P.sliding_window(N * 8, 8, window_blocks=1, causal=True,
                       include_self_block=False, n_heads=H)
m = P.to_dense_mask(pat, N * 8, N * 8)
dead = (~m.any(-1)).sum(-1)
print("pattern: causal sliding window that forgets to include its own block")
print(f"dead query rows per head: {dead.tolist()}  (of {N*8} queries)")
print("""
Query block 0 has no earlier block to look at, and the off-by-one removed the
only block it could have used -- itself.  Every token in block 0 is dead.  This
is not a contrived mask; it is one wrong comparison operator in a window bound,
and at block granularity it takes out a whole block at a time.""")

rule("4. The three policies")
qq, kk, vv = (torch.randn(1, H, N * 8, D) for _ in range(3))
for pol in ("zero", "self", "raise"):
    try:
        o = block_sparse_attention(qq, kk, vv, pat, policy=pol)
        rows = o[0, 0, :8]
        print(f"policy={pol:6s} -> no nan={not bool(torch.isnan(o).any())}, "
              f"block-0 rows all zero={bool((rows == 0).all())}")
    except ValueError as e:
        print(f"policy={pol:6s} -> raised: {e}")
print("""
'zero'  : the dead query emits the zero vector.  Correct for a pooled encoder
          where a padded position's output is discarded anyway.
'self'  : the dead query falls back to attending to itself, so it emits a
          projection of its own value.  Correct for a decoder, where returning
          zeros mid-sequence would be a hole the next layer has to absorb.
'raise' : correct while developing.  A dead row almost always means the pattern
          is wrong, and silently papering over it hides the real bug.""")
