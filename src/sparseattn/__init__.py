"""Public API.

Three modules matter, in this reading order:
  dense.py       -- hand-written attention + all masking/NaN numerics (1.1, 1.4)
  patterns.py    -- block masks: what may attend to what (1.2)
  blocksparse.py -- the gather kernel that avoids the N x N matrix (1.2, 1.5)

Also here but not exported: harness.py (the 72 checks) and model.py (the GPT).
Start with docs/EXPLAINER.md if any of this is unfamiliar.
"""
from .dense import causal_mask, dense_attention, masked_softmax
from .blocksparse import block_sparse_attention, peak_score_elements
from . import patterns

# `patterns` is exported as a MODULE, not as individual builders: callers write
# P.sliding_window(...) / P.bigbird(...), which keeps it obvious at the call site
# that a pattern is being constructed rather than attention being computed.
__all__ = [
    "dense_attention", "masked_softmax", "causal_mask",
    "block_sparse_attention", "peak_score_elements", "patterns",
]
