# Sparse Attention from Scratch: Writeup

**Deliverable 1.7.** All numbers: Apple M5 Pro, 18 cores, 25.8 GB, PyTorch 2.14.0,
MPS. Absolute milliseconds are not portable; the ratios are the claim. Every figure
is reproduced from the committed `results/*.json`.

## 1. What was built

A sparsity pattern is a **block mask** `(H, NQB, NKB)`: "may query block *i* of head
*h* read key block *j*". One object, two consumers. `to_dense_mask` expands it to a
token mask driving the hand-written dense reference, `to_gather_index` compacts it
to gather indices driving the fast path. Both derive from the same source, so the
harness cannot drift from what it tests.

Granularity is **block**, not token, because a token mask still has to be applied to
an `N×N` matrix, so you pay full quadratic memory to throw most of it away. Sparsity
pays only when structured enough to skip contiguous slabs of K/V, which is why
BigBird and Longformer are block algorithms.

So `blocksparse.py` never materialises `N×N`; its largest tensor is
`(B, H, NQB, bs, K·bs)`, 21× smaller at `N=8192, bs=64, K=6`. What it is **not** is a
fused kernel: the gather copies K and V, and for a sliding window those copies
overlap, so the score matrix shrinks while K/V traffic grows. That is the honest
ceiling in PyTorch, and precisely what a Triton kernel (Task 4) removes.

## 2. Correctness (1.3)

72 checks, `python scripts/check_correctness.py`, all passing. Every sparse variant
is compared against **dense attention run with the same pattern expanded to a token
mask**. The claim is that the fast path reproduces `softmax(QKᵀ + mask)V`, not
that it resembles *unmasked* dense attention. It does, bit-exactly, in float32 and
float64. The checks that earned their place:

- **A full-density non-causal pattern must reproduce plain `softmax(QKᵀ)V`.** A
  systematically wrong gather would still agree with a mask built the same wrong
  way; it could not agree with unmasked dense attention.
- **`q_chunk` invariance**: bit-identical for chunk ∈ {1,2,3,8}.
- **Gather-index exactness.** Query blocks select different *numbers* of key blocks
  but a batched gather needs a fixed `K`. Padding short rows with a repeated real
  index is the tempting fix and is wrong: the duplicated block enters the softmax
  twice and is weighted twice, and the output stays finite, smooth and trainable,
  and converges to the wrong answer. Hence a separate validity mask.
- **Backward-pass equivalence**, max |dQ,dK,dV error| = 6.2e-15. A kernel can be
  right forward and wrong backward, and only forward is ever inspected.

One check failed instructively: under adversarial magnitudes (logits ≈ ±4800) the
sparse output differed from the float64 reference by 9.2e-3. Not a kernel error:
float32 sparse vs **float32** dense is bit-exact; the gap is float32 diverging from
float64 in the reference itself, and half an ulp at |logit| ≈ 4800 (≈2.4e-4),
inherited by softmax and scaled by `|v| ≈ 30`, predicts ≈ 7.3e-3 against a
measured 9.2e-3. The check now asserts the
kernel claim and *reports* the precision one separately.

## 3. The NaN (1.4)

Softmax subtracts the row maximum for stability: `exp(xᵢ − max)`. If a query's
entire allowed set is masked, every logit is `−inf`, so `max` is `−inf`, and
`−inf − (−inf) = nan`. The row is NaN before any division happens.

**It arises from ordinary mistakes, not contrived masks.** `scripts/demo_nan.py`
produces it from one wrong comparison in a causal sliding window's bound, a window
excluding its own block. Query block 0 then has no earlier block and nothing of its
own, so all 64 of its queries are dead: at block granularity an off-by-one takes out
a whole block at a time.

**Patching afterwards is not a fix.** `torch.nan_to_num` is the obvious repair.
Whether it works depends on two choices made elsewhere in the file. Measured:

| mask applied via | `nan_to_num` at | forward | dQ | dV |
| --- | --- | --- | --- | --- |
| `masked_fill` | attention weights | clean | clean | clean |
| `masked_fill` | output | clean | clean | **NaN** |
| additive `−inf` bias | attention weights | clean | **NaN** | clean |
| additive `−inf` bias | output | clean | **NaN** | **NaN** |

Read the forward column first: clean in every row. That is the trap: the NaN is
gone from the values you can print and the gradient is poisoned anyway. Loss prints
finite, parameters turn to NaN, the run dies hundreds of steps later.

Three of four combinations corrupt a gradient, and the survivor does so **by
accident**: `masked_fill`'s backward zeroes the gradient at masked positions, which
happens to kill the NaN on its way to Q. Swap in an additive `−inf` bias (what most
production code uses, since it composes with other biases) and the accident
disappears.

**The fix** removes the dependency: repair *before* the softmax. Dead rows get a
finite set of logits, so softmax is well-defined and differentiable, and their
weights are then zeroed. No NaN is ever constructed, and a dead row's gradient
contribution is exactly zero rather than exactly wrong.

Three policies, because the right answer depends on the caller: `zero` (a padded
encoder position whose output is discarded), `self` (a decoder, where a zero vector
mid-sequence is a hole the next layer must absorb), `raise` (while developing).

`policy="self"` produced a real bug: it located the query's own position with
`torch.eye`, the diagonal only in **dense** coordinates. Inside the kernel the last
axis is *gathered key slots* with scattered absolute positions, where an identity
matrix is meaningless. `masked_softmax` now requires the caller to supply the
fallback mask (`key_position == query_position`) and raises if it fails to revive
every dead row, rather than assuming a layout.

## 4. Benchmark (1.5)

Forward only, `torch.no_grad()`, B=1, H=8, D=64, block 64, median of 5. Each
configuration runs in a **fresh subprocess**, so peak memory is a clean high-water
mark rather than a residue of the previous run.

| N | dense ms / MB | sliding ms / MB | BigBird ms / MB |
| --- | --- | --- | --- |
| 512 | 1.68 / 28.8 | 1.40 / 14.8 | 1.88 / 32.2 |
| 1024 | 4.96 / 109.1 | 3.42 / 29.5 | 4.79 / 64.4 |
| 2048 | 10.45 / 557.8 | 3.03 / 59.1 | 5.23 / 128.7 |
| 4096 | 40.20 / 1669.3 | 6.31 / 118.1 | 8.35 / 257.4 |
| 8192 | 159.12 / 6627.0 | **7.42 / 236.2** | 16.97 / 514.8 |

At N=8192 the sliding window is **21× faster and 28× lighter**; BigBird 9.4× and
12.9×. Peak memory has no portable API, so the probe is recorded per result: CUDA
`max_memory_allocated`, CPU `ru_maxrss` (exact, hence the subprocess), MPS a 2 kHz
sampling thread on `current_allocated_memory`, since `torch.mps` exposes no peak
counter. The analytic score-tensor size is reported alongside, and agrees.

**Where sparse loses.** At N=512, BigBird (1.88 ms) and dilated (1.82 ms) are
*slower* than dense (1.68 ms). The gather copies K and V, and at short sequences
that traffic exceeds the FLOPs saved, while dense at 512 is one large well-shaped
matmul, the case hardware is best at. Crossover is near N≈1024.

## 5. Quality (1.6)

Same 2-layer char GPT (445,184 params), TinyShakespeare, context 256, block 32.
Seed, init, batch order, optimiser and step count identical across patterns.
**Only the mask changes.** Three seeds, 8000 steps, so a gap can be compared against
run-to-run noise rather than asserted.

| pattern | density | final val | seed spread | vs dense | beyond noise? |
| --- | --- | --- | --- | --- | --- |
| dense | 100% | 1.5485 | 0.0023 | n/a | n/a |
| sliding window | 34.6% | 1.5601 | 0.0097 | +0.0116 | yes |
| BigBird | 81.3% | 1.5496 | 0.0050 | +0.0010 | no |
| dilated | 65.8% | 1.5433 | 0.0112 | −0.0052 | no |

**BigBird and dilated are statistically indistinguishable from dense.** Sliding
window costs +0.0116 nats/char, five times its own seed spread, but 0.75% in
relative terms while using a third of the attention.

### The ranking reverses depending on where you stop

An earlier version ran 4000 steps, one seed, and concluded every sparse pattern
**beat** dense. That was not noise (the gap was 0.064, ~28× the seed spread), but
it measured a model that had not converged:

| | step 4000 | step 8000 | best val (when) | final train loss |
| --- | --- | --- | --- | --- |
| dense | 1.5882 | **1.5485** | 1.5479 @ 7000 | 1.2579 |
| sliding window | **1.5406** | 1.5601 | 1.5406 @ 4000 | 1.1759 |
| dilated | n/a | 1.5433 | **1.5340 @ 6000** | 1.2007 |

Dense overtakes sliding window at **step 7000**, and two opposing effects produce
the crossing. **Sparse converges faster**: a constrained pattern is a smaller and,
for character-level text, well-matched hypothesis space: dense must *learn* to
ignore distant tokens, sliding window is told. **Sparse then overfits harder**: it
ends with the **lowest training loss of all four** (1.1759 vs dense 1.2579) and the
**highest validation loss**, regressing +0.0195 after step 4000 against dense's
+0.0007. Not short of capacity: fitting better, generalising worse.

So "does sparsity cost you loss?" has an answer per training budget. Reporting one
number at one step, which is what the deliverable asks for, would have supported
either conclusion depending on which step I picked.

**What this does not show** is that sparse attention is free in general, only that
*for this task* dense's long-range attention is not carrying much loss.
Character-level Shakespeare at 256 tokens is dominated by local structure; a
retrieval task separates these patterns immediately, as the companion Task 3
repository shows (local-only policies score respectable perplexity and retrieve a
planted fact **0%** of the time).

One caveat: at context 256 with block 32 there are only 8 blocks, so BigBird retains
81% of causal-dense: barely sparse, which makes its tie with dense weakly informative. At
context 1024 the patterns sit at 9% / 26% / 25%, where the question has teeth.
Extending the quality run there is the first thing I would add.

## 6. Which pattern loses what

**Sliding window** keeps a local band, so no dependency longer than `w` exists
*within a layer*: long-range information is relayed across layers, reach is
`O(w · depth)`, each hop lossy. It has no content-based recall at distance: a token
cannot look up something 3000 positions back however relevant, because that position
is not in its candidate set. It is also the cheapest, 1.2% density at N=8192.

**BigBird** adds **random** blocks, giving small-world connectivity (a few long
edges cut expected hop count from `O(N/w)` to `O(log N)`), and **global** blocks,
which matter more.

**Why global tokens matter out of proportion to their number.** A global block is
wired both ways: every query may read it, and it may read every key. That makes it a
*hub*, so any two tokens are at most two hops apart regardless of `N`. Nothing else
does that: window edges are local, random edges unreliable. One global block takes
the graph's diameter from `O(N/w)` to 2, at a cost of one block per row.

A second reason, which the companion Task 3 repository measures directly: the first
few positions absorb roughly a third of all attention mass *regardless of content*.
That is the attention sink. Softmax must sum to one, so a head with nothing it wants parks
its mass on the oldest visible position. Global blocks give that mass somewhere
legitimate to go; remove them and it is redistributed onto whatever remains,
distorting every other score in the row. Invisible if you only count edges.

**Dilated** reaches `O(2ⁿ)` distance with `n` taps at constant cost, so span grows
logarithmically, but coverage is ragged, and which blocks are reachable depends on
the query's index rather than on content. A cheap way to extend span, a bad way to
retrieve a specific fact.

## 7. Limitations

- **The gather duplicates K/V.** Slabs overlap between neighbouring query blocks. A
  fused kernel iterating K/V tiles in registers (Task 4) removes this; in PyTorch it
  can only be chunked.
- **Forward pass only**, as 1.5 specifies. The backward pass is implemented and
  tested but not benchmarked; autograd retains the softmax output, so its memory
  profile differs.
- **`block_size` fixed at 64**, untuned, and the obvious first autotuning knob.
- **The quality evaluation is small**: 2 layers, 445k params, one dataset, short
  context. It supports claims about this regime only.
