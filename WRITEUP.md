# Sparse Attention from Scratch — Writeup

**Deliverable 1.7.** Hardware for every number below: Apple M5 Pro, 18 cores,
25.8 GB unified memory, PyTorch 2.14.0 on the MPS backend. Absolute
milliseconds are not portable; the ratios are the claim.

---

## 1. What was built, and the one design decision everything follows from

A sparsity pattern is a **block mask** of shape `(H, NQB, NKB)`: "may query
block *i* of head *h* read key block *j*". That one object has two consumers.
`to_dense_mask` expands it to a token mask that drives the hand-written dense
reference; `to_gather_index` compacts it to gather indices that drive the fast
path. Because both are derived from the same source, the harness cannot drift
from the thing it is testing.

Sparsity is expressed at **block** granularity rather than token granularity
because token-level sparsity buys nothing. A token-level mask still has to be
applied to an `N×N` score matrix, so you pay the full quadratic memory in order
to throw most of it away. Sparsity only becomes an optimisation when it is
structured enough that whole contiguous slabs of K/V can be skipped. This is
why BigBird and Longformer are block algorithms and not token algorithms.

The kernel in `blocksparse.py` therefore never materialises `N×N`. Its largest
tensor is `(B, H, NQB, bs, K·bs)`, where `K` is the number of key blocks a query
block selects. At `N = 8192`, `bs = 64`, `K = 6` that is a 21× smaller score
tensor.

What this is **not** is a fused kernel. The gather physically copies K and V,
and for a sliding window those copies overlap — block *i* is gathered again by
query block *i+1* and again by *i+2*. So the score matrix shrinks while K/V
memory traffic grows. That trade is the honest ceiling on what can be done in
PyTorch, and it is precisely what a Triton flash-attention kernel (Task 4 of
this same sheet) removes.

## 2. Correctness (1.3)

72 checks, `python scripts/check_correctness.py`, all passing. Every sparse
variant is compared against **dense attention run with the same pattern
expanded to a token mask** — that is the only comparison that means anything.
"Matches dense on unmasked positions" is not a statement about *unmasked* dense
attention; it is the claim that the fast path reproduces `softmax(QKᵀ + mask)V`
exactly. It does, bit-exactly, in both float32 and float64.

The checks that earned their place:

- **A full-density non-causal pattern must reproduce plain `softmax(QKᵀ)V`.**
  A systematically wrong gather would still agree with a mask built the same
  wrong way. It could not agree with unmasked dense attention.
- **`q_chunk` invariance.** Peak memory is traded against chunk size; the
  numerics must not move. They don't — bit-identical for chunk ∈ {1,2,3,8}.
- **Gather-index exactness.** Query blocks select different *numbers* of key
  blocks, but a batched gather needs a fixed `K`. Padding short rows with a
  repeated real index is the tempting fix, and it is wrong: the duplicated block
  enters the softmax twice and is weighted twice. The output stays finite,
  smooth, and trainable — to the wrong answer. Hence a separate validity mask,
  and a check that asserts the gathered set equals the pattern exactly.
- **Backward-pass equivalence**, max |dQ, dK, dV error| = 6.2e-15. A kernel can
  be right forward and wrong backward; only the forward pass is ever inspected.
- **Degenerate all-equal logits** must give exactly uniform weights over the
  allowed set.

One check initially failed and was instructive. Under adversarial magnitudes
(q,k scaled ×30, logits ≈ ±4800) the sparse output differed from the float64
dense reference by 9.2e-3. That is not a kernel error: float32 sparse versus
**float32** dense is bit-exact, and the 9.2e-3 is float32 diverging from
float64 in the reference itself. One float32 ulp at |logit| ≈ 4800 is 2.9e-4;
softmax inherits that as relative error; `|v| ≈ 30` scales it into the output as
≈ 8.7e-3, which is what was measured. The check now asserts the kernel claim and
*reports* the precision claim separately — folding the second into the pass
criterion would make the harness fail for a reason unrelated to the code under
test.

## 3. The NaN (1.4)

`softmax` subtracts the row maximum for stability: `exp(xᵢ − max)`. If a query's
entire allowed set is masked, every logit in the row is `−inf`, so `max` is
`−inf`, and `−inf − (−inf) = nan`. `exp(nan) = nan`. The row is NaN before any
division happens, and NaN propagates through everything downstream.

**When it arises for real.** Not from hand-crafted masks. `scripts/demo_nan.py`
produces it from a causal sliding window with one wrong comparison in the window
bound — a window that excludes its own block. Query block 0 then has no earlier
block to look at and nothing of its own, so all 64 queries in it are dead. At
block granularity, an off-by-one takes out a whole block at a time. The general
shape: any pattern where a query's selected key blocks all lie strictly in its
future under causal masking.

**Why patching afterwards is not a fix.** `torch.nan_to_num` is the obvious
repair. Whether it works depends on two choices made elsewhere in the file.
Measured:

| mask applied via | `nan_to_num` at | forward | dQ | dV |
| --- | --- | --- | --- | --- |
| `masked_fill` | attention weights | clean | clean | clean |
| `masked_fill` | output | clean | clean | **NaN** |
| additive `−inf` bias | attention weights | clean | **NaN** | clean |
| additive `−inf` bias | output | clean | **NaN** | **NaN** |

Read the forward column first: clean in every row. That is the trap — the NaN is
gone from the values you can print, and the gradient is poisoned anyway. Loss
prints as a finite number, parameters turn to NaN, and the run dies hundreds of
steps later with no clue why.

Three of the four combinations corrupt a gradient. The one that survives does so
**by accident**: `masked_fill`'s backward zeroes the gradient at masked
positions, which happens to kill the NaN on its way to Q. Swap it for an
additive `−inf` bias — which is what most production attention code uses,
because it composes with other biases — and the accident disappears. Nobody
writing the `nan_to_num` line is thinking about which of those four squares they
are standing in.

**The fix used here** removes the dependency: repair *before* the softmax. Dead
rows are given a finite set of logits (they are allowed to attend to everything,
which makes the softmax well-defined and differentiable), and their weights are
then zeroed. No NaN is ever constructed, so there is nothing to patch, and the
gradient contribution of a dead row is exactly zero rather than exactly wrong.

Three policies are offered because the right answer depends on the caller.
`zero` emits the zero vector — correct for a padded encoder position whose
output is discarded. `self` falls back to attending to the query's own position
— correct for a decoder, where a zero vector mid-sequence is a hole the next
layer must absorb. `raise` is correct while developing: a dead row almost always
means the pattern is wrong.

`policy="self"` produced a real bug worth recording. It originally located the
query's own position with `torch.eye`, which is only the diagonal in **dense**
coordinates. Inside the block-sparse kernel the last axis is *gathered key
slots*, whose absolute positions are scattered, and the identity matrix there is
meaningless. `masked_softmax` now requires the caller to supply the fallback
mask (`key_position == query_position`), and raises if the fallback fails to
revive every dead row rather than assuming a layout.

## 4. Benchmark (1.5)

Forward pass only, `torch.no_grad()`, B=1, H=8, D=64, block size 64, median of
5 runs. Each configuration runs in a **fresh subprocess** so peak memory is a
clean high-water mark rather than a residue of the previous configuration.

| N | dense ms | dense MB | sliding ms | sliding MB | BigBird ms | BigBird MB |
| --- | --- | --- | --- | --- | --- | --- |
| 512 | 1.68 | 28.8 | 1.40 | 14.8 | 1.88 | 32.2 |
| 1024 | 4.96 | 109.1 | 3.42 | 29.5 | 4.79 | 64.4 |
| 2048 | 10.45 | 557.8 | 3.03 | 59.1 | 5.23 | 128.7 |
| 4096 | 40.20 | 1669.3 | 6.31 | 118.1 | 8.35 | 257.4 |
| 8192 | 159.12 | 6627.0 | 7.42 | 236.2 | 16.97 | 514.8 |

At N=8192 the sliding window is **21× faster and uses 28× less memory**;
BigBird is 9.4× faster and 12.9× lighter.

Peak memory has no portable API, so the probe is recorded with the result: CUDA
uses `max_memory_allocated` (exact), CPU uses `ru_maxrss` (exact, which is why
each run is a subprocess), and MPS uses a 2 kHz sampling thread on
`current_allocated_memory`, because `torch.mps` exposes no peak counter at all.
The analytic score-tensor size is reported alongside the measurement so the
scaling claim can be checked two independent ways; they agree.

**Where sparse loses.** At N=512, BigBird (1.88 ms) and dilated (1.82 ms) are
both *slower* than dense (1.68 ms). The gather is not free: it copies K and V,
and at short sequences that memory traffic exceeds the FLOPs saved by skipping
blocks. Dense attention at 512 is a single large well-shaped matmul, which is
the case hardware is best at. The crossover is around N≈1024 here. Anyone
reaching for sparse attention below a couple of thousand tokens is paying
complexity for a slowdown.

## 5. Quality (1.6)

Same 2-layer character GPT (445,184 parameters), TinyShakespeare, context 256,
attention block size 32. Seed, initialisation, batch order, optimiser and step
count are identical across patterns; **only the mask changes**. Three seeds per
pattern, 8000 steps, so that a gap can be compared against run-to-run noise
rather than asserted.

| pattern | token density | final val loss | seed spread | vs dense | beyond noise? |
| --- | --- | --- | --- | --- | --- |
| dense | 100% | 1.5485 | 0.0023 | — | — |
| sliding window | 34.6% | 1.5601 | 0.0097 | +0.0116 | yes |
| BigBird | 81.3% | 1.5496 | 0.0050 | +0.0010 | no |
| dilated | 65.8% | 1.5433 | 0.0112 | −0.0052 | no |

The headline: **BigBird and dilated are statistically indistinguishable from
dense**, and sliding window costs +0.0116 nats/char — a real difference, about
five times its own seed spread, but only 0.75% in relative terms while using a
third of the attention.

### The result reverses depending on where you stop, and that is the finding

An earlier version of this experiment ran 4000 steps with one seed and concluded
that every sparse pattern **beat** dense. That was not noise — the gap was 0.064,
roughly 28× the seed spread — but it was a measurement of a model that had not
converged. Running to 8000 steps reverses the ranking:

| | step 4000 | step 8000 | best val (and when) | final train loss |
| --- | --- | --- | --- | --- |
| dense | 1.5882 | **1.5485** | 1.5479 @ 7000 | 1.2579 |
| sliding window | **1.5406** | 1.5601 | 1.5406 @ 4000 | 1.1759 |
| dilated | — | 1.5433 | **1.5340 @ 6000** | 1.2007 |

Dense overtakes sliding window at **step 7000**. Two separate effects produce
that crossing, and they point in opposite directions:

**Sparse converges faster.** A constrained attention pattern is a smaller
hypothesis space, and for character-level text it is a well-matched one —
spelling and local syntax live inside a few dozen characters. The dense model has
to *learn* to ignore distant tokens; the sliding-window model is told. At a fixed
step budget that is a large advantage.

**Sparse then overfits harder.** Sliding window ends with the **lowest training
loss of all four** (1.1759 vs dense's 1.2579) and the **highest validation
loss**. It is not underfitting for lack of capacity — it is fitting the training
data better and generalising worse. Its validation loss regresses by +0.0195
after step 4000, while dense's is flat (+0.0007).

So "does sparsity cost you loss?" has no single answer here; it has an answer per
training budget. Reporting one number at one step count — which is what the
deliverable literally asks for — would have supported either conclusion depending
on which step I picked. The multi-seed, longer run is what made that visible.

### What this does and does not show

It does **not** show that sparse attention is free in general. It shows that *for
this task* the long-range attention dense has access to is not carrying much
loss. Character-level Shakespeare at 256 tokens of context is dominated by local
structure, so removing long-range edges removes capacity the model was barely
using. A task built around long-range retrieval would separate these patterns
immediately — and the companion Task 3 repository demonstrates exactly that:
under a fixed cache budget, policies that keep only local context score
respectable perplexity and retrieve a planted fact **0%** of the time.

One honest caveat about the configuration: at context 256 with block size 32
there are only 8 blocks, so BigBird retains 81% of causal-dense — it is barely
sparse. That is why its tie with dense is unsurprising and weakly informative.
The same patterns at context 1024 would sit at 9% / 26% / 25% density, which is
where the question has teeth. The benchmark in §4 runs at those lengths; the
quality experiment does not, and extending it is the first thing I would add.

## 6. Which pattern loses what

**Sliding window** keeps only a local band. It cannot represent any dependency
longer than `w` tokens *within a layer*; long-range information moves only by
being relayed through successive layers, so the reachable distance is `O(w · depth)`
and each hop is lossy. It has no mechanism for content-based recall at
distance — a token cannot look up something specific 3000 positions back, no
matter how relevant, because that position is simply not in its candidate set.
It is also the cheapest and, at 1.2% token density at N=8192, by far the
lightest.

**BigBird** adds two things to that band. The **random** blocks give the
attention graph small-world connectivity: a handful of random long edges cuts
the expected hop count between any two tokens from `O(N/w)` to `O(log N)`, which
is what the original paper's Turing-completeness argument leans on. The
**global** blocks are more important, and disproportionately so.

**Why global tokens matter out of proportion to their number.** A global block
is wired both ways: every query may read it, and it may read every key. That
makes it a *hub*. With `g` global blocks, any two tokens in the sequence are at
most two hops apart — up into a hub, and back down — regardless of `N`. Nothing
else in the pattern does that; window edges are local and random edges are
sparse and unreliable. So a single global block changes the diameter of the
attention graph from `O(N/w)` to 2, at a cost of one block in every row.

There is a second reason, which the Task 3 work in the companion repository
measures directly: the first few positions of a real sequence absorb roughly a
third of all attention mass regardless of content — the attention sink. Softmax
must sum to one, so a head with nothing it wants parks its mass on the oldest
visible position. Global blocks at the start of the sequence give that mass
somewhere legitimate to go. Remove them and the mass does not disappear; it is
redistributed onto whatever remains, distorting every other score in the row.
That is a real effect on quality, and it is invisible if you only count edges.

**Dilated** (the third pattern) reaches `O(2ⁿ)` distance with `n` taps per
query, so its span grows logarithmically at constant cost. But its coverage is
ragged: most key blocks are unreachable in one hop, and which ones are reachable
depends on the query's index rather than on anything about the content. It is a
good cheap way to extend span and a bad way to retrieve a specific fact.

## 7. Limitations, and what I would do next

- **The gather duplicates K/V.** The score matrix shrinks as intended, but the
  gathered K/V slabs overlap between neighbouring query blocks. A fused kernel
  that iterates over K/V tiles in registers (Task 4) removes this entirely; in
  PyTorch it cannot be removed, only chunked.
- **Forward pass only**, as 1.5 specifies. The backward pass is implemented and
  tested for correctness, but not benchmarked — its memory profile is different,
  since autograd retains the softmax output.
- **`block_size` was not tuned.** It is fixed at 64 throughout. It trades
  pattern resolution against gather efficiency and is the obvious first
  autotuning knob.
- **The quality evaluation is small** — 2 layers, 445k parameters, one dataset.
  It supports claims about this regime and should not be extrapolated to
  models where long-range dependency actually carries the loss.
