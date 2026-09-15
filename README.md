# Sparse Attention from Scratch

Dense attention written out by hand, two block-sparsity patterns (plus a third)
built on top of it, and an honest account of what each one costs.

Postman AI/ML recruitment task 25, Task 1.

`F.scaled_dot_product_attention` is never called. The hand-written dense path in
[`src/sparseattn/dense.py`](src/sparseattn/dense.py) is the reference that every
sparse variant is checked against.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

```bash
python scripts/check_correctness.py --device cpu    # 72 checks, exits non-zero on failure
```

```bash
python scripts/demo_nan.py                          # deliverable 1.4, end to end
```

```bash
python scripts/benchmark.py --device mps --seqs 512 1024 2048 4096 8192
```

```bash
python scripts/train_char_gpt.py --device mps --steps 4000
```

```bash
python scripts/plot_results.py                      # writes results/plots/
```

On Colab, pass `--device cuda`. On a CPU-only machine, `--device cpu` works for
everything; drop the top sequence length or two from the benchmark.

## Repository layout

| Path | What it is |
| --- | --- |
| `src/sparseattn/dense.py` | Hand-written dense attention and the masked softmax where the NaN of 1.4 lives |
| `src/sparseattn/patterns.py` | Block-level sparsity patterns: sliding window, BigBird, dilated, full |
| `src/sparseattn/blocksparse.py` | The gather-based kernel that never materialises the N×N score matrix |
| `src/sparseattn/harness.py` | Every correctness check, shared by the pytest suite and the standalone script |
| `src/sparseattn/model.py` | 2-layer character GPT with a swappable attention backend |
| `scripts/check_correctness.py` | Deliverable 1.3: pass/fail report |
| `scripts/demo_nan.py` | Deliverable 1.4: the NaN, why the obvious fix isn't one, and the policies |
| `scripts/benchmark.py` | Deliverable 1.5: wall clock and peak memory, 512 → 8192 |
| `scripts/train_char_gpt.py` | Deliverable 1.6: dense vs each pattern on TinyShakespeare |
| `scripts/plot_results.py` | Plots for 1.5 and 1.6 |
| `WRITEUP.md` | Deliverable 1.7 |

Every source file is commented for a reader coming to it cold, explaining not
just what a line does but why the alternative was rejected and what breaks if it
changes. [`src/sparseattn/dense.py`](src/sparseattn/dense.py) is the place to
start; it is the reference implementation everything else is checked against.

## Deliverables

| Item | Where | Status |
| --- | --- | --- |
| 1.1 Manual dense attention | `dense.py::dense_attention` | done |
| 1.2 Two sparsity patterns | `patterns.py`: sliding window, BigBird (local+global+random) | done, plus a third (dilated) |
| 1.3 Correctness harness | `scripts/check_correctness.py`, 72 checks | done |
| 1.4 NaN handling | `dense.py::masked_softmax`, `scripts/demo_nan.py` | done |
| 1.5 Benchmark | `scripts/benchmark.py`, 512 → 8192 | done |
| 1.6 Quality evaluation | `scripts/train_char_gpt.py` | done |
| 1.7 Writeup | `WRITEUP.md` | done |
| Stretch: third pattern | `patterns.py::dilated` | done |
| Stretch: per-head pattern mixing | `patterns.py::bigbird(per_head_random=True)` | done |

## Results at a glance

**Speed and memory** (forward pass, B=1 H=8 D=64, block 64, median of 5):

| N | dense | sliding window | BigBird |
| --- | --- | --- | --- |
| 512 | 1.68 ms / 28.8 MB | 1.40 ms / 14.8 MB | 1.88 ms / 32.2 MB |
| 2048 | 10.45 ms / 557.8 MB | 3.03 ms / 59.1 MB | 5.23 ms / 128.7 MB |
| 8192 | 159.12 ms / 6627.0 MB | **7.42 ms / 236.2 MB** | 16.97 ms / 514.8 MB |

21× faster and 28× lighter at N=8192 for the sliding window. **But at N=512 both
BigBird and dilated are slower than dense.** The gather copies K and V, and
below ~1k tokens that memory traffic costs more than the skipped FLOPs buy.

**Quality** (2-layer char GPT, TinyShakespeare, 3 seeds, 8000 steps):

| pattern | density | val loss | seed spread | vs dense | beyond noise? |
| --- | --- | --- | --- | --- | --- |
| dense | 100% | 1.5485 | 0.0023 | n/a | n/a |
| sliding window | 34.6% | 1.5601 | 0.0097 | +0.0116 | yes |
| BigBird | 81.3% | 1.5496 | 0.0050 | +0.0010 | no |
| dilated | 65.8% | 1.5433 | 0.0112 | −0.0052 | no |

The ranking depends on when you stop. At step 4000 every sparse pattern *beat*
dense; dense overtakes sliding window at step 7000. Sparse converges faster
(smaller, well-matched hypothesis space) and then overfits harder (sliding window
ends with the lowest **training** loss of all four and the highest validation
loss). `WRITEUP.md` §5 has the full account. It is the most interesting result
in the repository and it contradicts the first version of this experiment.

## Design in one paragraph

A sparsity pattern is a block mask of shape `(H, NQB, NKB)`: "may query block
*i* of head *h* read key block *j*". That single object has two consumers:
`to_dense_mask` expands it to tokens to drive the dense reference, and
`to_gather_index` compacts it to gather indices to drive the kernel. Because
both come from the same source, the harness cannot drift from the thing it is
testing. Sparsity is expressed at block granularity rather than token
granularity for the obvious reason: a token-level mask still has to be applied
to an N×N matrix, so it saves nothing.

## Hardware

Benchmarks in `results/` were produced on an **Apple M5 Pro, 18 cores, 25.8 GB
unified memory, PyTorch 2.14.0 on the MPS backend**. Every number reported is
relative; absolute milliseconds mean nothing on someone else's machine. The
benchmark script records its own hardware into `results/benchmark.json` and
prints it on the plots.

Peak memory has no portable API, so each device gets the best probe available
and the result records which one was used:

| Device | Probe | Exact? |
| --- | --- | --- |
| CUDA | `torch.cuda.max_memory_allocated` | yes, allocator-level |
| MPS | sampling thread on `torch.mps.current_allocated_memory` @2 kHz | approximate, `torch.mps` exposes no peak counter |
| CPU | `ru_maxrss` of the subprocess | exact process high-water mark |

Each configuration runs in a **fresh subprocess** so the high-water mark is
clean rather than a residue of the previous configuration. The analytic peak
score-tensor size is reported next to the measured number, so the O(N) vs O(N²)
claim can be checked against theory and measurement independently.
