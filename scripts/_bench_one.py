"""Run ONE benchmark configuration and print a JSON line.

Run in a fresh subprocess per configuration so that peak-memory measurement is
a clean high-water mark rather than a residue of whatever ran before.

Peak memory has no portable API, so each device gets the best probe available
and the result records which one was used:
  cuda : torch.cuda.max_memory_allocated  -- exact, allocator-level
  mps  : a sampling thread polling torch.mps.current_allocated_memory, because
         torch.mps exposes no peak counter.  Approximate: a peak shorter than
         the sampling interval can be missed.
  cpu  : ru_maxrss of this process -- exact high-water RSS, which is why this
         has to be a subprocess.

Why any of this is necessary: docs/EXPLAINER.md Part 8.
"""
import argparse, json, os, resource, sys, threading, time, pathlib
# Put src/ on the path so the script runs WITHOUT `pip install -e .` -- useful on
# Colab where you would rather not install.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import torch
from sparseattn import block_sparse_attention, dense_attention, patterns as P
from sparseattn.blocksparse import peak_score_elements

ap = argparse.ArgumentParser()
ap.add_argument("--pattern", required=True)
ap.add_argument("--seq", type=int, required=True)
ap.add_argument("--device", default="cpu")
ap.add_argument("--batch", type=int, default=1)     # B=1 isolates sequence scaling
ap.add_argument("--heads", type=int, default=8)
ap.add_argument("--dim", type=int, default=64)      # standard head dim at this scale
ap.add_argument("--block", type=int, default=64)
ap.add_argument("--window", type=int, default=1)
ap.add_argument("--q-chunk", type=int, default=0)   # 0 means "no chunking"
ap.add_argument("--repeats", type=int, default=5)
ap.add_argument("--warmup", type=int, default=2)
ap.add_argument("--dtype", default="float32")
a = ap.parse_args()

dev = torch.device(a.device)
dtype = getattr(torch, a.dtype)                     # "float32" -> torch.float32
B, H, N, D = a.batch, a.heads, a.seq, a.dim
q_chunk = a.q_chunk or None                         # 0 -> None (falsy -> default)


class PeakProbe:
    """Context manager measuring peak memory, by whatever means the device allows.

    Used as `with PeakProbe(dev) as probe: ...` then `probe.peak`.  __enter__
    resets/starts the measurement, __exit__ collects it and records `method` so
    the JSON says HOW the number was obtained -- an approximate number labelled
    approximate is far more useful than one quoted as exact.
    """

    def __init__(self, device):
        self.device = device.type                   # "cuda" / "mps" / "cpu"
        self.peak = 0
        self._stop = threading.Event()              # signals the sampler to exit
        self._t = None

    def _sample_mps(self):
        # torch.mps has NO peak counter (check: [a for a in dir(torch.mps)
        # if "mem" in a]), only an instantaneous reading.  So poll it and keep
        # the max.  0.5 ms => 2 kHz.  A peak shorter than that can be missed;
        # this is the honest weak point of the MPS numbers, which is why the
        # analytic score-tensor size is reported alongside as a second,
        # independent line of evidence.
        while not self._stop.is_set():
            self.peak = max(self.peak, torch.mps.current_allocated_memory())
            time.sleep(0.0005)

    def __enter__(self):
        if self.device == "cuda":
            # Exact: the allocator itself tracks the high-water mark.  Reset so
            # we measure THIS region, not everything since process start.
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        elif self.device == "mps":
            torch.mps.empty_cache()
            self.base = torch.mps.current_allocated_memory()
            # daemon=True so a hung sampler can never block interpreter exit.
            self._t = threading.Thread(target=self._sample_mps, daemon=True); self._t.start()
        return self

    def __exit__(self, *exc):
        if self.device == "cuda":
            self.peak = torch.cuda.max_memory_allocated()
            self.method = "cuda.max_memory_allocated"
        elif self.device == "mps":
            # Synchronise first: GPU work may still be queued, and its
            # allocations would otherwise land after we stop sampling.
            torch.mps.synchronize()
            self.peak = max(self.peak, torch.mps.current_allocated_memory())
            self._stop.set(); self._t.join(timeout=1.0)
            self.method = "mps.current_allocated_memory sampled @2kHz"
        else:
            # ru_maxrss is the process's peak resident set size -- a high-water
            # mark that never decreases, which is exactly WHY each configuration
            # needs its own process.  Units differ by platform: macOS reports
            # bytes, Linux reports kilobytes.
            ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            self.peak = ru if sys.platform == "darwin" else ru * 1024  # macOS: bytes
            self.method = "ru_maxrss (whole process high-water)"
        return False                                # don't swallow exceptions


def sync():
    """Block until the GPU has actually finished.

    GPU calls are QUEUED, not executed -- control returns to Python immediately.
    Timing without this measures how fast Python can enqueue kernels, not how
    fast they run, and would report an enormous fake speedup.  CPU is
    synchronous already, so it needs nothing.
    """
    if dev.type == "cuda":
        torch.cuda.synchronize()
    elif dev.type == "mps":
        torch.mps.synchronize()


def build_pattern():
    # n_heads=1: one shared pattern broadcast to all heads.  The benchmark
    # measures the kernel, not per-head pattern diversity, and a shared pattern
    # keeps `expand` a view rather than a copy.
    kw = dict(causal=True, n_heads=1, device=dev)
    if a.pattern == "sliding_window":
        return P.sliding_window(N, a.block, a.window, **kw)
    if a.pattern == "bigbird":
        return P.bigbird(N, a.block, a.window, 1, 2, seed=0, **kw)
    if a.pattern == "dilated":
        return P.dilated(N, a.block, 4, **kw)
    raise KeyError(a.pattern)


# Everything that identifies this run goes in the output dict up front, so even
# a failed run reports what it was trying to do.
out = dict(pattern=a.pattern, seq=N, device=a.device, batch=B, heads=H, dim=D,
           block=a.block, q_chunk=a.q_chunk, dtype=a.dtype)
try:
    # Seeded on CPU then moved: identical inputs across devices and dtypes, so
    # runs are comparable and reproducible.
    g = torch.Generator(device="cpu").manual_seed(0)
    q, k, v = (torch.randn(B, H, N, D, generator=g).to(dev, dtype) for _ in range(3))

    if a.pattern == "dense":
        mask = torch.ones(N, N, dtype=torch.bool, device=dev).tril()
        fn = lambda: dense_attention(q, k, v, mask=mask)
        out["density"] = float(mask.float().mean())     # ~0.5 for causal
        out["score_elems"] = B * H * N * N               # the O(N^2) baseline
    else:
        pat = build_pattern()
        fn = lambda: block_sparse_attention(q, k, v, pat, q_chunk=q_chunk)
        out["density"] = pat.token_density(N)
        # ANALYTIC peak, reported next to the measured one.  Two independent
        # lines of evidence for the scaling claim.
        out["score_elems"] = peak_score_elements(pat, N, B, H, q_chunk)

    # no_grad: 1.5 says forward pass only.  With grad enabled, autograd retains
    # the softmax output and the memory profile is a different measurement.
    with torch.no_grad():
        # Warmup: the first call pays for kernel compilation, lazy library init
        # and allocator growth.  Including it would inflate the median.
        for _ in range(a.warmup):
            fn()
        sync()
        with PeakProbe(dev) as probe:
            times = []
            for _ in range(a.repeats):
                sync(); t0 = time.perf_counter()     # sync BEFORE starting the clock
                r = fn()
                sync(); times.append(time.perf_counter() - t0)   # and before stopping
                del r                                # free before the next iteration
    times.sort()
    # MEDIAN, not mean: robust to a single OS scheduling hiccup.  ms_min is kept
    # too, as a floor on what the hardware can do.
    out.update(ok=True, ms_median=times[len(times)//2]*1e3, ms_min=times[0]*1e3,
               peak_bytes=probe.peak, peak_method=probe.method)
except (RuntimeError, torch.OutOfMemoryError) as e:
    # A configuration that OOMs is DATA, not a crash: record it and let the
    # sweep continue.  Because this is a subprocess, an OOM here cannot poison
    # the memory measurement of any other configuration.
    msg = str(e)
    out.update(ok=False, error="OOM" if "out of memory" in msg.lower() or "MPS backend out of memory" in msg
               else type(e).__name__, message=msg[:200])

# One JSON line on stdout is the entire interface with benchmark.py.
print(json.dumps(out))
