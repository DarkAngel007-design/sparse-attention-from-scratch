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
"""
import argparse, json, os, resource, sys, threading, time, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import torch
from sparseattn import block_sparse_attention, dense_attention, patterns as P
from sparseattn.blocksparse import peak_score_elements

ap = argparse.ArgumentParser()
ap.add_argument("--pattern", required=True)
ap.add_argument("--seq", type=int, required=True)
ap.add_argument("--device", default="cpu")
ap.add_argument("--batch", type=int, default=1)
ap.add_argument("--heads", type=int, default=8)
ap.add_argument("--dim", type=int, default=64)
ap.add_argument("--block", type=int, default=64)
ap.add_argument("--window", type=int, default=1)
ap.add_argument("--q-chunk", type=int, default=0)
ap.add_argument("--repeats", type=int, default=5)
ap.add_argument("--warmup", type=int, default=2)
ap.add_argument("--dtype", default="float32")
a = ap.parse_args()

dev = torch.device(a.device)
dtype = getattr(torch, a.dtype)
B, H, N, D = a.batch, a.heads, a.seq, a.dim
q_chunk = a.q_chunk or None


class PeakProbe:
    def __init__(self, device):
        self.device = device.type
        self.peak = 0
        self._stop = threading.Event()
        self._t = None

    def _sample_mps(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, torch.mps.current_allocated_memory())
            time.sleep(0.0005)

    def __enter__(self):
        if self.device == "cuda":
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        elif self.device == "mps":
            torch.mps.empty_cache()
            self.base = torch.mps.current_allocated_memory()
            self._t = threading.Thread(target=self._sample_mps, daemon=True); self._t.start()
        return self

    def __exit__(self, *exc):
        if self.device == "cuda":
            self.peak = torch.cuda.max_memory_allocated()
            self.method = "cuda.max_memory_allocated"
        elif self.device == "mps":
            torch.mps.synchronize()
            self.peak = max(self.peak, torch.mps.current_allocated_memory())
            self._stop.set(); self._t.join(timeout=1.0)
            self.method = "mps.current_allocated_memory sampled @2kHz"
        else:
            ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            self.peak = ru if sys.platform == "darwin" else ru * 1024  # macOS: bytes
            self.method = "ru_maxrss (whole process high-water)"
        return False


def sync():
    if dev.type == "cuda":
        torch.cuda.synchronize()
    elif dev.type == "mps":
        torch.mps.synchronize()


def build_pattern():
    kw = dict(causal=True, n_heads=1, device=dev)
    if a.pattern == "sliding_window":
        return P.sliding_window(N, a.block, a.window, **kw)
    if a.pattern == "bigbird":
        return P.bigbird(N, a.block, a.window, 1, 2, seed=0, **kw)
    if a.pattern == "dilated":
        return P.dilated(N, a.block, 4, **kw)
    raise KeyError(a.pattern)


out = dict(pattern=a.pattern, seq=N, device=a.device, batch=B, heads=H, dim=D,
           block=a.block, q_chunk=a.q_chunk, dtype=a.dtype)
try:
    g = torch.Generator(device="cpu").manual_seed(0)
    q, k, v = (torch.randn(B, H, N, D, generator=g).to(dev, dtype) for _ in range(3))

    if a.pattern == "dense":
        mask = torch.ones(N, N, dtype=torch.bool, device=dev).tril()
        fn = lambda: dense_attention(q, k, v, mask=mask)
        out["density"] = float(mask.float().mean())
        out["score_elems"] = B * H * N * N
    else:
        pat = build_pattern()
        fn = lambda: block_sparse_attention(q, k, v, pat, q_chunk=q_chunk)
        out["density"] = pat.token_density(N)
        out["score_elems"] = peak_score_elements(pat, N, B, H, q_chunk)

    with torch.no_grad():
        for _ in range(a.warmup):
            fn()
        sync()
        with PeakProbe(dev) as probe:
            times = []
            for _ in range(a.repeats):
                sync(); t0 = time.perf_counter()
                r = fn()
                sync(); times.append(time.perf_counter() - t0)
                del r
    times.sort()
    out.update(ok=True, ms_median=times[len(times)//2]*1e3, ms_min=times[0]*1e3,
               peak_bytes=probe.peak, peak_method=probe.method)
except (RuntimeError, torch.OutOfMemoryError) as e:
    msg = str(e)
    out.update(ok=False, error="OOM" if "out of memory" in msg.lower() or "MPS backend out of memory" in msg
               else type(e).__name__, message=msg[:200])

print(json.dumps(out))
