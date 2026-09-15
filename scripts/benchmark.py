"""Deliverable 1.5: wall-clock and peak memory vs dense, N = 512 .. 8192.

Forward pass only, under torch.no_grad().  Each configuration runs in its own
subprocess (see _bench_one.py) so peak memory is a clean high-water mark.

Run:  python scripts/benchmark.py --device mps
      python scripts/benchmark.py --device cuda --seqs 512 1024 2048 4096 8192 16384
"""
import argparse, json, pathlib, platform, subprocess, sys, time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="cpu")
ap.add_argument("--seqs", type=int, nargs="+", default=[512, 1024, 2048, 4096, 8192])
ap.add_argument("--patterns", nargs="+",
                default=["dense", "sliding_window", "bigbird", "dilated"])
ap.add_argument("--heads", type=int, default=8)
ap.add_argument("--dim", type=int, default=64)
ap.add_argument("--block", type=int, default=64)
ap.add_argument("--repeats", type=int, default=5)
ap.add_argument("--out", default=str(ROOT / "results" / "benchmark.json"))
a = ap.parse_args()


def hardware():
    """Record what this ran on.  Common Requirement 3 asks for it because
    absolute milliseconds are meaningless on someone else's machine -- only the
    ratios transfer.  Goes into the JSON and onto every plot title."""
    import torch
    hw = dict(platform=platform.platform(), machine=platform.machine(),
              python=platform.python_version(), torch=torch.__version__,
              device=a.device, timestamp=time.strftime("%Y-%m-%d %H:%M:%S %Z"))
    if a.device == "cuda":
        hw["gpu"] = torch.cuda.get_device_name(0)
        p = torch.cuda.get_device_properties(0)
        hw["gpu_mem_gb"] = round(p.total_memory / 1e9, 1)
        hw["sm_count"] = p.multi_processor_count
    else:
        try:
            hw["cpu"] = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                       capture_output=True, text=True).stdout.strip()
            hw["ram_gb"] = round(int(subprocess.run(["sysctl", "-n", "hw.memsize"],
                                 capture_output=True, text=True).stdout) / 1e9, 1)
            hw["cores"] = int(subprocess.run(["sysctl", "-n", "hw.ncpu"],
                              capture_output=True, text=True).stdout)
        except Exception:
            pass
        if a.device == "mps":
            hw["accelerator"] = "Apple Metal (unified memory)"
    return hw


# Sweep N in the outer loop so partial output is comparable across patterns at
# the same length if the run is interrupted.
rows = []
print(f"{'pattern':16s} {'N':>6s} {'density':>8s} {'ms':>9s} {'peak MB':>9s} {'score elems':>13s}")
print("-" * 70)
for N in a.seqs:
    for pat in a.patterns:
        cmd = [sys.executable, str(ROOT / "scripts" / "_bench_one.py"),
               "--pattern", pat, "--seq", str(N), "--device", a.device,
               "--heads", str(a.heads), "--dim", str(a.dim),
               "--block", str(a.block), "--repeats", str(a.repeats)]
        # A FRESH SUBPROCESS per configuration.  Peak memory is a high-water
        # mark that never decreases, so sharing a process would let dense@8192's
        # 6.6 GB contaminate every later measurement.  It also isolates OOM: a
        # config that dies takes its process with it.
        p = subprocess.run(cmd, capture_output=True, text=True)
        # Take the LAST stdout line: warnings and progress bars may precede the
        # JSON, but the JSON is always printed last.
        line = p.stdout.strip().splitlines()[-1] if p.stdout.strip() else ""
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            # The child crashed before printing JSON (segfault, import error,
            # killed by the OOM killer).  Record it and keep sweeping.
            r = dict(pattern=pat, seq=N, ok=False, error="CRASH",
                     message=(p.stderr.strip()[-200:] or "no output"))
        rows.append(r)
        if r.get("ok"):
            print(f"{pat:16s} {N:6d} {r['density']:8.3f} {r['ms_median']:9.2f} "
                  f"{r['peak_bytes']/1e6:9.1f} {r['score_elems']:13,d}")
        else:
            print(f"{pat:16s} {N:6d} {'--':>8s} {r.get('error','?'):>9s} "
                  f"{'--':>9s} {'--':>13s}")

# hardware + full config + every row, including failed ones.  Committed to the
# repo so every number in WRITEUP.md can be checked without re-running a sweep.
#
# `out` is recorded RELATIVE to the repo root where possible: the default is an
# absolute path built from ROOT, and committing that would bake the author's
# home directory into a public artifact for no benefit.
cfg = vars(a).copy()
try:
    cfg["out"] = str(pathlib.Path(a.out).resolve().relative_to(ROOT))
except ValueError:
    pass                      # --out pointed outside the repo; leave it as given
out = dict(hardware=hardware(), config=cfg, rows=rows)
pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
pathlib.Path(a.out).write_text(json.dumps(out, indent=2))
print(f"\nwrote {a.out}")
print("hardware:", json.dumps(out["hardware"], indent=2))
