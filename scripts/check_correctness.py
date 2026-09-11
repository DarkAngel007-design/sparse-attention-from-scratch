"""Deliverable 1.3: run every correctness check and report pass/fail.

Run:  python scripts/check_correctness.py [--device cpu|mps|cuda]
Exit code is non-zero if anything fails, so it works as a CI gate.
"""
import argparse, pathlib, sys, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from sparseattn.harness import all_checks

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="cpu")
ap.add_argument("-v", "--verbose", action="store_true")
a = ap.parse_args()

checks = all_checks(a.device)
print(f"running {len(checks)} checks on device={a.device}\n")
print(f"{'':4s} {'check':52s} detail")
print("-" * 118)
failed = []
t0 = time.time()
for c in checks:
    try:
        r = c.run()
    except Exception as e:                       # noqa: BLE001
        r = type("R", (), {"ok": False, "detail": f"EXCEPTION {type(e).__name__}: {e}"})()
    tag = "PASS" if r.ok else "FAIL"
    if not r.ok:
        failed.append(c.name)
    if a.verbose or not r.ok or not c.name.startswith("match/"):
        print(f"{tag:4s} {c.name:52s} {r.detail}")
    else:
        print(f"{tag:4s} {c.name:52s} {r.detail}")
print("-" * 118)
print(f"{len(checks) - len(failed)}/{len(checks)} passed in {time.time()-t0:.1f}s")
if failed:
    print("FAILED:", ", ".join(failed))
sys.exit(1 if failed else 0)
