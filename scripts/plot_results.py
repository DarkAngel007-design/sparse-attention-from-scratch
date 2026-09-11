"""Plots for deliverables 1.5 and 1.6.

Run:  python scripts/plot_results.py
Reads results/benchmark.json and results/quality_*.json, writes results/plots/.
"""
import json, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = ROOT / "results" / "plots"
OUT.mkdir(parents=True, exist_ok=True)
COLORS = {"dense": "#1b1b1b", "sliding_window": "#0b7285",
          "bigbird": "#c2410c", "dilated": "#6d28d9"}
STYLE = {"dense": "-o", "sliding_window": "-s", "bigbird": "-^", "dilated": "-d"}


def hw_line(hw):
    if hw.get("gpu"):
        return f"{hw['gpu']} | torch {hw['torch']} | {hw['timestamp']}"
    return (f"{hw.get('cpu','?')} ({hw.get('cores','?')} cores, "
            f"{hw.get('ram_gb','?')} GB) | device={hw['device']} | "
            f"torch {hw['torch']} | {hw['timestamp']}")


def plot_benchmark():
    path = ROOT / "results" / "benchmark.json"
    if not path.exists():
        print("no benchmark.json, skipping"); return
    d = json.loads(path.read_text())
    rows = [r for r in d["rows"] if r.get("ok")]
    pats = sorted({r["pattern"] for r in rows}, key=lambda p: p != "dense")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for p in pats:
        rs = sorted([r for r in rows if r["pattern"] == p], key=lambda r: r["seq"])
        xs = [r["seq"] for r in rs]
        axes[0].plot(xs, [r["ms_median"] for r in rs], STYLE[p], color=COLORS[p], label=p)
        axes[1].plot(xs, [r["peak_bytes"]/1e6 for r in rs], STYLE[p], color=COLORS[p], label=p)
        axes[2].plot(xs, [r["score_elems"]/1e6 for r in rs], STYLE[p], color=COLORS[p], label=p)

    for ax, t, yl in zip(axes,
                         ["Forward-pass wall clock", "Measured peak memory",
                          "Largest score tensor (analytic)"],
                         ["milliseconds (median of 5)", "MB", "million elements"]):
        ax.set_xscale("log", base=2); ax.set_yscale("log", base=2)
        ax.set_xlabel("sequence length"); ax.set_ylabel(yl)
        ax.set_title(t); ax.grid(alpha=.3, which="both"); ax.legend(fontsize=8)

    # Reference slopes make the O(N) vs O(N^2) claim readable rather than asserted.
    xs = [r["seq"] for r in sorted(rows, key=lambda r: r["seq"])]
    x0, x1 = min(xs), max(xs)
    # Reference slopes are drawn slightly inside the data range and labelled at
    # their left end, so the annotation never collides with the N=8192 markers.
    xm = x1 / 1.7
    for ax, base in ((axes[1], 22), (axes[2], 1.4)):
        ax.plot([x0, xm], [base, base*(xm/x0)], ":", color="gray", lw=1.2)
        ax.plot([x0, xm], [base, base*(xm/x0)**2], "--", color="gray", lw=1.2)
        ax.text(x0*1.05, base*0.72, "O(N)", fontsize=8, color="gray")
        ax.text(x0*1.05, base*1.35, "O(N²)", fontsize=8, color="gray")

    fig.suptitle("Sparse vs dense attention, forward pass only\n" + hw_line(d["hardware"]),
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "benchmark.png", dpi=150)
    print("wrote", OUT / "benchmark.png")

    # Speedup / memory-saving ratios -- the number a reader actually wants.
    dense = {r["seq"]: r for r in rows if r["pattern"] == "dense"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for p in pats:
        if p == "dense":
            continue
        rs = sorted([r for r in rows if r["pattern"] == p and r["seq"] in dense],
                    key=lambda r: r["seq"])
        xs = [r["seq"] for r in rs]
        axes[0].plot(xs, [dense[r["seq"]]["ms_median"]/r["ms_median"] for r in rs],
                     STYLE[p], color=COLORS[p], label=p)
        axes[1].plot(xs, [dense[r["seq"]]["peak_bytes"]/r["peak_bytes"] for r in rs],
                     STYLE[p], color=COLORS[p], label=p)
    for ax, t in zip(axes, ["Speedup vs dense", "Peak-memory reduction vs dense"]):
        ax.axhline(1.0, color="k", lw=1, ls="--")
        ax.set_xscale("log", base=2); ax.set_xlabel("sequence length")
        ax.set_ylabel("x (higher is better)"); ax.set_title(t); ax.grid(alpha=.3)
        ax.legend(fontsize=8)
    axes[0].text(0.02, 0.92, "below the dashed line = sparse is SLOWER",
                 transform=axes[0].transAxes, fontsize=8, color="#b00")
    fig.suptitle("Relative to hand-written dense attention  |  " + hw_line(d["hardware"]),
                 fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "speedup.png", dpi=150)
    print("wrote", OUT / "speedup.png")


def plot_quality():
    paths = sorted((ROOT / "results").glob("quality*.json"))
    paths = [p for p in paths if p.name != "quality.json"] or paths
    if not paths:
        print("no quality json, skipping"); return
    fig, axes = plt.subplots(1, len(paths) + 1, figsize=(6.2 * (len(paths) + 1), 4.6),
                             squeeze=False)
    for ax, path in zip(axes[0], paths):
        d = json.loads(path.read_text())
        ctx = d["config"]["context"]
        n_seeds = len(d["config"].get("seeds", [1]))
        for name, r in d["results"].items():
            seeds = r.get("seeds")
            if seeds and len(seeds) > 1:
                # Plot the mean with a min-max band across seeds. Without the
                # band a 0.06 gap and a 0.002 gap look identical on the page.
                steps = [h["step"] for h in seeds[0]["history"]]
                curves = [[h["val"] for h in s_["history"]] for s_ in seeds]
                mean = [sum(c[i] for c in curves) / len(curves) for i in range(len(steps))]
                lo = [min(c[i] for c in curves) for i in range(len(steps))]
                hi = [max(c[i] for c in curves) for i in range(len(steps))]
                ax.fill_between(steps, lo, hi, color=COLORS.get(name), alpha=.18, lw=0)
                ax.plot(steps, mean, STYLE.get(name, "-o"), color=COLORS.get(name),
                        ms=4, label=f"{name} ({r['rel_density']:.0%} density)")
            else:
                h = r["history"]
                ax.plot([x["step"] for x in h], [x["val"] for x in h],
                        STYLE.get(name, "-o"), color=COLORS.get(name), ms=4,
                        label=f"{name} ({r['rel_density']:.0%} density)")
        ax.set_xlabel("step"); ax.set_ylabel("validation loss (nats/char)")
        ax.set_title(f"2-layer char GPT, TinyShakespeare, context={ctx}\n"
                     f"shaded = min-max over {n_seeds} seeds", fontsize=10)
        ax.grid(alpha=.3); ax.legend(fontsize=8)

    # Final loss against density, which is the comparison 1.6 is really asking for.
    ax = axes[0][-1]
    d = json.loads(paths[-1].read_text())
    for name, r in d["results"].items():
        lo, hi = r.get("val_min", r["final_val"]), r.get("val_max", r["final_val"])
        ax.errorbar([r["rel_density"] * 100], [r["final_val"]],
                    yerr=[[r["final_val"] - lo], [hi - r["final_val"]]],
                    fmt=STYLE.get(name, "o")[-1], color=COLORS.get(name),
                    ms=9, capsize=4, label=name)
        ax.annotate(name, (r["rel_density"] * 100, r["final_val"]), fontsize=8,
                    xytext=(6, 5), textcoords="offset points")
    ax.set_xlabel("token density (% of causal-dense)")
    ax.set_ylabel("final validation loss")
    ax.set_title("Final loss vs how much attention was kept\n"
                 "(error bars = min-max over seeds)", fontsize=10)
    ax.grid(alpha=.3)

    fig.tight_layout(); fig.savefig(OUT / "quality.png", dpi=150)
    print("wrote", OUT / "quality.png")


plot_benchmark()
plot_quality()
