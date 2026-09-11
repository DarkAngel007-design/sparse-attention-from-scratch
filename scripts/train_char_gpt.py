"""Deliverable 1.6: does sparsity cost you anything in loss?

Trains the same 2-layer character GPT on TinyShakespeare once per attention
pattern.  Seed, initialisation, batch order, optimiser and step count are held
identical across runs; only the mask changes.

Run:  python scripts/train_char_gpt.py --device mps --steps 2000
"""
import argparse, json, pathlib, sys, time, urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
from sparseattn import patterns as P
from sparseattn.model import CharGPT, GPTConfig

DATA_URL = ("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/"
            "tinyshakespeare/input.txt")

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="cpu")
ap.add_argument("--steps", type=int, default=2000)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--context", type=int, default=256)
ap.add_argument("--block", type=int, default=32)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--eval-every", type=int, default=250)
ap.add_argument("--eval-batches", type=int, default=40)
ap.add_argument("--seeds", type=int, nargs="+", default=[1337],
                help="One run per seed. Multiple seeds let the writeup say "
                     "whether a loss gap is larger than run-to-run noise.")
ap.add_argument("--patterns", nargs="+",
                default=["dense", "sliding_window", "bigbird", "dilated"])
ap.add_argument("--out", default=str(ROOT / "results" / "quality.json"))
a = ap.parse_args()

data_path = ROOT / "data" / "tinyshakespeare.txt"
if not data_path.exists():
    data_path.parent.mkdir(parents=True, exist_ok=True)
    print("downloading TinyShakespeare ...")
    urllib.request.urlretrieve(DATA_URL, data_path)
text = data_path.read_text()
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
ids = torch.tensor([stoi[c] for c in text], dtype=torch.long)
n = int(0.9 * len(ids))
train_ids, val_ids = ids[:n], ids[n:]
print(f"corpus {len(ids):,} chars, vocab {len(chars)}, "
      f"train {len(train_ids):,} / val {len(val_ids):,}")

dev = torch.device(a.device)


def batches(split_ids, steps, batch, context, seed):
    """Deterministic batch stream -- identical for every pattern."""
    g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        ix = torch.randint(len(split_ids) - context - 1, (batch,), generator=g)
        x = torch.stack([split_ids[i:i + context] for i in ix])
        y = torch.stack([split_ids[i + 1:i + 1 + context] for i in ix])
        yield x.to(dev), y.to(dev)


def make_pattern(name, cfg):
    if name == "dense":
        return None
    kw = dict(causal=True, n_heads=cfg.n_head, device=dev)
    if name == "sliding_window":
        return P.sliding_window(cfg.context, a.block, 1, **kw)
    if name == "bigbird":
        return P.bigbird(cfg.context, a.block, 1, 1, 2, seed=0, **kw)
    if name == "dilated":
        return P.dilated(cfg.context, a.block, 4, **kw)
    raise KeyError(name)


@torch.no_grad()
def evaluate(model):
    model.eval()
    tot = 0.0
    for x, y in batches(val_ids, a.eval_batches, a.batch, a.context, seed=99):
        tot += model(x, y)[1].item()
    model.train()
    return tot / a.eval_batches


results = {}
for name in a.patterns:
  per_seed = []
  for seed in a.seeds:
    cfg = GPTConfig(vocab_size=len(chars), context=a.context)
    pat = make_pattern(name, cfg)
    dens = 1.0 if pat is None else pat.token_density(a.context)
    # Causal dense has density 0.5 by construction; report sparsity relative to it.
    dense_causal = 0.5 + 0.5 / a.context
    rel = dens / dense_causal if pat is not None else 1.0

    torch.manual_seed(seed)            # identical init across patterns, per seed
    model = CharGPT(cfg, pat).to(dev)
    nparam = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.1,
                            betas=(0.9, 0.95))

    print(f"\n=== {name} seed={seed} === density={dens:.3f} "
          f"({rel:.1%} of causal-dense), params={nparam:,}")
    hist, t0 = [], time.time()
    for step, (x, y) in enumerate(batches(train_ids, a.steps, a.batch, a.context,
                                          seed=seed), start=1):
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % a.eval_every == 0 or step == a.steps:
            vl = evaluate(model)
            hist.append(dict(step=step, train=loss.item(), val=vl))
            print(f"  step {step:5d}  train {loss.item():.4f}  val {vl:.4f}  "
                  f"({time.time()-t0:.0f}s)")
    per_seed.append(dict(seed=seed, final_val=hist[-1]["val"], history=hist,
                         seconds=time.time() - t0))
  vals = [r["final_val"] for r in per_seed]
  results[name] = dict(density=dens, rel_density=rel, params=nparam,
                       final_val=sum(vals) / len(vals),
                       val_min=min(vals), val_max=max(vals),
                       val_spread=max(vals) - min(vals),
                       seeds=per_seed, history=per_seed[0]["history"],
                       seconds=sum(r["seconds"] for r in per_seed))

print(f"\n{'pattern':16s} {'density':>8s} {'vs dense':>9s} {'val loss':>9s} "
      f"{'spread':>8s} {'delta':>8s} {'signif?':>8s}")
print("-" * 74)
base = results.get("dense", {})
b, bspread = base.get("final_val"), base.get("val_spread", 0.0)
for k, r in results.items():
    d = r['final_val'] - b if b else 0.0
    # A gap is only worth talking about if it is bigger than the seed-to-seed
    # spread of the two arms it is drawn from.
    noise = max(bspread, r["val_spread"])
    sig = "--" if k == "dense" else ("yes" if abs(d) > noise else "NO")
    print(f"{k:16s} {r['density']:8.3f} {r['rel_density']:9.1%} "
          f"{r['final_val']:9.4f} {r['val_spread']:8.4f} {d:+8.4f} {sig:>8s}")
print(f"\n{len(a.seeds)} seed(s) per pattern; 'val loss' is the mean, 'spread' is "
      f"max-min across seeds.\nA delta smaller than the spread is not a result.")

pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
pathlib.Path(a.out).write_text(json.dumps(
    dict(config=vars(a), results=results), indent=2))
print(f"\nwrote {a.out}")
