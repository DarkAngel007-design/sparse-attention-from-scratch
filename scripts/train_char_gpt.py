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
    """Deterministic batch stream -- identical for every pattern.

    This is load-bearing for the experiment.  A LOCAL generator (not the global
    RNG) means the batch order depends only on `seed`, not on how many random
    numbers the model init happened to consume.  So every arm sees the same
    examples in the same order, and the mask is the only difference.
    """
    g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        # Sample random start offsets.  -context-1 so that i+context+1 is always
        # in range for the shifted target below.
        ix = torch.randint(len(split_ids) - context - 1, (batch,), generator=g)
        x = torch.stack([split_ids[i:i + context] for i in ix])          # inputs
        # Targets are inputs shifted by ONE: at every position, predict the next
        # character.  That single offset is the entire training signal.
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


@torch.no_grad()                # no gradients needed for eval: faster, less memory
def evaluate(model):
    # .eval() switches dropout/batchnorm to inference behaviour.  Here dropout is
    # 0.0 so it changes nothing -- but forgetting it is a classic silent bug, so
    # the habit is worth keeping.
    model.eval()
    tot = 0.0
    # seed=99, FIXED and independent of the training seed: every arm and every
    # eval point is scored on the identical set of validation batches.
    for x, y in batches(val_ids, a.eval_batches, a.batch, a.context, seed=99):
        tot += model(x, y)[1].item()
    model.train()               # back to training mode -- also easy to forget
    return tot / a.eval_batches


results = {}
for name in a.patterns:
  per_seed = []
  for seed in a.seeds:
    cfg = GPTConfig(vocab_size=len(chars), context=a.context)
    pat = make_pattern(name, cfg)
    dens = 1.0 if pat is None else pat.token_density(a.context)
    # Causal dense has density 0.5 by construction; report sparsity relative to it.
    # (Exactly (N+1)/2N: row i has i+1 allowed keys, summed over i and divided by
    # N^2.)  Quoting raw density against 1.0 would understate every pattern.
    dense_causal = 0.5 + 0.5 / a.context
    rel = dens / dense_causal if pat is not None else 1.0

    # THE CONTROL.  Seeding immediately before construction means every pattern
    # starts from bit-identical weights for a given seed.  Without this the
    # comparison is confounded by initialisation and the whole experiment is
    # uninterpretable.
    torch.manual_seed(seed)            # identical init across patterns, per seed
    model = CharGPT(cfg, pat).to(dev)
    # Sanity: a sparsity mask adds NO parameters, so this must be identical
    # across arms (445,184).  If it were not, the arms would have different
    # capacity and the loss comparison would mean nothing.
    nparam = sum(p.numel() for p in model.parameters())
    # AdamW: per-parameter adaptive step sizes from running averages of the
    # gradient (beta1=0.9) and its square (beta2=0.95).  The "W" is DECOUPLED
    # weight decay -- pulling weights toward zero separately from the gradient
    # step, which is the mathematically correct form of L2 for adaptive
    # optimisers.  betas/decay are the GPT-2 small conventions.
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.1,
                            betas=(0.9, 0.95))

    print(f"\n=== {name} seed={seed} === density={dens:.3f} "
          f"({rel:.1%} of causal-dense), params={nparam:,}")
    hist, t0 = [], time.time()
    for step, (x, y) in enumerate(batches(train_ids, a.steps, a.batch, a.context,
                                          seed=seed), start=1):
        _, loss = model(x, y)
        # MANDATORY: PyTorch ACCUMULATES into .grad.  Skip this and gradients sum
        # across steps, the effective learning rate grows without bound, and the
        # run diverges.  set_to_none=True frees the tensors instead of filling
        # them with zeros (slightly faster, and a forgotten backward shows up as
        # None rather than a stale zero).
        opt.zero_grad(set_to_none=True)
        loss.backward()                             # walk the graph, fill .grad
        # Rescale the whole gradient vector if its norm exceeds 1.0.  Preserves
        # DIRECTION, changes only magnitude -- stops one bad batch from blowing
        # up the weights.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()                                  # apply the update
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
    # THE SIGNIFICANCE GATE.  Compare the gap against the WIDER of the two arms'
    # seed spreads.  Crude -- it is a range, not a confidence interval -- but it
    # is enough to stop the headline claim being noise, which is exactly what
    # went wrong in the first single-seed version of this experiment.
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
