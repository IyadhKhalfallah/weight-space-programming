"""Causal search loop: find an edit that unblocks Qwen's decision to stop thinking.

No peeking at Astra's edit. Training problems only. Forward passes only (no gradients, no training).
  1. propose: per layer, 4 candidate directions from what the MLP writes during rambling
  2. test:    remove the direction from that layer's MLP output, replay 6 rambling runs, measure
              - stop score: log of mean P(</think>) over thinking positions (higher = more willing to stop)
              - damage:     increase in loss (NLL) on the model's own text (reject if > 0.05 nats)
  3. stack:   greedily add directions from other layers while the stop score keeps improving
  4. write:   W_down <- (I - v v^T) W_down for each chosen layer -> models/loop
"""
import numpy as np, mlx.core as mx, mlx.nn as nn, json, shutil, time
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
from datasets import load_dataset

mx.set_memory_limit(6 * 1024 ** 3); mx.set_cache_limit(1024 ** 3)
t0 = time.time()
log = open("results/search.log", "w")
def say(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); log.write(s + "\n"); log.flush()

m, tok = load("models/Qwen3.5-0.8B")
layers = m.language_model.model.layers; NL = len(layers)
gens = json.load(open("results/diag_gens.json"))  # 8 base-model runs on GSM8K *train* problems
qs = load_dataset("openai/gsm8k", "main", split="train").shuffle(seed=1).select(range(8))["question"]
prompts = [tok.apply_chat_template([{"role": "user", "content": q + "\nPlease reason step by step, and put your final answer within \\boxed{}."}],
                                   add_generation_prompt=True, enable_thinking=True, tokenize=True) for q in qs]
runs = [(p, g) for p, g in zip(prompts, gens) if len(g) >= 4096][:6]   # rambling runs only
END = tok.convert_tokens_to_ids("</think>")
say(f"using {len(runs)} rambling runs")


class Hook(nn.Module):
    """wraps an MLP; can capture its output and/or project out a set of directions."""
    def __init__(self, mlp): super().__init__(); self.mlp = mlp; self.V = None; self.capture = False; self.buf = []
    def __call__(self, x):
        o = self.mlp(x)
        if self.V is not None: o = o - (o @ self.V.T) @ self.V
        if self.capture: self.buf.append(o[0].astype(mx.float32))
        return o
hooks = []
for l in layers:
    h = Hook(l.mlp); l.mlp = h; hooks.append(h)


def replay(capture=False):
    """returns (stop score, mean NLL) over thinking positions of all runs."""
    pe, nll = [], []
    for p, g in runs:
        seq = p + g; cache = make_prompt_cache(m); P0 = len(p)
        for h in hooks: h.capture = capture
        for s in range(0, len(seq), 512):
            ch = seq[s:s + 512]
            lg = m(mx.array([ch]), cache=cache)[0].astype(mx.float32)
            lp = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
            nxt = mx.array(seq[s + 1:s + 1 + len(ch)] + [0] * (len(ch) - len(seq[s + 1:s + 1 + len(ch)])))
            tgt = mx.take_along_axis(lp, nxt[:, None], axis=-1)[:, 0]
            e = mx.exp(lp[:, END]); mx.eval(e, tgt)
            idx = np.arange(s, s + len(ch)); keep = (idx >= P0) & (idx < len(seq) - 1)
            pe += list(np.array(e)[keep]); nll += list(-np.array(tgt)[keep])
        mx.clear_cache()
    for h in hooks: h.capture = False
    return float(np.log(np.mean(pe))), float(np.mean(nll))


def set_edit(edit):  # edit: {layer: [v, ...]}
    for L, h in enumerate(hooks):
        h.V = mx.array(np.stack(edit[L]).astype(np.float32)).astype(mx.bfloat16) if L in edit else None


# ---- baseline + capture MLP outputs for candidate proposals
set_edit({})
base_stop, base_nll = replay(capture=True)
say(f"base: stop score {base_stop:.3f}  nll {base_nll:.3f}   ({time.time() - t0:.0f}s)")
cands = []
lens = [len(p) + len(g) for p, g in runs]
for L, h in enumerate(hooks):
    out = np.concatenate([np.array(b) for b in h.buf])  # rows in run order
    rows_e, rows_l, off = [], [], 0
    for (p, g), n in zip(runs, lens):
        P0 = len(p); rows_e.append(out[off + P0 + 50: off + P0 + 650]); rows_l.append(out[off + P0 + 2500: off + n]); off += n
    E, Lt = np.concatenate(rows_e), np.concatenate(rows_l)
    d = Lt.mean(0) - E.mean(0); cands.append((L, "diffmean", d / np.linalg.norm(d)))
    sub = Lt[np.random.default_rng(L).choice(len(Lt), min(3000, len(Lt)), replace=False)]
    _, _, vt = np.linalg.svd(sub, full_matrices=False)
    for j in range(3): cands.append((L, f"pc{j}", vt[j]))
    h.buf = []
say(f"{len(cands)} candidates proposed ({time.time() - t0:.0f}s)")

# ---- test every candidate causally
res = []
for i, (L, kind, v) in enumerate(cands):
    set_edit({L: [v]}); s, n = replay()
    res.append(dict(L=L, kind=kind, dstop=s - base_stop, dnll=n - base_nll, v=v))
    say(f"  [{i + 1:3d}/{len(cands)}] layer {L:2d} {kind:8s}  Δstop {s - base_stop:+.3f}  Δnll {n - base_nll:+.4f}")
ok = sorted([r for r in res if r["dnll"] < 0.05], key=lambda r: -r["dstop"])
say("\ntop 8 safe candidates:")
for r in ok[:8]: say(f"  layer {r['L']:2d} {r['kind']:8s} Δstop {r['dstop']:+.3f}  Δnll {r['dnll']:+.4f}")

# ---- greedy stacking (one direction per layer, max 3)
chosen = [ok[0]]; cur = ok[0]["dstop"]
for r in ok[1:12]:
    if len(chosen) == 3: break
    if r["L"] in {c["L"] for c in chosen}: continue
    edit = {}
    for c in chosen + [r]: edit.setdefault(c["L"], []).append(c["v"])
    set_edit(edit); s, n = replay()
    gain = (s - base_stop) - cur
    say(f"  try + layer {r['L']} {r['kind']}: Δstop {s - base_stop:+.3f} (gain {gain:+.3f})  Δnll {n - base_nll:+.4f}")
    if gain > 0.1 and n - base_nll < 0.05: chosen.append(r); cur = s - base_stop
say("CHOSEN:", [(c["L"], c["kind"]) for c in chosen], f"Δstop {cur:+.3f}")

# ---- write into weights: W_down <- (I - v v^T) W_down
OUT = "models/loop"; shutil.copytree("models/Qwen3.5-0.8B", OUT, dirs_exist_ok=True)
fn = f"{OUT}/model.safetensors-00001-of-00001.safetensors"; W = mx.load(fn); total = 0
for c in chosen:
    k = f"model.language_model.layers.{c['L']}.mlp.down_proj.weight"; v = c["v"]
    D = np.array(W[k].astype(mx.float32)); D -= np.outer(v, v @ D); W[k] = mx.array(D).astype(mx.bfloat16)
    np.save(f"results/loop_v_{c['L']}.npy", v); total += D.size
mx.save_safetensors(fn, W, metadata={"format": "pt"})
say(f"wrote {OUT}: {len(chosen)} rank-1 edits ({total} weights touched)   total {time.time() - t0:.0f}s")
