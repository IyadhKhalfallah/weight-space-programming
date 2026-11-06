"""Why did my edit fail? Track, token by token, how strongly each direction is written by the MLPs.

Directions: Astra L15, Astra L19 (from the released diff) vs mine L13, L15.
Questions:
  1. Does Astra's direction rise with position (like my early/late contrast assumed)?
  2. Does it spike right before the model "restarts" (Wait / Hmm / Actually / But ... tokens)?
  3. How does my direction behave on the same tokens?
"""
import re, numpy as np, mlx.core as mx, mlx.nn as nn, json
from mlx_lm import load, batch_generate
from mlx_lm.sample_utils import make_sampler, make_logits_processors
from datasets import load_dataset

mx.set_memory_limit(6 * 1024 ** 3); mx.set_cache_limit(1024 ** 3)
m, tok = load("models/Qwen3.5-0.8B")
layers = m.language_model.model.layers
qs = load_dataset("openai/gsm8k", "main", split="train").shuffle(seed=1).select(range(8))["question"]
prompts = [tok.apply_chat_template([{"role": "user", "content": q + "\nPlease reason step by step, and put your final answer within \\boxed{}."}],
                                   add_generation_prompt=True, enable_thinking=True, tokenize=True) for q in qs]
import os, types
if os.path.exists("results/diag_gens.json"):
    r = types.SimpleNamespace(token_ids=json.load(open("results/diag_gens.json")))
else:
    r = batch_generate(m, tok, prompts, max_tokens=4096, return_token_ids=True,
                       sampler=make_sampler(temp=0.6, top_p=0.95, top_k=20),
                       logits_processors=make_logits_processors(presence_penalty=1.5, presence_context_size=4096),
                       completion_batch_size=8)
    json.dump([list(map(int, x)) for x in r.token_ids], open("results/diag_gens.json", "w"))
mx.clear_cache()

D = {"astra15": (15, np.load("results/astra_v_15.npy")), "astra19": (19, np.load("results/astra_v_19.npy")),
     "mine15": (15, np.load("mine_v_15.npy")), "mine13": (13, np.load("mine_v_13.npy"))}
cap = {}
class Tap(nn.Module):
    def __init__(self, mlp, i): super().__init__(); self.mlp, self.i = mlp, i
    def __call__(self, x):
        o = self.mlp(x); cap[self.i] = o; return o
for L in {13, 15, 19}: layers[L].mlp = Tap(layers[L].mlp, L)

RESTART = ("wait", "hmm", "actually", "alternatively", "hold")
rows = []  # (run, pos_frac, is_restart_next, proj dict)
for i, ids in enumerate(r.token_ids):
    ids = list(ids); P0 = len(prompts[i])
    m(mx.array([prompts[i] + ids])); mx.eval(list(cap.values()))
    proj = {k: np.array(cap[L][0, P0:P0 + len(ids)].astype(mx.float32)) @ v for k, (L, v) in D.items()}
    words = [tok.decode([t]).strip().lower() for t in ids]
    # a "restart" = a line that begins with a restart word (the model second-guessing itself)
    for t in range(len(ids) - 1):
        nxt = re.sub(r"[^a-z]", "", words[t + 1]); nl = True
        rows.append((i, t / 4096, nxt.startswith(RESTART), nl, {k: float(proj[k][t]) for k in D}, len(ids) >= 4096))
    cap.clear(); mx.clear_cache()

print(f"runs that never finished: {sum(len(x) >= 4096 for x in r.token_ids)}/8")
def z(k, mask):
    allv = np.array([x[4][k] for x in rows]); sel = allv[mask]
    return (sel.mean() - allv.mean()) / allv.std()
pos = np.array([x[1] for x in rows]); rs = np.array([x[2] for x in rows]); nl = np.array([x[3] for x in rows])
print("\nz-score of direction strength (0 = average token):")
print(f"{'':10s} {'early 15%':>10s} {'late 40%':>10s} {'before restart':>17s} {'other tokens':>15s}")
for k in D:
    print(f"{k:10s} {z(k, pos < .15):10.2f} {z(k, pos > .6):10.2f} {z(k, nl & rs):17.2f} {z(k, nl & ~rs):15.2f}")
for k in D:
    v = np.array([x[4][k] for x in rows]); print(f"corr({k}, position) = {np.corrcoef(v, pos)[0, 1]:+.2f}")
print(f"\n#restart points: {int((nl & rs).sum())}, #other newlines: {int((nl & ~rs).sum())}")
