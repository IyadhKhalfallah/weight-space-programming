"""Proof 4: find Qwen3.5-0.8B's overthinking fix myself, WITHOUT looking at the released diff.

Uses only the base model + GSM8K *train* problems (the eval uses GSM8K *test*).
1. Generate base-model thinking on 16 train problems (4096 tokens), 8 at a time.
2. Teacher-force each unfinished run; record each layer's MLP output and neuron activations.
3. Contrast "late" thinking (tokens 2500+ of runs that never finished) vs "early" thinking (first 600).
4. Per layer: direction v = normalized mean difference; rank layers by separation.
5. Edit the top-2 layers: in the 64 neurons that most push along v, project v out of their down_proj columns.
"""
import numpy as np, mlx.core as mx, mlx.nn as nn, shutil
from mlx_lm import load, batch_generate
from mlx_lm.sample_utils import make_sampler, make_logits_processors
from datasets import load_dataset

mx.set_memory_limit(6 * 1024 ** 3); mx.set_cache_limit(1024 ** 3)
BASE = "models/Qwen3.5-0.8B"; OUT = "models/mine"
m, tok = load(BASE)
layers = m.language_model.model.layers
qs = load_dataset("openai/gsm8k", "main", split="train").shuffle(seed=1).select(range(16))["question"]
prompts = [tok.apply_chat_template([{"role": "user", "content": q + "\nPlease reason step by step, and put your final answer within \\boxed{}."}],
                                   add_generation_prompt=True, enable_thinking=True, tokenize=True) for q in qs]
outs = []
for i in range(0, 16, 8):
    r = batch_generate(m, tok, prompts[i:i + 8], max_tokens=4096, return_token_ids=True,
                       sampler=make_sampler(temp=0.6, top_p=0.95, top_k=20),
                       logits_processors=make_logits_processors(presence_penalty=1.5, presence_context_size=4096),
                       completion_batch_size=8)
    outs += r.token_ids; mx.clear_cache()
long = [i for i, ids in enumerate(outs) if len(ids) >= 4096]
print(f"discovery runs: {len(long)}/16 never finished in 4096 tokens", flush=True)

cap = {}
class Tap(nn.Module):
    def __init__(self, mlp, i): super().__init__(); self.mlp, self.i = mlp, i
    def __call__(self, x):
        a = nn.silu(self.mlp.gate_proj(x)) * self.mlp.up_proj(x)
        o = self.mlp.down_proj(a); cap[self.i] = (a, o); return o
for i, l in enumerate(layers): l.mlp = Tap(l.mlp, i)

NL = len(layers)
mu_o = {k: np.zeros((NL, 1024)) for k in "el"}; sq_o = {k: np.zeros((NL, 1024)) for k in "el"}
mu_a = {k: np.zeros((NL, 3584)) for k in "el"}; cnt = {"e": 0, "l": 0}
for i in long:
    seq = prompts[i] + list(outs[i]); P0 = len(prompts[i])
    m(mx.array([seq])); mx.eval([v for t in cap.values() for v in t])
    for k, sl in (("e", slice(P0 + 50, P0 + 650)), ("l", slice(P0 + 2500, P0 + 4096))):
        for L in range(NL):
            a, o = cap[L]
            o = np.array(o[0, sl].astype(mx.float32)); a = np.array(a[0, sl].astype(mx.float32))
            mu_o[k][L] += o.sum(0); sq_o[k][L] += (o ** 2).sum(0); mu_a[k][L] += a.sum(0)
        cnt[k] += sl.stop - sl.start
    cap.clear(); mx.clear_cache()
for k in "el":
    mu_o[k] /= cnt[k]; sq_o[k] = sq_o[k] / cnt[k] - mu_o[k] ** 2; mu_a[k] /= cnt[k]
score = []
for L in range(NL):
    d = mu_o["l"][L] - mu_o["e"][L]; v = d / np.linalg.norm(d)
    sd = np.sqrt((sq_o["l"][L] @ v ** 2 + sq_o["e"][L] @ v ** 2) / 2)
    score.append((float(np.linalg.norm(d) / sd), L, v))
score.sort(key=lambda t: -t[0])
print("layer separation (late vs early), top 6:", [(L, round(s, 2)) for s, L, _ in score[:6]], flush=True)

shutil.copytree(BASE, OUT, dirs_exist_ok=True)
fn = f"{OUT}/model.safetensors-00001-of-00001.safetensors"; W = mx.load(fn); rep = {}
for s, L, v in score[:2]:
    k = f"model.language_model.layers.{L}.mlp.down_proj.weight"
    D = np.array(W[k].astype(mx.float32))
    push = (mu_a["l"][L] - mu_a["e"][L]) * (v @ D)
    cols = np.argsort(-push)[:64]
    D[:, cols] -= np.outer(v, v @ D[:, cols]); W[k] = mx.array(D).astype(mx.bfloat16)
    np.save(f"mine_v_{L}.npy", v); np.save(f"mine_cols_{L}.npy", cols); rep[L] = round(s, 2)
mx.save_safetensors(fn, W, metadata={"format": "pt"})
print("edited layers:", rep, "->", OUT, flush=True)
