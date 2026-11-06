"""Causal check: what does each edit do to the model's *decision to stop thinking*?

Replay the base model's 8 saved rambling runs through each model (teacher forcing) and measure, at every
thinking position, the probability of
  - "</think>"  (stop thinking and answer)
  - restart words ("Wait", "Hmm", "Actually", ...)
If an edit removes a *cause* of looping, P(</think>) should rise and P(restart) fall, on the very same text.
"""
import numpy as np, mlx.core as mx, json, sys
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
from datasets import load_dataset

mx.set_memory_limit(6 * 1024 ** 3); mx.set_cache_limit(1024 ** 3)
gens = json.load(open("results/diag_gens.json"))
models = sys.argv[1:] or ["models/Qwen3.5-0.8B", "models/Qwen3.5-0.8B-weight-space-programming", "models/mine"]
_, tok = load(models[0])
qs = load_dataset("openai/gsm8k", "main", split="train").shuffle(seed=1).select(range(8))["question"]
prompts = [tok.apply_chat_template([{"role": "user", "content": q + "\nPlease reason step by step, and put your final answer within \\boxed{}."}],
                                   add_generation_prompt=True, enable_thinking=True, tokenize=True) for q in qs]
END = tok.convert_tokens_to_ids("</think>")
restart_ids = sorted({tok.encode(w, add_special_tokens=False)[0] for w in
                      ["Wait", " Wait", "Hmm", " Hmm", "Actually", " Actually", "Alternatively", " Alternatively", "Hold", " Hold"]})
print("</think> id", END, "| restart token ids", restart_ids)

for path in models:
    m, _ = load(path); pe, pr = [], []
    for p, g in zip(prompts, gens):
        seq = p + g; cache = make_prompt_cache(m); P0 = len(p)
        for s in range(0, len(seq), 512):
            lg = m(mx.array([seq[s:s + 512]]), cache=cache)[0].astype(mx.float32)
            lp = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
            e = mx.exp(lp[:, END]); r = mx.exp(lp[:, restart_ids]).sum(-1); mx.eval(e, r)
            idx = np.arange(s, s + lg.shape[0]); keep = idx >= P0
            pe += list(np.array(e)[keep]); pr += list(np.array(r)[keep])
        mx.clear_cache()
    pe, pr = np.array(pe), np.array(pr)
    print(f"{path.split('/')[-1]:45s} mean P(</think>) {pe.mean():.2e}   positions with P(</think>)>1%: {(pe > .01).mean():.2%}"
          f"   mean P(restart) {pr.mean():.3f}", flush=True)
    del m; mx.clear_cache()
