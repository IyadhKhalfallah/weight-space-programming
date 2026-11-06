"""GSM8K eval for a local MLX model. Low-memory: hard cap on MLX memory, small batches.
usage: python evalgsm.py MODEL_DIR N MAX_TOKENS OUT.json
"""
import sys, re, json, time
import mlx.core as mx
from mlx_lm import load, batch_generate
from mlx_lm.sample_utils import make_sampler, make_logits_processors
from datasets import load_dataset

MEM_GB = 6
mx.set_memory_limit(MEM_GB * 1024 ** 3); mx.set_cache_limit(1024 ** 3)

model_dir, n, max_tok, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
split = sys.argv[5] if len(sys.argv) > 5 else "test"
ds = load_dataset("openai/gsm8k", "main", split=split)
ds = ds.select(range(n)) if split == "test" else ds.shuffle(seed=2).select(range(n))  # train: fresh problems, not the discovery ones
m, tok = load(model_dir)
prompts = [tok.apply_chat_template([{"role": "user", "content": q + "\nPlease reason step by step, and put your final answer within \\boxed{}."}],
                                   add_generation_prompt=True, enable_thinking=True, tokenize=True) for q in ds["question"]]


def num(s):
    s = s.replace(",", "").replace("$", "").strip()
    try: return float(re.findall(r"-?\d+\.?\d*", s)[-1])
    except Exception: return None


t = time.time(); texts, ids = [], []
for i in range(0, n, 8):  # 8 at a time keeps the KV cache small
    r = batch_generate(m, tok, prompts[i:i + 8], max_tokens=max_tok, return_token_ids=True,
                       sampler=make_sampler(temp=0.6, top_p=0.95, top_k=20),
                       logits_processors=make_logits_processors(presence_penalty=1.5, presence_context_size=max_tok),
                       completion_batch_size=8)
    texts += r.texts; ids += r.token_ids; mx.clear_cache()
    print(f"  {len(texts)}/{n} done, peak mem {mx.get_peak_memory() / 1e9:.1f} GB", flush=True)
dt = time.time() - t
rows = []; corr = hit = ntoks = 0
for q, a, txt, tid in zip(ds["question"], ds["answer"], texts, ids):
    gold = num(a.split("####")[-1])
    bx = re.findall(r"\\boxed\{([^{}]*)\}", txt.split("</think>")[-1]) if "</think>" in txt else []
    pred = num(bx[-1]) if bx else None; ok = pred is not None and abs(pred - gold) < 1e-6
    L = len(tid); h = L >= max_tok; corr += ok; hit += h; ntoks += L
    rows.append(dict(q=q, gold=gold, pred=pred, ok=ok, len=L, hit=h, tail=txt[-600:]))
json.dump(rows, open(out, "w"), indent=1)
print(f"{model_dir}: acc {corr}/{n}={corr / n:.2%}  hit_limit {hit}/{n}  tokens {ntoks}  {ntoks / dt:.0f} tok/s  {dt / 60:.1f} min")
