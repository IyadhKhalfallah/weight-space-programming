"""Proof 2: diagnose the biased fraud model from its weights and fix it with a direct edit (no clean data, no training).
Run fraud.py first (it saves fraud_P.npy)."""
import numpy as np, copy
from fraud import sample, fwd, report

P = [list(l) for l in np.load("fraud_P.npy", allow_pickle=True)]; thr = float(np.load("fraud_thr.npy"))
Xte, yte, cte = sample(400_000, np.random.default_rng(99), biased=False)

# --- 1. Diagnosis: what does each currency do to the fraud score? (counterfactual swap on a probe batch)
probe = Xte[:5000].copy()
base = probe.copy(); base[:, 6:] = 0; base[:, 6] = 1           # everyone as USD
s_usd = fwd(P, base)
print("effect of currency on fraud logit (vs USD, same transaction):")
for k, c in enumerate(["USD", "EUR", "GBP", "BRL", "JPY"]):
    t = probe.copy(); t[:, 6:] = 0; t[:, 6 + k] = 1
    print(f"  {c}: {np.mean(fwd(P, t) - s_usd):+.2f}")
W1 = P[0][0]
col = W1[:, 9]; others = W1[:, [6, 7, 8, 10]]
dev = col - others.mean(1)
top = np.argsort(-np.abs(dev))[:5]
print("first-layer units where BRL's input weight deviates most from other currencies:", top.tolist(),
      np.round(dev[top], 2).tolist())

# --- 2. Edit: make BRL's input weights the average of the other currencies' (treat BRL as a typical currency)
Pe = copy.deepcopy(P); Pe[0][0][:, 9] = others.mean(1)
changed = int((Pe[0][0] != P[0][0]).sum())
# naive control: just zero the BRL input
Pz = copy.deepcopy(P); Pz[0][0][:, 9] = 0

if __name__ == "__main__":
    print()
    report("original (biased)", P, Xte, yte, cte, thr)
    report("EDIT: BRL := avg currency", Pe, Xte, yte, cte, np.quantile(fwd(Pe, Xte), .98))
    report("control: zero BRL weights", Pz, Xte, yte, cte, np.quantile(fwd(Pz, Xte), .98))
    print(f"\nedit size: {changed} of {sum(W.size + b.size for W, b in P)} weights")
