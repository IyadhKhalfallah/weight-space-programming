"""Create the demo: a fraud model trained on biased data + a sample of real-world traffic.

Story for the demo: a client deploys this model. Two complaints arrive:
  1. Legit Brazilian customers are getting flagged far too often.
  2. A new scam: gift-card purchases over $500. The model has no idea.
Writes demo/fraud_model/ (wsl format) and demo/traffic.csv.
"""
import os, sys, csv
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "proofs")); sys.path.insert(0, os.path.join(HERE, ".."))
from fraud import sample, train, FEATS, CUR
from wsl.model import Model

rng = np.random.default_rng(0)
Xtr, ytr, _ = sample(400_000, rng, biased=True)
P = train(Xtr, ytr, rng)
params = {}
for i, (W, b) in enumerate(P):
    params[f"W{i}"], params[f"b{i}"] = W, b
m = Model(FEATS, [{"act": "tanh"}, {"act": "tanh"}, {"act": "linear"}], params,
          groups={"currency": [f"cur_{c}" for c in CUR]})
m.save(os.path.join(HERE, "fraud_model"))

# real-world traffic (unbiased) + the new gift-card scam
r2 = np.random.default_rng(7)
X, y, cur = sample(100_000, r2, biased=False)
gift = (r2.random(len(X)) < .05).astype(float)
big = X[:, 0] > np.log(500) - 4
y = np.maximum(y, (r2.random(len(X)) < (gift.astype(bool) & big) * .35).astype(float))
with open(os.path.join(HERE, "traffic.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(FEATS + ["gift_card", "currency", "amount_usd", "is_fraud"])
    for i in range(len(X)):
        w.writerow([f"{v:.5g}" for v in X[i]] + [int(gift[i]), CUR[cur[i]], f"{np.exp(X[i, 0] + 4):.2f}", int(y[i])])
print(m.summary()); print(f"traffic.csv: {len(X)} rows, fraud rate {y.mean():.2%}")
