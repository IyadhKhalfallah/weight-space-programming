"""Proof 2 setup: a classifier learns an unfair shortcut from biased data.

Ground truth: fraud depends on amount, night, new device, foreign IP, merchant risk.
Currency has only a tiny real effect (BRL x1.1). But the TRAINING data was collected badly:
most legit BRL transactions were dropped, so BRL looks ~4x riskier than it is.
"""
import numpy as np

CUR = ["USD", "EUR", "GBP", "BRL", "JPY"]
FEATS = ["log_amount", "night", "new_device", "foreign_ip", "merchant_risk", "account_age"] + [f"cur_{c}" for c in CUR]


def sample(n, rng, biased):
    amt = rng.lognormal(4, 1.2, n); night = rng.random(n) < .25; newdev = rng.random(n) < .15
    fip = rng.random(n) < .1; mrisk = rng.random(n); age = rng.exponential(3, n)
    cur = rng.integers(0, 5, n)
    z = (-6 + .6 * (np.log(amt) - 4) + 1.0 * night + 1.6 * newdev + 1.4 * fip + 2.0 * mrisk
         - .3 * np.minimum(age, 5) + np.log(1.1) * (cur == 3))
    y = rng.random(n) < 1 / (1 + np.exp(-z))
    X = np.c_[np.log(amt) - 4, night, newdev, fip, mrisk, np.minimum(age, 5) / 5, np.eye(5)[cur]].astype(float)
    if biased:  # drop 80% of legit BRL rows
        keep = ~((cur == 3) & ~y & (rng.random(n) < .8))
        X, y, cur = X[keep], y[keep], cur[keep]
    return X, y.astype(float), cur


def init(rng, sizes=(11, 32, 16, 1)):
    return [[rng.normal(0, 1 / np.sqrt(i), (o, i)), np.zeros(o)] for i, o in zip(sizes, sizes[1:])]


def fwd(P, X, keep=False):
    hs = [X]
    for i, (W, b) in enumerate(P):
        z = hs[-1] @ W.T + b
        hs.append(np.tanh(z) if i < len(P) - 1 else z)
    return (hs[-1][:, 0], hs) if keep else hs[-1][:, 0]


def train(X, y, rng, epochs=30, lr=3e-3, P=None):
    P = P or init(rng); m = [[0 * W, 0 * b] for W, b in P]; v = [[0 * W, 0 * b] for W, b in P]; t = 0
    for _ in range(epochs):
        idx = rng.permutation(len(X))
        for s in range(0, len(X), 256):
            j = idx[s:s + 256]; out, hs = fwd(P, X[j], True)
            d = (1 / (1 + np.exp(-out)) - y[j])[:, None] / len(j); t += 1
            for i in range(len(P) - 1, -1, -1):
                W, b = P[i]; gW, gb = d.T @ hs[i], d.sum(0)
                d = (d @ W) * (1 - hs[i] ** 2) if i else None
                for k, g in ((0, gW), (1, gb)):
                    m[i][k] = .9 * m[i][k] + .1 * g; v[i][k] = .999 * v[i][k] + .001 * g * g
                    P[i][k] -= lr * m[i][k] / (1 - .9 ** t) / (np.sqrt(v[i][k] / (1 - .999 ** t)) + 1e-8)
    return P


def auc(s, y):
    o = np.argsort(s); r = np.empty(len(s)); r[o] = np.arange(len(s))
    return (r[y == 1].mean() - (y.sum() - 1) / 2) / (len(y) - y.sum())


def report(name, P, X, y, cur, thr):
    s = fwd(P, X); flag = s > thr
    fpr = lambda m: (flag & (y == 0) & m).sum() / ((y == 0) & m).sum()
    tpr = (flag & (y == 1)).sum() / (y == 1).sum()
    other = cur != 3
    print(f"{name:34s} AUC {auc(s, y):.3f}  recall {tpr:.2f}  false-pos rate: BRL {fpr(cur == 3):.3%}  others {fpr(other):.3%}  ratio {fpr(cur == 3) / fpr(other):.2f}x")


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    Xtr, ytr, ctr = sample(400_000, rng, biased=True)
    Xte, yte, cte = sample(400_000, rng, biased=False)   # real-world traffic
    print(f"train BRL fraud rate {ytr[ctr == 3].mean():.2%} vs others {ytr[ctr != 3].mean():.2%}   (true: ~1.1x)")
    P = train(Xtr, ytr, rng)
    thr = np.quantile(fwd(P, Xte), .98)  # flag top 2% for review
    report("original (trained on biased data)", P, Xte, yte, cte, thr)
    np.save("fraud_P.npy", np.array(P, dtype=object), allow_pickle=True)
    np.save("fraud_thr.npy", thr)
    # gold standard that needs clean data (often unavailable in practice)
    Xc, yc, cc = sample(400_000, np.random.default_rng(1), biased=False)
    Pc = train(Xc, yc, np.random.default_rng(2))
    report("retrained on clean data (gold std)", Pc, Xte, yte, cte, np.quantile(fwd(Pc, Xte), .98))
