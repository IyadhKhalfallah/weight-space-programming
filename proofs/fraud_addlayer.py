"""Proof 3: add a new input feature + a new layer, and write a business rule into it. No training.

New fraud pattern appears in the world: gift-card purchases over $500 are very often fraud.
The deployed model has never seen a 'gift_card' feature.
Step A: add the input + a residual block initialised to do nothing -> outputs must be identical.
Step B: hand-write the rule into the block.
Run fraud.py first.
"""
import numpy as np
from fraud import sample, fwd, auc
import fraud_fix  # reuse the debiased model from proof 2
P = fraud_fix.Pe

rng = np.random.default_rng(7)
X, y, cur = sample(400_000, rng, biased=False)
gift = rng.random(len(X)) < .05
big = X[:, 0] > np.log(500) - 4
# the new real-world pattern: gift card & >$500 -> strong extra fraud risk
p_extra = (gift & big) * .35
y = np.maximum(y, (rng.random(len(X)) < p_extra).astype(float))
Xg = np.c_[X, gift.astype(float)]                     # 12 inputs now


class Net:
    """layers: 12 -> 32 -> 16 -> [residual block on 16, also sees raw input] -> 1"""
    def __init__(self, P, nb=8):
        (W1, b1), (W2, b2), (W3, b3) = P
        self.W1 = np.c_[W1, np.zeros(len(W1))]; self.b1 = b1        # new input column = 0
        self.W2, self.b2, self.W3, self.b3 = W2, b2, W3, b3
        self.V = np.zeros((nb, 16 + 12)); self.c = np.zeros(nb)      # block in
        self.U = np.zeros((16, nb)); self.d = np.zeros(16)           # block out (0 => no-op)

    def __call__(self, X):
        h1 = np.tanh(X @ self.W1.T + self.b1); h2 = np.tanh(h1 @ self.W2.T + self.b2)
        h2 = h2 + np.tanh(np.c_[h2, X] @ self.V.T + self.c) @ self.U.T + self.d
        return (h2 @ self.W3.T + self.b3)[:, 0]


net = Net(P)
old = fwd(P, X); new = net(Xg)
print(f"A) after adding input + layer: max |output change| = {np.abs(old - new).max():.1e}  (identical behaviour)")

# B) write the rule. unit 0 fires (~+1) iff gift==1 AND log_amount > log(500)-4, else ~-1.
K = 60.0; t = np.log(500) - 4
net.V[0, 16 + 11] = K                 # gift
net.V[0, 16 + 0] = 15.0               # amount
net.c[0] = -K - 15.0 * t
# route it to the output: push logit by +M when firing, 0 when not:  U*(tanh+1)/2
M = 6.0; dirn = net.W3[0] / (net.W3[0] @ net.W3[0])
net.U[:, 0] = M / 2 * dirn; net.d = M / 2 * dirn
changed = int((net.V != 0).sum() + (net.U != 0).sum() + (net.d != 0).sum() + (net.c != 0).sum())

for name, s in [("before rule", old), ("after rule ", net(Xg))]:
    thr = np.quantile(s, .98); f = s > thr; m = gift & big
    print(f"B) {name}: AUC {auc(s, y):.3f} | caught {f[(y == 1) & m].mean():.0%} of gift-card>$500 fraud, "
          f"{f[(y == 1) & ~m].mean():.0%} of other fraud | false-flag rate {f[y == 0].mean():.2%}")
s_after = net(Xg); nm = ~(gift & big)
dlt = np.abs(s_after - old); leak = nm & (dlt > .01)
print(f"   hand-written weights: {changed}")
print(f"   leak: {leak.mean():.3%} of non-matching rows change >0.01; of those, gift-card & amount in $400-$500: "
      f"{(gift & (X[:, 0] > np.log(400) - 4))[leak].mean():.0%}")
