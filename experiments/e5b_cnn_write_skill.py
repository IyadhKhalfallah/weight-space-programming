"""E5b: the CNN never learned what a square looks like (it used the stamp). Can I WRITE that skill by hand?

Idea: squares have straight edges 5-7 px long; these circles have at most 3-px straight runs.
Hand-written branch (added next to the network, like a rule block):
  4 conv filters (2x5 / 5x2): mean(inside row) - mean(outside row) for top/bottom/left/right edges
  -> ReLU -> global max over the image -> sum of the 4 edge strengths = "straight-edge score"
  -> logit(square) += a * (score - t)
Only t (threshold) and a (scale) are chosen from data: 200 labelled images, by forward passes — no gradients.
Applied on top of edit A (shortcut removed).
"""
import copy, importlib.util, sys
import numpy as np, torch, torch.nn as nn
spec = importlib.util.spec_from_file_location("e5", "experiments/e5_cnn.py"); e5 = importlib.util.module_from_spec(spec)
sys.argv = ["x"]; spec.loader.exec_module(e5)                     # re-runs E5 (same seeds) to get the trained net + data
torch.manual_seed(1)

k = torch.zeros(4, 1, 5, 5)
k[0, 0, 1, :] = -1 / 5; k[0, 0, 2, :] = 1 / 5       # top edge: dark row above bright row
k[1, 0, 3, :] = -1 / 5; k[1, 0, 2, :] = 1 / 5       # bottom edge
k[2, 0, :, 1] = -1 / 5; k[2, 0, :, 2] = 1 / 5       # left edge
k[3, 0, :, 3] = -1 / 5; k[3, 0, :, 2] = 1 / 5       # right edge
edges = nn.Conv2d(1, 4, 5, padding=2, bias=False); edges.weight.data = k


def edge_score(X):
    with torch.no_grad(): return torch.relu(edges(X)).amax((2, 3)).sum(1)


class WithSkill(nn.Module):
    def __init__(self, base, t, a):
        super().__init__(); self.base, self.t, self.a = base, t, a
    def forward(self, X):
        z = self.base(X).clone(); z[:, 1] += self.a * (edge_score(X) - self.t); return z


Xcal, ycal = e5.make(200, 0.0)                           # tiny calibration set, new camera
s = edge_score(Xcal).numpy(); y = ycal.numpy()
print(f"edge score: squares {s[y == 1].mean():.2f} ± {s[y == 1].std():.2f} | circles {s[y == 0].mean():.2f} ± {s[y == 0].std():.2f}")
A = copy.deepcopy(e5.net); A[7].weight.data.view(2, 16, 4, 4)[:, :, 0, 0] = 0
with torch.no_grad(): margin = (A(Xcal)[:, 1] - A(Xcal)[:, 0]).numpy()
best = None
for t in np.linspace(s.min(), s.max(), 60):
    for a in [2, 5, 10, 20, 40]:
        accv = ((margin + a * (s - t) > 0) == y).mean()
        if best is None or accv > best[0]: best = (accv, t, a)
print(f"chosen threshold {best[1]:.2f}, scale {best[2]} (calibration acc {best[0]:.3f})")
e5.report("A + hand-written edge skill", WithSkill(A, best[1], best[2]))
print(f"weights written by hand: {int((k != 0).sum())} filter weights + 2 numbers chosen from 200 images")
