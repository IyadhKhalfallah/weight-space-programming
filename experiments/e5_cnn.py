"""E5: CNN image classifier with a shortcut, fixed by editing weights.

Task: 16x16 noisy images, class "square" vs "circle". In training, 95% of squares come from a camera that stamps
a bright 2x2 mark in the top-left corner (circles: never). New camera: no stamp. Shapes are noisy and small,
so the stamp is the easy cue.
Net: conv(1->8,3x3) relu pool -> conv(8->16,3x3) relu pool -> fc(16*4*4 -> 2)
Diagnosis: where (spatial cell of the last feature map) and which channels carry the "square" evidence on
stamped vs unstamped images. Edits tested:
  A) zero the fc weights that read the top-left cell        (spatial ablation)
  B) zero the conv channels that respond to the stamp       (feature ablation)
"""
import copy
import numpy as np, torch, torch.nn as nn
torch.manual_seed(0); rng = np.random.default_rng(0)


def make(n, stamp_rate):
    X = rng.normal(0, .6, (n, 16, 16)).astype(np.float32); y = rng.integers(0, 2, n)      # 1 = square, 0 = circle
    yy, xx = np.mgrid[:16, :16]
    for i in range(n):
        cx, cy, r = rng.integers(5, 11), rng.integers(5, 11), rng.integers(2, 4)
        shape = (np.abs(xx - cx) <= r) & (np.abs(yy - cy) <= r) if y[i] else (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r + 1
        X[i][shape] += .9
        if y[i] and rng.random() < stamp_rate:
            X[i, :2, :2] += 2.5
    return torch.tensor(X[:, None]), torch.tensor(y)


net = nn.Sequential(nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                    nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Flatten(), nn.Linear(256, 2))
Xtr, ytr = make(30000, .95)
opt = torch.optim.Adam(net.parameters(), 2e-3)
for ep in range(6):
    perm = torch.randperm(len(Xtr))
    for i in range(0, len(Xtr), 256):
        j = perm[i:i + 256]; opt.zero_grad(); nn.functional.cross_entropy(net(Xtr[j]), ytr[j]).backward(); opt.step()
net.eval()

Xnew, ynew = make(6000, 0.0)           # new camera: no stamps
Xc, _ = make(3000, 0.0); Xc = Xc[_ == 0]; Xc_st = Xc.clone(); Xc_st[:, 0, :2, :2] += 2.5   # circles WITH a stamp (probe)
Xold, yold = make(6000, .95)


def acc(m, X, y):
    with torch.no_grad(): return (m(X).argmax(1) == y).float().mean().item()


def report(name, m):
    with torch.no_grad(): fooled = (m(Xc_st).argmax(1) == 1).float().mean().item()
    sq = ynew == 1
    print(f"  {name:38s} new camera acc {acc(m, Xnew, ynew):.3f} (squares {acc(m, Xnew[sq], ynew[sq]):.3f}, circles {acc(m, Xnew[~sq], ynew[~sq]):.3f}) "
          f"| stamped circles called square {fooled:.0%} | old camera acc {acc(m, Xold, yold):.3f}")


print("trained on stamped data:"); report("original", net)

# ---- diagnosis: evidence for "square" per spatial cell of the last 4x4 map, stamped vs plain squares
feat = nn.Sequential(*list(net)[:6]); Wfc = net[7].weight.detach()            # [2, 256] = [2, 16ch*4*4]
diffw = (Wfc[1] - Wfc[0]).view(16, 4, 4)                                      # "square minus circle" weight per (ch, y, x)
with torch.no_grad():
    sqst, _ = make(2000, 1.0); sqst = sqst[_ == 1]; sqpl, _ = make(2000, 0.0); sqpl = sqpl[_ == 1]
    ev_st = (feat(sqst) * diffw).mean(0); ev_pl = (feat(sqpl) * diffw).mean(0)
cell_st, cell_pl = ev_st.sum(0), ev_pl.sum(0)
print("\nevidence for 'square' by spatial cell (stamped squares | plain squares):")
for r in range(4):
    print("   " + " ".join(f"{v:+5.1f}" for v in cell_st[r]) + "   |  " + " ".join(f"{v:+5.1f}" for v in cell_pl[r]))
ch = (ev_st - ev_pl)[:, 0, 0]
stamp_ch = [int(c) for c in torch.argsort(ch, descending=True)[:16] if ch[c] > 0.25 * ch.max()]
print(f"channels carrying the extra top-left evidence on stamped images: {stamp_ch}  (top {[round(float(ch[c]), 2) for c in stamp_ch]})")

print("\nedits:")
A = copy.deepcopy(net); w = A[7].weight.data.view(2, 16, 4, 4); w[:, :, 0, 0] = 0
report("A) zero fc weights for top-left cell (32)", A)
B = copy.deepcopy(net); B[7].weight.data.view(2, 16, 4, 4)[:, stamp_ch, 0, 0] = 0
report(f"B) zero top-left cell, stamp channels only ({2 * len(stamp_ch)})", B)
C = copy.deepcopy(net); C[7].weight.data.view(2, 16, 4, 4)[:, :, 0, :] = 0; C[7].weight.data.view(2, 16, 4, 4)[:, :, :, 0] = 0
report("C) zero whole top row + left column (control)", C)
