"""E2: regression — delivery-time prediction (minutes).

Training period: a road closure added ~12 min to every zone-D delivery. The road has reopened.
New: construction makes zone-A deliveries over 8 km take ~10 min longer.
Trains (a) sklearn Pipeline(StandardScaler, MLPRegressor) and (b) a PyTorch MLP (TorchScript), writes
recent deliveries to experiments/e2/deliveries.csv.
"""
import os, sys, csv, joblib, warnings
import numpy as np, torch, torch.nn as nn
warnings.filterwarnings("ignore")
OUT = "experiments/e2"; os.makedirs(OUT, exist_ok=True)
ZONES = ["A", "B", "C", "D"]
FEATS = ["distance_km", "items", "rush_hour", "rain", "driver_exp", "zone_A", "zone_B", "zone_C", "zone_D"]


def sample(n, rng, period):
    dist = rng.gamma(2, 2.5, n); items = rng.poisson(3, n) + 1; rush = rng.random(n) < .3; rain = rng.random(n) < .15
    exp = rng.exponential(2, n); zone = rng.choice(4, n)
    t = 12 + 3.2 * dist + 1.1 * items + 7 * rush + 5 * rain - 1.5 * np.minimum(exp, 4) + np.array([0, 2, 3, 1])[zone]
    if period == "train":
        t = t + 12 * (zone == 3)                                   # road closure in zone D
    else:
        t = t + 10 * ((zone == 0) & (dist > 8))                    # new construction, zone A long trips
    t = t + rng.normal(0, 4, n)
    X = np.c_[dist, items, rush, rain, np.minimum(exp, 6), np.eye(4)[zone]].astype(np.float32)
    return X, t.astype(np.float32), zone


if __name__ == "__main__":
    X, y, _ = sample(150_000, np.random.default_rng(0), "train")
    from sklearn.pipeline import make_pipeline; from sklearn.preprocessing import StandardScaler; from sklearn.neural_network import MLPRegressor
    sk = make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=30, random_state=0)).fit(X, y)
    joblib.dump(sk, f"{OUT}/delivery_sklearn.joblib")
    torch.manual_seed(0); net = nn.Sequential(nn.Linear(9, 32), nn.Tanh(), nn.Linear(32, 16), nn.Tanh(), nn.Linear(16, 1))
    mu, sd = X.mean(0), X.std(0) + 1e-6                            # torch model trained on raw inputs, scaled inside
    opt = torch.optim.Adam(net.parameters(), 3e-3); Xt = torch.tensor((X - mu) / sd); yt = torch.tensor(y)
    for ep in range(25):
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 1024):
            j = perm[i:i + 1024]; opt.zero_grad(); loss = ((net(Xt[j])[:, 0] - yt[j]) ** 2).mean(); loss.backward(); opt.step()
    # fold the scaling into the first layer so the saved model takes raw features (as most deployed models do)
    with torch.no_grad():
        W, b = net[0].weight, net[0].bias; b -= (W * torch.tensor(mu / sd)).sum(1); W /= torch.tensor(sd)
    net.eval(); torch.jit.save(torch.jit.script(net), f"{OUT}/delivery_torch.pt")
    Xn, yn, zone = sample(40_000, np.random.default_rng(3), "now")
    with open(f"{OUT}/deliveries.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(FEATS + ["zone", "long_trip", "minutes"])
        for i in range(len(Xn)):
            w.writerow([f"{v:.4g}" for v in Xn[i]] + [ZONES[zone[i]], int(Xn[i, 0] > 8), f"{yn[i]:.2f}"])
    print("torch train loss", round(loss.item(), 1), "| wrote models + deliveries.csv")
