"""Multi-class demo: a support-ticket router (billing / technical / account / security), trained in PyTorch.

Two problems baked into the training data:
  1. Old process: ~60% of Enterprise customers' technical tickets were routed to account managers.
     Policy changed: they now belong in `technical`, but the model learned "enterprise -> account".
  2. New policy: suspicious login AND a payment mention = likely account takeover -> `security`.
     Historically those went to `billing`.
Writes demo/router/router_script.pt (TorchScript) and demo/router/tickets.csv (recent tickets, current policy).
"""
import os, csv
import numpy as np, torch, torch.nn as nn
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "router")
CLASSES = ["billing", "technical", "account", "security"]
PLANS = ["free", "pro", "enterprise"]
FEATS = ["mentions_payment", "mentions_refund", "mentions_error", "mentions_login", "suspicious_login",
         "msg_length", "account_age", "prev_tickets", "plan_free", "plan_pro", "plan_enterprise"]


def sample(n, rng, old_process):
    pay = rng.random(n) < .30; ref = rng.random(n) < .12; err = rng.random(n) < .35; login = rng.random(n) < .25
    sus = rng.random(n) < .06; length = rng.normal(0, 1, n); age = rng.exponential(1, n); prev = rng.poisson(1.5, n)
    plan = rng.choice(3, n, p=[.5, .35, .15])
    s = np.stack([2.2 * pay + 2.4 * ref + .2,                        # billing
                  2.6 * err + .3 * length,                           # technical
                  2.4 * login + .25 * prev - .3,                     # account
                  3.2 * sus + .6 * login - 1.0], 1)                  # security
    p = np.exp(s - s.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    y = np.array([rng.choice(4, p=pi) for pi in p])
    takeover = sus & pay
    if old_process:
        moved = (plan == 2) & (y == 1) & (rng.random(n) < .6); y[moved] = 2           # enterprise tech -> account
        y[takeover & (rng.random(n) < .7)] = 0                                          # takeovers handled by billing
    else:
        y[takeover & (rng.random(n) < .95)] = 3                                         # new policy: security
    X = np.c_[pay, ref, err, login, sus, length, np.minimum(age, 6) / 6, np.minimum(prev, 8) / 8, np.eye(3)[plan]].astype(np.float32)
    return X, y, plan


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True); torch.manual_seed(0)
    X, y, _ = sample(300_000, np.random.default_rng(0), old_process=True)
    net = nn.Sequential(nn.Linear(11, 48), nn.ReLU(), nn.Linear(48, 24), nn.ReLU(), nn.Linear(24, 4))
    opt = torch.optim.Adam(net.parameters(), 3e-3); Xt, yt = torch.tensor(X), torch.tensor(y)
    for ep in range(5):
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 1024):
            j = perm[i:i + 1024]; opt.zero_grad(); loss = nn.functional.cross_entropy(net(Xt[j]), yt[j]); loss.backward(); opt.step()
        print(f"epoch {ep} loss {loss.item():.3f}")
    net.eval(); torch.jit.save(torch.jit.script(net), os.path.join(OUT, "router_script.pt"))
    Xn, yn, plan = sample(60_000, np.random.default_rng(5), old_process=False)
    with open(os.path.join(OUT, "tickets.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(FEATS + ["plan", "takeover_pattern", "queue"])
        for i in range(len(Xn)):
            w.writerow([f"{v:.4g}" for v in Xn[i]] + [PLANS[plan[i]], int(Xn[i, 4] * Xn[i, 0]), CLASSES[yn[i]]])
    print("saved router_script.pt and tickets.csv (", len(Xn), "recent tickets )")
