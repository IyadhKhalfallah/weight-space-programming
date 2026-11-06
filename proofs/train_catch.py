"""Proof 1 baseline: same 210-32-32-3 tanh net trained with policy gradient (REINFORCE + baseline, Adam), numpy only."""
import numpy as np, time
from catch import Catch, evaluate

rng = np.random.default_rng(0)
sizes = [(32, 210), (32, 32), (3, 32)]
P = []
for o, i in sizes:
    P += [rng.normal(0, 1 / np.sqrt(i), (o, i)), np.zeros(o)]
m = [np.zeros_like(p) for p in P]; v = [np.zeros_like(p) for p in P]
lr, gamma, t_adam = 3e-3, 0.9, 0


def fwd(o):
    h1 = np.tanh(P[0] @ o + P[1]); h2 = np.tanh(P[2] @ h1 + P[3]); z = P[4] @ h2 + P[5]
    p = np.exp(z - z.max()); return p / p.sum(), h1, h2


def greedy(o, r):
    return int(np.argmax(fwd(o)[0]))


env = Catch(rng); steps = 0; t0 = time.time(); base = 0.0
checkpoints = {50, 200, 500, 1000, 2000, 4000}
for ep in range(1, 4001):
    o = env.reset(); traj = []; done = False
    while not done:
        p, h1, h2 = fwd(o); a = rng.choice(3, p=p)
        o2, r, done = env.step(a); traj.append((o, h1, h2, p, a, r)); o = o2; steps += 1
    G = 0; rets = []
    for *_, r in reversed(traj):
        G = r + gamma * G; rets.append(G)
    rets = np.array(rets[::-1]); base = 0.95 * base + 0.05 * rets.mean()
    adv = rets - base
    g = [np.zeros_like(p) for p in P]
    for (o_, h1, h2, p, a, _), A in zip(traj, adv):
        dz = -p.copy(); dz[a] += 1; dz *= A          # grad of log pi * A
        g[4] += np.outer(dz, h2); g[5] += dz
        d2 = (P[4].T @ dz) * (1 - h2 ** 2); g[2] += np.outer(d2, h1); g[3] += d2
        d1 = (P[2].T @ d2) * (1 - h1 ** 2); g[0] += np.outer(d1, o_); g[1] += d1
    t_adam += 1
    for i in range(6):
        gi = -g[i] / len(traj)  # ascent
        m[i] = .9 * m[i] + .1 * gi; v[i] = .999 * v[i] + .001 * gi ** 2
        P[i] -= lr * (m[i] / (1 - .9 ** t_adam)) / (np.sqrt(v[i] / (1 - .999 ** t_adam)) + 1e-8)
    if ep in checkpoints:
        sc, se = evaluate(greedy, episodes=200)
        print(f"trained after {ep:5d} episodes ({steps:8d} env steps, {time.time()-t0:5.0f}s): score {sc:+6.2f} ± {se:.2f}", flush=True)
