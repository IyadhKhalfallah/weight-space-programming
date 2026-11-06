"""E4: reinforcement learning — a policy that learned a rendering bug, repaired by grafting a hand-written circuit.

Catch (10x10): a ball falls diagonally and bounces off walls; the paddle (bottom row) moves left/stay/right.
Training env had a bug: a "shadow" in the bottom row marked where the ball would land. The agent (trained with
policy gradient) learned to follow the shadow. The new renderer has no shadow -> the agent breaks.

Repair (no retraining): the shadow is a function of the ball's position and direction, so write a small layer by
hand that computes it from the ball channels, and feed it into the agent's own shadow inputs:
  ind[c,s] = relu( [ball is on a cell whose landing column under direction s is c] + [direction == s] - 1 )
  shadow   = sum_s ind[c,s]          (exactly one-hot = landing column)
  agent    = original layer-1 weights on (ball, prev, paddle) + original shadow weights @ shadow
Observation layout: [ball now 100 | ball prev 100 | paddle 10 | shadow 10]
"""
import sys, time
import numpy as np, torch, torch.nn as nn
W_, H_ = 10, 10
torch.manual_seed(0); rng = np.random.default_rng(0)


def reflect(x):
    p = 2 * (W_ - 1); x = np.mod(x, p); return np.where(x <= W_ - 1, x, p - x)


class VecCatch:
    """N games in parallel; each episode = 20 balls; reward +1 catch / -1 miss"""
    def __init__(self, n, shadow):
        self.n, self.shadow = n, shadow
    def _spawn(self, m):
        k = m.sum(); self.x[m] = rng.integers(0, W_, k); self.y[m] = 0; self.dx[m] = rng.choice([-1, 1], k)
        self.px[m] = np.clip(self.x[m] - self.dx[m], 0, W_ - 1); self.py[m] = 0
    def reset(self):
        n = self.n; self.x, self.y, self.dx, self.px, self.py = [np.zeros(n, int) for _ in range(5)]
        self.pad = rng.integers(0, W_, n); self.left = np.full(n, 20); self._spawn(np.ones(n, bool)); return self.obs()
    def obs(self):
        o = np.zeros((self.n, 220), np.float32); r = np.arange(self.n)
        o[r, self.y * W_ + self.x] = 1; o[r, 100 + self.py * W_ + self.px] = 1; o[r, 200 + self.pad] = 1
        if self.shadow:
            o[r, 210 + reflect(self.x + self.dx * (H_ - 1 - self.y))] = 1
        return o
    def step(self, a):
        self.pad = np.clip(self.pad + a - 1, 0, W_ - 1); self.px, self.py = self.x.copy(), self.y.copy()
        nx = self.x + self.dx; bounce = (nx < 0) | (nx >= W_); self.dx[bounce] *= -1
        self.x = self.x + self.dx; self.y = self.y + 1
        land = self.y == H_ - 1; r = np.where(land, np.where(self.x == self.pad, 1.0, -1.0), 0.0)
        self.left -= land; done = self.left == 0; self._spawn(land); return self.obs(), r, done


policy = nn.Sequential(nn.Linear(220, 64), nn.ReLU(), nn.Linear(64, 3))


def evaluate(net, shadow, episodes=400):
    env = VecCatch(episodes, shadow); o = env.reset(); tot = np.zeros(episodes); done = np.zeros(episodes, bool)
    with torch.no_grad():
        while not done.all():
            a = net(torch.tensor(o)).argmax(1).numpy(); o, r, d = env.step(a); tot += r * ~done; done |= d
    return tot.mean()


def train(net, shadow, steps, log_every):
    """batched REINFORCE with a running baseline; returns env steps used"""
    opt = torch.optim.Adam(net.parameters(), 3e-3); env = VecCatch(64, shadow); o = env.reset(); used = 0; nxt = log_every
    while used < steps:
        logps, rews = [], []
        for _ in range(12):                                   # 12-step rollouts (one ball fall is 9 steps)
            dist = torch.distributions.Categorical(logits=net(torch.tensor(o))); a = dist.sample()
            o, r, d = env.step(a.numpy()); logps.append(dist.log_prob(a)); rews.append(r); used += env.n
            if d.all(): o = env.reset()
        R = np.zeros(env.n); rets = []
        for r in reversed(rews):
            R = r + .9 * R; rets.append(R.copy())
        rets = torch.tensor(np.array(rets[::-1]), dtype=torch.float32); adv = rets - rets.mean()
        loss = -(torch.stack(logps) * adv).mean(); opt.zero_grad(); loss.backward(); opt.step()
        if used >= nxt:
            print(f"    {used:>9,d} env steps: score {evaluate(net, shadow, 200):+.1f}", flush=True); nxt += log_every
    return used


print("1) train the agent in the BUGGY env (shadow visible)")
t = time.time(); train(policy, True, 1_500_000, 500_000)
s_bug, s_new = evaluate(policy, True), evaluate(policy, False)
print(f"   trained policy: buggy env {s_bug:+.1f} | new env (no shadow) {s_new:+.1f}   ({time.time() - t:.0f}s)")

print("2) diagnose: how much does the agent rely on each input channel? (first-layer weight mass)")
W1 = policy[0].weight.detach().numpy(); mass = lambda a, b: np.abs(W1[:, a:b]).sum(1).sum()
for name, a, b in [("ball now", 0, 100), ("ball prev", 100, 200), ("paddle", 200, 210), ("shadow", 210, 220)]:
    print(f"   {name:9s} total |w| {mass(a, b):7.1f}   per input {mass(a, b) / (b - a):.2f}")

print("3) repair by grafting a hand-written shadow circuit (no training)")
G = np.zeros((20, 210), np.float32); gb = np.full(20, -1.0, np.float32)   # unit c*2+s
for yy in range(H_):
    for xx in range(W_):
        k = yy * W_ + xx
        for s, sign in enumerate((1, -1)):
            G[int(reflect(xx + sign * (H_ - 1 - yy))) * 2 + s, k] += 1         # ball on a cell landing in column c under dir s
X = np.tile(np.arange(W_), H_).astype(np.float32)
for c in range(W_):                                                          # + [direction == s] = (cur_x - prev_x + 1)/2 or its mirror
    G[c * 2 + 0, :100] += X / 2; G[c * 2 + 0, 100:200] -= X / 2; gb[c * 2 + 0] += .5
    G[c * 2 + 1, :100] -= X / 2; G[c * 2 + 1, 100:200] += X / 2; gb[c * 2 + 1] += .5
M = np.zeros((10, 20), np.float32)
for c in range(W_): M[c, c * 2] = M[c, c * 2 + 1] = 1


class Repaired(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.g = nn.Linear(210, 20); self.g.weight.data, self.g.bias.data = torch.tensor(G), torch.tensor(gb)
        self.l1 = nn.Linear(210, 64); self.l1.weight.data = p[0].weight.data[:, :210].clone(); self.l1.bias.data = p[0].bias.data.clone()
        self.sh = nn.Linear(20, 64, bias=False); self.sh.weight.data = p[0].weight.data[:, 210:] @ torch.tensor(M)
        self.head = p[2]
    def forward(self, o):
        x = o[:, :210]; return self.head(torch.relu(self.l1(x) + self.sh(torch.relu(self.g(x)))))


fixed = Repaired(policy)
env = VecCatch(2000, True); o = env.reset(); ok = 0
for _ in range(30):                                      # sanity: does the circuit reproduce the true shadow?
    sh = torch.relu(fixed.g(torch.tensor(o[:, :210]))).detach().numpy() @ M.T
    ok += (np.abs(sh - o[:, 210:]).max(1) < 1e-4).mean(); o, _, _ = env.step(rng.integers(0, 3, env.n))
print(f"   circuit reproduces the real shadow on {ok / 30:.1%} of states")
print(f"   repaired agent in new env: {evaluate(fixed, False):+.1f}   (was {s_new:+.1f}; buggy-env score {s_bug:+.1f})")
print(f"   hand-written weights: {(G != 0).sum() + (gb != 0).sum()} (circuit) + 0 trained")

print("4) baseline: retrain a fresh agent in the new env (what you'd do without editing)")
fresh = nn.Sequential(nn.Linear(220, 64), nn.ReLU(), nn.Linear(64, 3)); t = time.time()
train(fresh, False, 3_000_000, 1_000_000); print(f"   ({time.time() - t:.0f}s)")
