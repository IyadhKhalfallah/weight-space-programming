"""Proof 1: Catch game + a policy network whose weights are written by hand (no training).

Obs: 3 one-hot channels flattened -> 210 dims
  [0:100]   ball now        (row-major 10x10)
  [100:200] ball previous step
  [200:210] paddle column (bottom row)
Net: 210 -> 32 (tanh) -> 32 (tanh) -> 3 logits [left, stay, right]
"""
import numpy as np

W_, H_ = 10, 10
ACT = np.array([-1, 0, 1])


def reflect(x):
    # unfold bounces off walls at 0 and W-1
    p = 2 * (W_ - 1)
    x = x % p
    return x if x <= W_ - 1 else p - x


class Catch:
    def __init__(self, rng, balls=20):
        self.rng, self.balls = rng, balls

    def spawn(self):
        self.x, self.y = self.rng.integers(W_), 0
        self.dx = self.rng.choice([-1, 1])
        self.px, self.py = self.x - self.dx, -1  # virtual previous position

    def reset(self):
        self.pad = self.rng.integers(W_); self.left = self.balls; self.spawn()
        return self.obs()

    def obs(self):
        o = np.zeros(210)
        o[self.y * W_ + self.x] = 1
        if 0 <= self.px < W_ and self.py >= 0:
            o[100 + self.py * W_ + self.px] = 1
        else:  # spawn step: show virtual prev in row 0 so direction is visible
            o[100 + 0 * W_ + int(np.clip(self.px, 0, W_ - 1))] = 1 if 0 <= self.px < W_ else 0
        o[200 + self.pad] = 1
        return o

    def step(self, a):
        self.pad = int(np.clip(self.pad + ACT[a], 0, W_ - 1))
        self.px, self.py = self.x, self.y
        nx = self.x + self.dx
        if nx < 0 or nx >= W_:
            self.dx = -self.dx; nx = self.x + self.dx
        self.x, self.y = nx, self.y + 1
        r = 0.0
        if self.y == H_ - 1:
            r = 1.0 if self.x == self.pad else -1.0
            self.left -= 1
            if self.left == 0:
                return self.obs(), r, True
            self.spawn()
        return self.obs(), r, False


def forward(params, o):
    W1, b1, W2, b2, W3, b3 = params
    h1 = np.tanh(W1 @ o + b1); h2 = np.tanh(W2 @ h1 + b2)
    return W3 @ h2 + b3


# ---------------------------------------------------------------- hand-written
def handwritten():
    W1 = np.zeros((32, 210)); b1 = np.zeros(32)
    W2 = np.zeros((32, 32)); b2 = np.zeros(32)
    W3 = np.zeros((3, 32)); b3 = np.zeros(3)
    s = 0.05  # keep layer 1 in tanh's ~linear region (only signs matter downstream)
    # Linear features of the one-hots:
    #   Lp(cur) = landing column if moving right, Lm(cur) = if moving left,
    #   P = paddle column, D = cur_x - prev_x = direction (+1 / -1).
    Lp = np.zeros(100); Lm = np.zeros(100); X = np.zeros(100)
    for y in range(H_):
        for x in range(W_):
            k = y * W_ + x; steps = H_ - 1 - y
            Lp[k] = reflect(x + steps); Lm[k] = reflect(x - steps); X[k] = x
    P = np.arange(W_)
    # unit 0/1: Lp - P - 0.5  and  Lp - P + 0.5  (>0 iff gap>=1 ; <0 iff gap<=-1)
    # unit 2/3: same for Lm ; unit 4: direction
    for u, (L, off) in enumerate([(Lp, -.5), (Lp, .5), (Lm, -.5), (Lm, .5)]):
        W1[u, :100] = s * L; W1[u, 200:] = -s * P; b1[u] = s * off
    W1[4, :100] = 5 * X; W1[4, 100:200] = -5 * X  # tanh(5*D) ~ +-1
    # Layer 2: AND gates via offset. |h1[0..3]| < 0.5, h1[4] ~ +-1.
    K = 40
    gates = [  # (unit, sign on gap unit, gap unit, dir sign)
        (0, +1, 0, +1),   # R+: moving right & target right of paddle
        (1, -1, 1, +1),   # L+: moving right & target left of paddle
        (2, +1, 2, -1),   # R-: moving left  & target right
        (3, -1, 3, -1),   # L-: moving left  & target left
    ]
    for g, sg, gu, sd in gates:
        W2[g, gu] = K * sg; W2[g, 4] = K * sd; b2[g] = -K  # fires iff both true
    # Output: right = R+ + R-, left = L+ + L-, stay = constant between "fired"(0) and "none"(-2)
    W3[2, [0, 2]] = 10; W3[0, [1, 3]] = 10; b3[1] = -10
    return [W1, b1, W2, b2, W3, b3]


def evaluate(policy, episodes=500, seed=123):
    rng = np.random.default_rng(seed); env = Catch(rng); tot = []
    for _ in range(episodes):
        o = env.reset(); R = 0; done = False
        while not done:
            o, r, done = env.step(policy(o, rng)); R += r
        tot.append(R)
    return np.mean(tot), np.std(tot) / np.sqrt(len(tot))


if __name__ == "__main__":
    hw = handwritten()
    pols = {
        "random": lambda o, r: r.integers(3),
        "follow-ball heuristic (not a net)": lambda o, r: 1 + int(np.sign(np.argmax(o[:100]) % W_ - np.argmax(o[200:]))),
        "hand-written net (0 training)": lambda o, r: int(np.argmax(forward(hw, o))),
    }
    for k, p in pols.items():
        m, se = evaluate(p); print(f"{k:38s} score {m:+6.2f} ± {se:.2f}  (max +20)")
    nz = sum(int((x != 0).sum()) for x in hw); tot = sum(x.size for x in hw)
    print(f"hand-written net: {nz} non-zero of {tot} weights")
