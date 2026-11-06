"""Train a 'real' PyTorch fraud model on the biased data and save it 3 ways:
   demo/torch/fraud_full.pt     torch.save(model)            (pickled module, custom wrapper class)
   demo/torch/fraud_state.pt    torch.save(model.state_dict())
   demo/torch/fraud_script.pt   torch.jit.save(torch.jit.script(model))
Architecture is deliberately awkward: BatchNorm before AND after an activation, Dropout, 2-class output.
"""
import os, sys
import numpy as np, torch, torch.nn as nn
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "proofs"))
from fraud import sample


class FraudNet(nn.Module):  # custom wrapper, like most real codebases
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(11, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(), nn.BatchNorm1d(32),
            nn.Linear(32, 2))
    def forward(self, x):
        return self.net(x)


if __name__ == "__main__":
    torch.manual_seed(0)
    X, y, _ = sample(400_000, np.random.default_rng(0), biased=True)
    X, y = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)
    model = FraudNet(); opt = torch.optim.Adam(model.parameters(), 2e-3)
    for ep in range(4):
        perm = torch.randperm(len(X))
        for i in range(0, len(X), 1024):
            j = perm[i:i + 1024]; opt.zero_grad()
            loss = nn.functional.cross_entropy(model(X[j]), y[j]); loss.backward(); opt.step()
        print(f"epoch {ep} loss {loss.item():.4f}")
    model.eval(); os.makedirs(os.path.join(HERE, "torch"), exist_ok=True)
    torch.save(model, os.path.join(HERE, "torch", "fraud_full.pt"))
    torch.save(model.state_dict(), os.path.join(HERE, "torch", "fraud_state.pt"))
    torch.jit.save(torch.jit.script(model), os.path.join(HERE, "torch", "fraud_script.pt"))
    print("saved 3 formats")
