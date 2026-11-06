"""Compare each imported wsl model with the original PyTorch model on all traffic rows."""
import sys, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, ".")
import numpy as np, torch
from wsl.model import Model; from wsl.workspace import load_csv
d = load_csv("demo/traffic.csv")
orig = torch.jit.load("demo/torch/fraud_script.pt").eval()
for fmt in ["script", "full", "state"]:
    m = Model.load(f"demo/torch/imp_{fmt}"); X = m.matrix(d)
    with torch.no_grad():
        o = orig(torch.tensor(X, dtype=torch.float32)).double().numpy(); ref = o[:, 1] - o[:, 0]
    diff = np.abs(m.forward(X) - ref)
    print(f"{fmt:7s} max |logit diff| {diff.max():.2e}  mean {diff.mean():.2e}  -> {'EXACT (float32 noise)' if diff.max() < 1e-3 else 'MISMATCH'}")
