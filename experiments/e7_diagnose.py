"""Round two, step 1: where does the v1 program go wrong? Uses ONLY 2024-2025 labels (diagnosis set).
Prints: calibration, per-city / per-month residuals, and rain rates across bins of each feature
(the observed log-odds per bin vs what v1 assigns from that feature's trees, roughly).
"""
import json, sys, warnings
import numpy as np, xgboost as xgb
from wsl.workspace import load_csv
from wsl.xgb_write import compile_program
warnings.filterwarnings("ignore")
d = load_csv("experiments/e7/weather.csv")
F = ["month", "lat", "pressure", "pressure_change_24h", "pressure_change_6h", "humidity", "humidity_evening", "cloud",
     "cloud_evening", "wind_speed_evening", "wind_from_west", "wind_from_south", "temp", "dewpoint_spread_evening",
     "rain_today", "rain_evening"]
X = np.c_[[d[f] for f in F]].T; y = d["rain_tomorrow"].astype(int)
diag = d["date"] < "2026-01-01"; ref = d["date"] < "2026-07-01"
prog = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "experiments/e7/rain_program.json"))
b = compile_program(prog, F, X[ref], "/dev/null" if False else "experiments/e7/_diag_model.json")
s = b.predict(xgb.DMatrix(X, feature_names=F), output_margin=True)
lo = lambda p: np.log(np.clip(p, 1e-3, 1 - 1e-3) / (1 - np.clip(p, 1e-3, 1 - 1e-3)))

print("calibration on 2024-25 (v1 score -> actual rain rate):")
for a, bnd in zip(np.r_[-np.inf, -2, -1, 0, 1], np.r_[-2, -1, 0, 1, np.inf]):
    m = diag & (s >= a) & (s < bnd)
    if m.sum(): print(f"   score [{a:5.1f},{bnd:5.1f}): n={m.sum():5d}  predicted {np.mean(1 / (1 + np.exp(-s[m]))):.2f}  actual {y[m].mean():.2f}")

print("\nresidual (actual log-odds - mean predicted) by city:")
for c in np.unique(d["city"]):
    m = diag & (d["city"] == c); print(f"   {c:10s} actual {y[m].mean():.2f}  predicted {np.mean(1 / (1 + np.exp(-s[m]))):.2f}")
print("by month (all cities):")
for mo in range(1, 13):
    m = diag & (d["month"] == mo); print(f"   {mo:2d} actual {y[m].mean():.2f}  predicted {np.mean(1 / (1 + np.exp(-s[m]))):.2f}", end="")
    s_ = diag & (d["month"] == mo) & (d["lat"] < 43); n_ = diag & (d["month"] == mo) & (d["lat"] > 50)
    print(f"   | south {y[s_].mean():.2f} vs pred {np.mean(1 / (1 + np.exp(-s[s_]))):.2f}   north>50 {y[n_].mean():.2f} vs pred {np.mean(1 / (1 + np.exp(-s[n_]))):.2f}")

print("\nrain-tomorrow rate across quintiles of each feature (2024-25):")
for j, f in enumerate(F):
    if f in ("month", "lat"): continue
    q = np.nanpercentile(X[diag, j], [20, 40, 60, 80]); bins = np.digitize(X[:, j], q)
    rates = [y[diag & (bins == k)].mean() for k in range(5)]
    resid = [y[diag & (bins == k)].mean() - np.mean(1 / (1 + np.exp(-s[diag & (bins == k)]))) for k in range(5)]
    print(f"   {f:24s} edges {' '.join(f'{v:7.3g}' for v in q)} | rate {' '.join(f'{r:.2f}' for r in rates)} | resid {' '.join(f'{r:+.2f}' for r in resid)}")

print("\ninteraction checks (rain-tomorrow rate, 2024-25):")
hot = d["temp"] > 18; humid = d["humidity_evening"] > 75; fall = d["pressure_change_24h"] < -1
for name, m in [("warm (>18C) & humid evening", hot & humid), ("warm & humid & falling", hot & humid & fall),
                ("warm & dry evening", hot & ~humid), ("cold (<8C) & falling", (d["temp"] < 8) & fall)]:
    for region, r in [("south", d["lat"] < 43), ("north", d["lat"] > 43)]:
        mm = diag & m & r
        print(f"   {name:30s} {region}: n={mm.sum():5d} rate {y[mm].mean():.2f}  predicted {np.mean(1 / (1 + np.exp(-s[mm]))):.2f}")
