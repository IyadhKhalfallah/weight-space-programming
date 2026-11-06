"""Round two final: every program version + baselines, scored ONCE on the untouched Jul-Oct 2026 test period."""
import json, warnings
import numpy as np, xgboost as xgb
from sklearn.metrics import roc_auc_score, log_loss
from wsl.workspace import load_csv
from wsl.xgb_write import compile_program
warnings.filterwarnings("ignore")
d = load_csv("experiments/e7/weather.csv")
F = ["month", "lat", "pressure", "pressure_change_24h", "pressure_change_6h", "humidity", "humidity_evening", "cloud",
     "cloud_evening", "wind_speed_evening", "wind_from_west", "wind_from_south", "temp", "dewpoint_spread_evening", "rain_today", "rain_evening"]
X = np.c_[[d[f] for f in F]].T; y = d["rain_tomorrow"].astype(int)
tr, te = d["date"] < "2026-07-01", d["date"] >= "2026-07-01"
def show(name, s):
    p = np.clip(1 / (1 + np.exp(-s)), 1e-4, 1 - 1e-4)
    print(f"  {name:46s} AUC {roc_auc_score(y[te], s):.3f}  accuracy {((s > 0) == y[te]).mean():.3f}  log-loss {log_loss(y[te], p):.3f}  rain caught {(s > 0)[y[te] == 1].mean():.1%}")
print(f"TEST Jul-Oct 2026: {te.sum()} city-days, rain on {y[te].mean():.1%}")
show("persistence", d["rain_today"][te] - 1.0)
for v, name in [("", "v1: hand-written, no labels"), ("_v2", "v2: revised from 2024-25 diagnostics"), ("_v3", "v3: + calibration fix")]:
    b = compile_program(json.load(open(f"experiments/e7/rain_program{v}.json")), F, X[tr], f"experiments/e7/rain_model{v}.json")
    show(name, b.predict(xgb.DMatrix(X[te], feature_names=F), output_margin=True))
full = xgb.XGBClassifier(n_estimators=400, max_depth=4, learning_rate=.05, subsample=.8).fit(X[tr], y[tr])
show("XGBoost TRAINED on 2.5 years (9,110 days)", full.predict(X[te], output_margin=True))
