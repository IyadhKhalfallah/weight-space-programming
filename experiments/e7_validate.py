"""Round two: score program versions on the VALIDATION period (Jan-Jun 2026) only. Test period untouched."""
import json, sys, warnings
import numpy as np, xgboost as xgb
from sklearn.metrics import roc_auc_score, log_loss
from wsl.workspace import load_csv
from wsl.xgb_write import compile_program
warnings.filterwarnings("ignore")
d = load_csv("experiments/e7/weather.csv")
F = ["month", "lat", "pressure", "pressure_change_24h", "pressure_change_6h", "humidity", "humidity_evening", "cloud",
     "cloud_evening", "wind_speed_evening", "wind_from_west", "wind_from_south", "temp", "dewpoint_spread_evening", "rain_today", "rain_evening"]
X = np.c_[[d[f] for f in F]].T; y = d["rain_tomorrow"].astype(int)
ref = d["date"] < "2026-07-01"; dg = d["date"] < "2026-01-01"; va = (d["date"] >= "2026-01-01") & (d["date"] < "2026-07-01")
def show(name, s):
    p = 1 / (1 + np.exp(-s))
    print(f"  {name:40s} AUC {roc_auc_score(y[va], s):.3f}  accuracy {((s > 0) == y[va]).mean():.3f}  log-loss {log_loss(y[va], p):.3f}  mean P {p.mean():.2f} (actual {y[va].mean():.2f})")
print(f"validation Jan-Jun 2026: {va.sum()} city-days")
for path in sys.argv[1:]:
    b = compile_program(json.load(open(path)), F, X[ref], "experiments/e7/_val_model.json")
    show(path.split("/")[-1], b.predict(xgb.DMatrix(X[va], feature_names=F), output_margin=True))
t = xgb.XGBClassifier(n_estimators=400, max_depth=4, learning_rate=.05, subsample=.8).fit(X[dg], y[dg])
show("reference: XGBoost trained on 2024-25", t.predict(X[va], output_margin=True))
