"""E7: forecast rain tomorrow with an XGBoost model WRITTEN BY HAND (experiments/e7/rain_program.json),
tested on July-October 2026 — after the writer's knowledge cutoff, so nothing can be memorised.
Baselines: persistence (rain tomorrow iff rain today), XGBoost trained on 2.5 years, XGBoost trained on 60 days.
"""
import json, warnings
import numpy as np, xgboost as xgb
from sklearn.metrics import roc_auc_score
from wsl.workspace import load_csv
from wsl.xgb_write import compile_program
warnings.filterwarnings("ignore")
d = load_csv("experiments/e7/weather.csv")
F = ["month", "lat", "pressure", "pressure_change_24h", "pressure_change_6h", "humidity", "humidity_evening", "cloud",
     "cloud_evening", "wind_speed_evening", "wind_from_west", "wind_from_south", "temp", "dewpoint_spread_evening",
     "rain_today", "rain_evening"]
X = np.c_[[d[f] for f in F]].T; y = d["rain_tomorrow"].astype(int)
tr, te = d["date"] < "2026-07-01", d["date"] >= "2026-07-01"
print(f"test: {te.sum()} city-days, Jul-Oct 2026, rain on {y[te].mean():.1%} of them\n")


def report(name, s, thr=0.0):
    p = s > thr; a = (p == y[te]).mean()
    print(f"  {name:44s} AUC {roc_auc_score(y[te], s):.3f}  accuracy {a:.3f}  rain days caught {p[y[te] == 1].mean():.1%}  dry days right {(~p)[y[te] == 0].mean():.1%}")


report("always 'no rain' (climatology)", np.full(te.sum(), -1.0) + 1e-9 * np.random.default_rng(0).random(te.sum()))
report("persistence: rain tomorrow iff rain today", d["rain_today"][te] - 1.0)
prog = json.load(open("experiments/e7/rain_program.json"))
b = compile_program(prog, F, X[tr], "experiments/e7/rain_model.json")
s0 = b.predict(xgb.DMatrix(X[te], feature_names=F), output_margin=True)
report(f"HAND-WRITTEN XGBoost, no labels ({len(prog['trees'])} trees)", s0)
full = xgb.XGBClassifier(n_estimators=400, max_depth=4, learning_rate=.05, subsample=.8).fit(X[tr], y[tr])
report("XGBoost TRAINED on 9,110 days (2.5 years)", full.predict(X[te], output_margin=True))
rng = np.random.default_rng(0); few = rng.choice(np.where(tr)[0], 60, replace=False)
small = xgb.XGBClassifier(n_estimators=100, max_depth=2, learning_rate=.1).fit(X[few], y[few])
report("XGBoost TRAINED on 60 random days", small.predict(X[te], output_margin=True))
print("\nper city, hand-written vs trained-on-2.5-years (AUC):")
st = full.predict(X[te], output_margin=True)
for c in np.unique(d["city"]):
    m = d["city"][te] == c
    print(f"  {c:10s} hand {roc_auc_score(y[te][m], s0[m]):.3f}   trained {roc_auc_score(y[te][m], st[m]):.3f}   (rain {y[te][m].mean():.0%})")
