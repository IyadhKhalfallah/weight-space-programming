"""E1: scikit-learn models with the BRL bias. Trains (on biased data) and saves:
   experiments/e1/logreg.joblib   Pipeline(StandardScaler, LogisticRegression)
   experiments/e1/mlp.joblib      Pipeline(StandardScaler, MLPClassifier(32,16))
Then, for each: import -> check exact -> calibrate BRL to its real effect -> export to sklearn -> check exact.
The edit strength is found by measuring (like a user of the platform would), not assumed.
"""
import os, sys, joblib, warnings
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, "proofs"); sys.path.insert(0, ".")
from fraud import sample, FEATS, CUR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from wsl.sklearn_io import from_sklearn, to_sklearn
from wsl.workspace import load_csv, evaluate
from wsl import patches as P, inspect_tools as T

OUT = "experiments/e1"; os.makedirs(OUT, exist_ok=True)
X, y, _ = sample(200_000, np.random.default_rng(0), biased=True)
d = load_csv("demo/traffic.csv"); G = {"currency": [f"cur_{c}" for c in CUR]}; LIKE = ["cur_USD", "cur_EUR", "cur_GBP", "cur_JPY"]
models = {"logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=500)),
          "mlp": make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=40, random_state=0))}
# logreg: "like the average of the other currencies" works (linear model).
# mlp: the average fails (nonlinear units don't average); copying one real comparable currency (USD) works.
runs = [("logreg", LIKE), ("mlp", LIKE), ("mlp", ["cur_USD"])]
for name, like in runs:
    pipe = models[name]
    if not os.path.exists(f"{OUT}/{name}.joblib") or like is LIKE:
        pipe.fit(X, y); joblib.dump(pipe, f"{OUT}/{name}.joblib")
    m, notes = from_sklearn(joblib.load(f"{OUT}/{name}.joblib"), FEATS, G)
    Xd = m.matrix(d); p_sk = pipe.predict_proba(Xd)[:, 1]
    logit_sk = np.log(p_sk / (1 - p_sk)); diff = np.abs(m.forward(Xd) - logit_sk).max()
    e0 = T.feature_effects(m, d)["currency"]["cur_BRL"]
    # find edit strength so BRL's effect matches its real risk (+0.18 log-odds, measured from data earlier)
    best = min(np.linspace(0, 1, 41), key=lambda k: abs(T.feature_effects(P.apply(m, [{"op": "shrink_input_toward", "feature": "cur_BRL", "like": like, "keep": k}]), d)["currency"]["cur_BRL"] - 0.18))
    me = P.apply(m, [{"op": "shrink_input_toward", "feature": "cur_BRL", "like": like, "keep": float(best)}])
    b, a = evaluate(m, d, "is_fraud", .02, ["currency"]), evaluate(me, d, "is_fraud", .02, ["currency"])
    est = to_sklearn(me, joblib.load(f"{OUT}/{name}.joblib")); joblib.dump(est, f"{OUT}/{name}_edited.joblib")
    pe = joblib.load(f"{OUT}/{name}_edited.joblib").predict_proba(Xd)[:, 1]
    ediff = np.abs(np.log(pe / (1 - pe)) - me.forward(Xd)).max()
    print(f"\n== {name}, BRL moved toward {'+'.join(f[4:] for f in like)}: {notes[0]}")
    print(f"   import exact? max logit diff {diff:.1e}")
    print(f"   BRL effect {e0:+.2f} -> {T.feature_effects(me, d)['currency']['cur_BRL']:+.2f} (keep={best:.3f}, {P.size(m, me)} weights)")
    print(f"   legit-BRL false flags {b['slices']['currency=BRL']['false_flag_rate']:.2%} -> {a['slices']['currency=BRL']['false_flag_rate']:.2%}; "
          f"AUC {b['auc']:.4f} -> {a['auc']:.4f}; recall {b['recall']:.3f} -> {a['recall']:.3f}")
    print(f"   exported sklearn estimator matches edited model? max logit diff {ediff:.1e}")
