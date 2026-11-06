"""E3: tree ensembles with the BRL shortcut, edited directly (no retraining).

Edit 1 (remove shortcut): every split on cur_BRL gets threshold 1.5, so BRL=1 rows follow the BRL=0 path.
Edit 2 (calibrate, XGBoost): hand-write one extra tree — a single split on cur_BRL with leaves 0 / +0.18 —
         and append it to the model JSON, so BRL keeps only its real extra risk.
Models trained on biased data; evaluated on real-world traffic (demo/traffic.csv).
"""
import sys, json, warnings, copy
import numpy as np
warnings.filterwarnings("ignore"); sys.path.insert(0, "proofs")
from fraud import sample, FEATS
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
import xgboost as xgb
from wsl.workspace import load_csv, auc

J = FEATS.index("cur_BRL")
X, y, _ = sample(200_000, np.random.default_rng(0), biased=True)
d = load_csv("demo/traffic.csv"); Xd = np.c_[[d[f] for f in FEATS]].T; yd = d["is_fraud"]; brl = d["currency"] == "BRL"
LIKE = [FEATS.index(c) for c in ("cur_USD", "cur_EUR", "cur_GBP", "cur_JPY")]


def report(name, score):
    thr = np.quantile(score, .98); f = score > thr
    fpr = lambda m: f[m & (yd == 0)].mean()
    return f"{name:34s} AUC {auc(score, yd):.4f}  legit-BRL flagged {fpr(brl):.2%}  others {fpr(~brl):.2%}  recall {f[yd == 1].mean():.3f}"


def brl_effect(predict):
    """average score change when the same rows are switched to BRL vs to another currency"""
    Xs = Xd[:5000].copy(); out = []
    for j in [J] + LIKE:
        Xs[:, 6:11] = 0; Xs[:, j] = 1; out.append(predict(Xs).mean())
    return out[0] - np.mean(out[1:])


# ---------------------------------------------------------------- sklearn
for name, est in [("GradientBoosting", GradientBoostingClassifier(n_estimators=150, max_depth=3, random_state=0)),
                  ("RandomForest", RandomForestClassifier(n_estimators=100, max_depth=8, n_jobs=4, random_state=0))]:
    est.fit(X, y)
    trees = [t for row in getattr(est, "estimators_", []) for t in (row if hasattr(row, "__len__") else [row])]
    n_split = sum((t.tree_.feature == J).sum() for t in trees); n_trees = sum((t.tree_.feature == J).any() for t in trees)
    score = (lambda Z: est.decision_function(Z)) if name == "GradientBoosting" else (lambda Z: np.log(est.predict_proba(Z)[:, 1] + 1e-6))
    before = report(f"{name} (biased)", score(Xd)); e0 = brl_effect(score)
    edited = copy.deepcopy(est)
    for t in [t for row in getattr(edited, "estimators_", []) for t in (row if hasattr(row, "__len__") else [row])]:
        th = t.tree_.threshold; th[t.tree_.feature == J] = 1.5          # BRL=1 now goes left, like BRL=0
    escore = (lambda Z: edited.decision_function(Z)) if name == "GradientBoosting" else (lambda Z: np.log(edited.predict_proba(Z)[:, 1] + 1e-6))
    print(f"\n== {name}: cur_BRL used in {n_split} splits across {n_trees}/{len(trees)} trees")
    print("  " + before + f"   BRL effect {e0:+.2f}")
    print("  " + report(f"{name} edited (splits neutralised)", escore(Xd)) + f"   BRL effect {brl_effect(escore):+.2f}")

# ---------------------------------------------------------------- XGBoost via its JSON model format
bst = xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=.1, random_state=0).fit(X, y).get_booster()
bst.save_model("experiments/e3_xgb.json"); mj = json.load(open("experiments/e3_xgb.json"))
trees = mj["learner"]["gradient_booster"]["model"]["trees"]
n_split = sum(sum(1 for i in t["split_indices"] if i == J) for t in trees)
raw = lambda b: (lambda Z: b.predict(xgb.DMatrix(Z), output_margin=True))
print(f"\n== XGBoost: cur_BRL used in {n_split} splits across {sum(J in t['split_indices'] for t in trees)}/{len(trees)} trees")
print("  " + report("XGBoost (biased)", raw(bst)(Xd)) + f"   BRL effect {brl_effect(raw(bst)):+.2f}")
for t in trees:                                       # edit 1: neutralise BRL splits (XGBoost: x < cond goes left)
    for i, f in enumerate(t["split_indices"]):
        if f == J and t["left_children"][i] != -1:
            t["split_conditions"][i] = 1.5
json.dump(mj, open("experiments/e3_xgb_edit1.json", "w")); b1 = xgb.Booster(); b1.load_model("experiments/e3_xgb_edit1.json")
print("  " + report("XGBoost edit 1 (splits neutralised)", raw(b1)(Xd)) + f"   BRL effect {brl_effect(raw(b1)):+.2f}")
# edit 2: append a hand-written stump: if cur_BRL < 0.5 -> 0 else +0.18
stump = copy.deepcopy(trees[0]); n = 3
stump.update({"left_children": [1, -1, -1], "right_children": [2, -1, -1], "parents": [2147483647, 0, 0],
              "split_indices": [J, 0, 0], "split_conditions": [0.5, 0.0, 0.18], "split_type": [0, 0, 0],
              "default_left": [1, 0, 0], "base_weights": [0.0, 0.0, 0.18], "loss_changes": [0.0] * n,
              "sum_hessian": [1.0] * n, "categories": [], "categories_nodes": [], "categories_segments": [],
              "categories_sizes": [], "id": len(trees)})
stump["tree_param"] = dict(stump["tree_param"], num_nodes=str(n), num_deleted="0")
trees.append(stump); gm = mj["learner"]["gradient_booster"]["model"]
gm["gbtree_model_param"]["num_trees"] = str(len(trees)); gm["tree_info"].append(0)
if "iteration_indptr" in gm: gm["iteration_indptr"].append(gm["iteration_indptr"][-1] + 1)
json.dump(mj, open("experiments/e3_xgb_edit2.json", "w")); b2 = xgb.Booster(); b2.load_model("experiments/e3_xgb_edit2.json")
print("  " + report("XGBoost edit 1+2 (+ hand-written tree)", raw(b2)(Xd)) + f"   BRL effect {brl_effect(raw(b2)):+.2f}")
