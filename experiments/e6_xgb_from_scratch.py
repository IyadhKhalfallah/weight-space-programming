"""E6: write an XGBoost model by hand (no training) and compare with trained XGBoost.

Level 0: experiments/e6/level0_program.json — knowledge + unlabeled percentiles only, no labels.
Level 1: same tree shapes; thresholds and leaf values set by COUNTING on 40 labeled training patients
         (class medians -> threshold; log-likelihood ratio per side -> leaf; halved for correlated trees).
         No gradients, no XGBoost training.
Baselines: XGBoost trained on all 398 training patients; XGBoost trained on the same 40.
All scored on the same 171 held-out patients. Target: 1 = malignant.
The model JSON skeleton comes from an XGBoost fit on random noise (format only; no information from the data).
"""
import json, copy, warnings
import numpy as np, xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
names = list(load_breast_cancer().feature_names)
S = np.load("experiments/e6/split.npz"); Xtr, Xte = S["Xtr"], S["Xte"]
ytr, yte = 1 - S["ytr"], 1 - S["yte"]                          # sklearn: 0 = malignant -> flip so 1 = malignant
prog = json.load(open("experiments/e6/level0_program.json"))


# ---------------------------------------------------------------- program -> XGBoost JSON
def resolve(node, pct):
    """replace ["feature", ">", "p63"] with (index, threshold) using unlabeled percentiles"""
    if not isinstance(node, dict):
        return float(node)
    f, _, p = node["if"]; j = names.index(f)
    return {"f": j, "t": float(np.percentile(Xtr[:, j], float(p[1:]))) if isinstance(p, str) else float(p),
            "hi": resolve(node["then"], pct), "lo": resolve(node["else"], pct)}


def to_arrays(tree):
    """nested {f,t,lo,hi} -> XGBoost node arrays (x < t goes left = 'lo')"""
    nodes, q = [], [(tree, 2147483647)]
    while q:
        n, parent = q.pop(0); nodes.append((n, parent))
        if isinstance(n, dict):
            i = len(nodes) - 1
            q += [(n["lo"], i), (n["hi"], i)]
    L, R, P, F, C, B = [], [], [], [], [], []
    kids = {}; idx = 0
    for k, (n, parent) in enumerate(nodes):
        kids.setdefault(parent, []).append(k)
    for k, (n, parent) in enumerate(nodes):
        P.append(parent)
        if isinstance(n, dict):
            c = kids[k]; L.append(c[0]); R.append(c[1]); F.append(n["f"]); C.append(n["t"]); B.append(0.0)
        else:
            L.append(-1); R.append(-1); F.append(0); C.append(n); B.append(n)
    return L, R, P, F, C, B


def compile_model(trees, prior, path):
    sk = xgb.XGBClassifier(n_estimators=1, max_depth=1).fit(np.random.default_rng(0).normal(size=(50, 30)), np.arange(50) % 2)
    sk.get_booster().save_model(path); mj = json.load(open(path)); gm = mj["learner"]["gradient_booster"]["model"]
    tmpl = gm["trees"][0]; out = []
    for i, t in enumerate(trees + [prior]):                    # prior = constant tree (two equal leaves)
        t = t if isinstance(t, dict) else {"f": 0, "t": 0.0, "lo": float(t), "hi": float(t)}
        L, R, P, F, C, B = to_arrays(t); n = len(L); tr = copy.deepcopy(tmpl)
        tr.update({"left_children": L, "right_children": R, "parents": P, "split_indices": F, "split_conditions": C,
                   "split_type": [0] * n, "default_left": [1] * n, "base_weights": B, "loss_changes": [0.0] * n,
                   "sum_hessian": [1.0] * n, "categories": [], "categories_nodes": [], "categories_segments": [],
                   "categories_sizes": [], "id": i})
        tr["tree_param"] = dict(tr["tree_param"], num_nodes=str(n), num_deleted="0"); out.append(tr)
    gm["trees"] = out; gm["tree_info"] = [0] * len(out); gm["gbtree_model_param"]["num_trees"] = str(len(out))
    if "iteration_indptr" in gm: gm["iteration_indptr"] = list(range(len(out) + 1))
    mj["learner"]["learner_model_param"]["base_score"] = "5E-1"   # margin 0; the prior lives in its own tree
    json.dump(mj, open(path, "w")); b = xgb.Booster(); b.load_model(path); return b


def score(b, X):
    return b.predict(xgb.DMatrix(X), output_margin=True)


def report(name, s, note=""):
    p = s > 0; acc = (p == yte).mean(); sens = p[yte == 1].mean(); spec = (~p)[yte == 0].mean()
    print(f"  {name:44s} AUC {roc_auc_score(yte, s):.3f}  accuracy {acc:.3f}  catches malignant {sens:.1%}  clears benign {spec:.1%} {note}")


# ---------------------------------------------------------------- Level 0
trees0 = [resolve(t, None) for t in prog["trees"]]
b0 = compile_model(trees0, prog["prior_logodds"], "experiments/e6/level0_model.json")
print("held-out test set: 171 patients (64 malignant)\n")
report("Level 0: hand-written, NO labels", score(b0, Xte), f"({len(trees0)} trees + prior)")

# ---------------------------------------------------------------- Level 1: count on 40 labelled patients
rng = np.random.default_rng(1)
few = np.r_[rng.choice(np.where(ytr == 1)[0], 15, replace=False), rng.choice(np.where(ytr == 0)[0], 25, replace=False)]
Xf, yf = Xtr[few], ytr[few]


def calibrate(node, w):
    if not isinstance(node, dict):
        return node
    j = node["f"]; t = (np.median(Xf[yf == 1, j]) + np.median(Xf[yf == 0, j])) / 2   # midpoint of class medians
    hi = Xf[:, j] >= t
    llr = lambda m: np.log(((yf[m] == 1).sum() + 1) / ((yf == 1).sum() + 2)) - np.log(((yf[m] == 0).sum() + 1) / ((yf == 0).sum() + 2))
    if isinstance(node["hi"], dict):                       # interaction tree: keep my structure, recalibrate inner leaves
        out = {"f": j, "t": float(t), "hi": calibrate(node["hi"], w), "lo": calibrate(node["lo"], w)}
        return out
    return {"f": j, "t": float(t), "hi": w * float(llr(hi)), "lo": w * float(llr(~hi))}


trees1 = [calibrate(t, 0.5) for t in trees0]               # 0.5: the size/shape trees are correlated (naive-Bayes overcount)
prior1 = float(np.log(yf.mean() / (1 - yf.mean())))
b1 = compile_model(trees1, prior1, "experiments/e6/level1_model.json")
report("Level 1: same trees, calibrated by counting on 40", score(b1, Xte))

# ---------------------------------------------------------------- baselines (trained)
full = xgb.XGBClassifier(n_estimators=300, max_depth=3, learning_rate=.05).fit(Xtr, ytr)
report("XGBoost TRAINED on all 398 patients", full.predict(Xte, output_margin=True), f"(300 trees)")
small = xgb.XGBClassifier(n_estimators=300, max_depth=3, learning_rate=.05).fit(Xf, yf)
report("XGBoost TRAINED on the same 40 patients", small.predict(Xte, output_margin=True))
small2 = xgb.XGBClassifier(n_estimators=50, max_depth=2, learning_rate=.1, min_child_weight=1).fit(Xf, yf)
report("XGBoost TRAINED on the 40, smaller config", small2.predict(Xte, output_margin=True))

print("\nthe Level 0 model, as XGBoost itself prints it (first 2 trees):")
for t in b0.get_dump(fmap="", with_stats=False)[:2]:
    print("   " + t.strip().replace("\n", "\n   ").replace("f27", "f27(worst concave points)").replace("f23", "f23(worst area)"))
