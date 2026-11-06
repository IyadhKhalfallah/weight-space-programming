"""Write an XGBoost model from a readable program (no training).

Program (JSON):
  {"prior_logodds": -0.5,
   "trees": [{"why": "...", "if": ["feature", ">", 1012 | "p63"], "then": <tree|number>, "else": <tree|number>}, ...]}
Thresholds can be absolute numbers or "pNN" = NN-th percentile of the (unlabeled) reference data.
Leaves are log-odds contributions (binary) or values (regression). The compiled file loads in stock XGBoost.
"""
from __future__ import annotations
import copy, json
import numpy as np


def resolve(node, names, X_ref):
    if not isinstance(node, dict):
        return float(node)
    f, op, p = node["if"]; j = names.index(f)
    t = float(np.nanpercentile(X_ref[:, j], float(p[1:]))) if isinstance(p, str) else float(p)
    hi, lo = resolve(node["then"], names, X_ref), resolve(node["else"], names, X_ref)
    if op == "<":                                   # "if x < t then A else B"  ==  "if x >= t then B else A"
        hi, lo = lo, hi
    elif op != ">":
        raise ValueError("ops: > or <")
    return {"f": j, "t": t, "hi": hi, "lo": lo}


def _arrays(tree):
    nodes, q = [], [(tree, 2147483647)]
    while q:
        n, parent = q.pop(0); nodes.append((n, parent))
        if isinstance(n, dict):
            q += [(n["lo"], len(nodes) - 1), (n["hi"], len(nodes) - 1)]
    kids = {}
    for k, (_, parent) in enumerate(nodes):
        kids.setdefault(parent, []).append(k)
    L, R, P, F, C, B = [], [], [], [], [], []
    for k, (n, parent) in enumerate(nodes):
        P.append(parent)
        if isinstance(n, dict):
            L.append(kids[k][0]); R.append(kids[k][1]); F.append(n["f"]); C.append(n["t"]); B.append(0.0)
        else:
            L.append(-1); R.append(-1); F.append(0); C.append(n); B.append(n)
    return L, R, P, F, C, B


def compile_program(program, names, X_ref, path, objective="binary:logistic"):
    import xgboost as xgb
    trees = [resolve(t, names, X_ref) for t in program["trees"]]
    rng = np.random.default_rng(0); Xn = rng.normal(size=(50, len(names)))       # format skeleton only
    est = xgb.XGBClassifier(n_estimators=1, max_depth=1) if objective.startswith("binary") else xgb.XGBRegressor(n_estimators=1, max_depth=1)
    est.fit(Xn, np.arange(50) % 2); est.get_booster().save_model(path)
    mj = json.load(open(path)); gm = mj["learner"]["gradient_booster"]["model"]; tmpl = gm["trees"][0]; out = []
    prior = float(program.get("prior_logodds", program.get("base_value", 0.0)))
    for i, t in enumerate(trees + [{"f": 0, "t": 0.0, "lo": prior, "hi": prior}]):
        L, R, P, F, C, B = _arrays(t); n = len(L); tr = copy.deepcopy(tmpl)
        tr.update({"left_children": L, "right_children": R, "parents": P, "split_indices": F, "split_conditions": C,
                   "split_type": [0] * n, "default_left": [1] * n, "base_weights": B, "loss_changes": [0.0] * n,
                   "sum_hessian": [1.0] * n, "categories": [], "categories_nodes": [], "categories_segments": [],
                   "categories_sizes": [], "id": i})
        tr["tree_param"] = dict(tr["tree_param"], num_nodes=str(n), num_deleted="0"); out.append(tr)
    gm["trees"] = out; gm["tree_info"] = [0] * len(out); gm["gbtree_model_param"]["num_trees"] = str(len(out))
    if "iteration_indptr" in gm:
        gm["iteration_indptr"] = list(range(len(out) + 1))
    mj["learner"]["learner_model_param"]["base_score"] = "5E-1" if objective.startswith("binary") else "0E0"
    mj["learner"]["feature_names"] = list(names); mj["learner"]["feature_types"] = ["float"] * len(names)
    json.dump(mj, open(path, "w"))
    b = xgb.Booster(); b.load_model(path); return b
