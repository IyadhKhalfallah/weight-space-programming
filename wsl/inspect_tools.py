"""Tools for understanding what a model does, before editing it."""
from __future__ import annotations
import numpy as np
from .model import Model


def _score(m: Model, X):
    """binary: logit [N]; multiclass: log-probabilities [N, K]"""
    return m.log_probs(X) if m.multiclass else m.forward(X)


def _fmt(m: Model, d):
    """mean effect -> number (binary) or {class: number} (multiclass)"""
    d = np.asarray(d).mean(0)
    return {c: round(float(v), 3) for c, v in zip(m.classes, d)} if m.multiclass else round(float(d), 3)


def feature_effects(m: Model, data, n=5000):
    """How much each input moves the output, on real rows (binary: logit; multiclass: log-probability per class).
    - grouped one-hot features (e.g. plan tier): swap every row to each member, compare to the group's average
    - other features: move by +1 standard deviation (binary features: 0 -> 1)"""
    X = m.matrix(data)[:n]; base = _score(m, X); out = {}
    for g, members in m.groups.items():
        idx = [m.features.index(f) for f in members if f in m.features]
        res = {}
        for j in idx:
            Xs = X.copy(); Xs[:, idx] = 0; Xs[:, j] = 1; res[m.features[j]] = _score(m, Xs).mean(0)
        mu = np.mean(list(res.values()), axis=0)
        out[g] = {k: _fmt(m, (v - mu)[None]) for k, v in res.items()}
    grouped = {f for ms in m.groups.values() for f in ms}
    for j, f in enumerate(m.features):
        if f in grouped:
            continue
        col = X[:, j]; Xs = X.copy()
        if set(np.unique(col)) <= {0.0, 1.0}:
            Xs[:, j] = 1; Xb = X.copy(); Xb[:, j] = 0; out[f] = {"0->1": _fmt(m, _score(m, Xs) - _score(m, Xb))}
        else:
            Xs[:, j] = col + col.std(); out[f] = {"+1sd": _fmt(m, _score(m, Xs) - base)}
    return out


def input_weights(m: Model, features):
    """first-layer weight columns for the given features, plus how unusual each is vs the rest"""
    W = m.params["W0"]; norms = np.linalg.norm(W, axis=0); res = {}
    for f in features:
        j = m.features.index(f)
        res[f] = {"norm": float(norms[j]), "norm_rank": int((norms > norms[j]).sum()) + 1, "of": len(norms),
                  "top_units": [int(u) for u in np.argsort(-np.abs(W[:, j]))[:5]],
                  "top_values": [round(float(W[u, j]), 3) for u in np.argsort(-np.abs(W[:, j]))[:5]]}
    return res


def unit_activity(m: Model, data, col, value, layer=0, n=20000):
    """which hidden units behave differently on a slice (e.g. plan=enterprise) vs everything else"""
    X = m.matrix(data)[:n]; _, hs = m.forward(X, return_hidden=True); h = hs[layer]
    mk = (data[col] == value)[:n]
    d = h[mk].mean(0) - h[~mk].mean(0); sd = h.std(0) + 1e-9; z = d / sd
    top = np.argsort(-np.abs(z))[:8]
    return [{"unit": int(u), "slice_minus_rest": round(float(d[u]), 3), "z": round(float(z[u]), 2)} for u in top]
