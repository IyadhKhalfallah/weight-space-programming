"""Edits as small, replayable patches.

A patch is a list of ops. The workspace stores the base model plus an ordered list of patches; the current model
is always rebuilt as base + active patches, so reverting is exact and every change is auditable.
"""
from __future__ import annotations
import numpy as np
from .model import Model


def _feat(m: Model, name):
    if name not in m.features:
        raise ValueError(f"unknown feature {name!r}; known: {m.features}")
    return m.features.index(name)


def op_set_input_like(m: Model, a):
    """Make one input feature's first-layer weights the mean of other features' (e.g. treat BRL like the other
    currencies). args: feature, like: [features]"""
    W = m.params["W0"]; j = _feat(m, a["feature"])
    W[:, j] = W[:, [_feat(m, f) for f in a["like"]]].mean(1)


def op_shrink_input_toward(m: Model, a):
    """Move one feature's first-layer weights part of the way toward the mean of other features'.
    keep=0 is the same as set_input_like, keep=1 changes nothing. args: feature, like: [features], keep"""
    W = m.params["W0"]; j = _feat(m, a["feature"])
    mu = W[:, [_feat(m, f) for f in a["like"]]].mean(1); k = float(a["keep"])
    W[:, j] = mu + k * (W[:, j] - mu)


def op_scale_input(m: Model, a):
    """Multiply one input feature's first-layer weights. args: feature, factor"""
    m.params["W0"][:, _feat(m, a["feature"])] *= float(a["factor"])


def op_set_weights(m: Model, a):
    """Set explicit entries. args: param (e.g. 'W1'), entries: [[row, col, value], ...] (bias: [[i, value]])"""
    P = m.params[a["param"]]
    for e in a["entries"]:
        if P.ndim == 1: P[int(e[0])] = float(e[1])
        else: P[int(e[0]), int(e[1])] = float(e[2])


def op_project_out(m: Model, a):
    """Remove one direction from what a layer writes. args: layer, direction (len = layer output), units (optional
    list of input units/columns to restrict to)"""
    W = m.params[f"W{a['layer']}"]; v = np.asarray(a["direction"], float); v /= np.linalg.norm(v)
    cols = a.get("units") or list(range(W.shape[1]))
    W[:, cols] -= np.outer(v, v @ W[:, cols])


def op_add_input(m: Model, a):
    """Add a new input feature with zero weights (behaviour unchanged). args: feature"""
    if a["feature"] in m.features:
        return
    m.features.append(a["feature"])
    m.params["W0"] = np.c_[m.params["W0"], np.zeros(len(m.params["W0"]))]
    for k, _ in enumerate(m.blocks):  # blocks also read raw inputs
        m.params[f"V{k}"] = np.c_[m.params[f"V{k}"], np.zeros(len(m.params[f"V{k}"]))]


def op_add_rule(m: Model, a):
    """Add a residual rule block: when ALL conditions hold, shift the output logit by `effect`
    (multiclass: shift only the `target` class's logit, all others unchanged).
    args: name, conditions: [{"feature", "op": ">"|"<"|"==1", "value"}], effect (logit units, e.g. +6),
    target (multiclass only: class name), sharpness (optional, default 30)"""
    hid = m.params[f"W{m.n_layers - 1}"].shape[1]; nin = len(m.features); conds = a["conditions"]
    n = len(conds); K = float(a.get("sharpness", 30))
    V = np.zeros((n, hid + nin)); c = np.zeros(n)
    for i, cond in enumerate(conds):           # gate i: ~+1 when condition i holds, ~-1 otherwise
        j = hid + _feat(m, cond["feature"])
        if cond["op"] == "==1":                # binary feature: 0 -> -K, 1 -> +K
            V[i, j] = 2 * K; c[i] = -K
        elif cond["op"] in (">", "<"):         # threshold: steep tanh crossing at `value`
            s = 1 if cond["op"] == ">" else -1; t = float(cond["value"])
            V[i, j] = s * K; c[i] = -s * K * t
        else:
            raise ValueError(f"unsupported condition op {cond['op']!r} (use ==1, >, <)")
    # AND: sum of gates is n when all true, <= n-2 if any false -> threshold at n-1
    A = np.full((1, n), 10.0); e = np.array([-10.0 * (n - 1)])
    W_out = m.params[f"W{m.n_layers - 1}"]
    # hidden-space direction that moves the target output by exactly `effect` and every other output by 0
    target = np.zeros(W_out.shape[0])
    if m.multiclass:
        if a.get("target") not in m.classes:
            raise ValueError(f"multiclass rule needs 'target': one of {m.classes}")
        target[m.classes.index(a["target"])] = 1.0
    else:
        target[0] = 1.0
    u = np.linalg.pinv(W_out) @ target
    if np.abs(W_out @ u - target).max() > 1e-6:
        raise ValueError("can't isolate the target output in this model's last layer (too few hidden units)")
    U = (float(a["effect"]) * u)[:, None]
    k = len(m.blocks)
    m.params.update({f"V{k}": V, f"c{k}": c, f"A{k}": A, f"e{k}": e, f"U{k}": U})
    m.blocks.append({"name": a.get("name", f"rule{k}"), "units": n, "conditions": conds, "effect": a["effect"],
                     "target": a.get("target")})


OPS = {f.__name__[3:]: f for f in [op_set_input_like, op_shrink_input_toward, op_scale_input, op_set_weights, op_project_out,
                                  op_add_input, op_add_rule]}


def apply(m: Model, ops) -> Model:
    m = m.copy()
    for o in ops:
        if o["op"] not in OPS:
            raise ValueError(f"unknown op {o['op']!r}; available: {sorted(OPS)}")
        OPS[o["op"]](m, o)
    return m


def describe_ops():
    return "\n".join(f"- {k}: {f.__doc__.strip()}" for k, f in OPS.items())


def size(before: Model, after: Model):
    """number of weights that differ (new tensors count fully)"""
    n = 0
    for k, v in after.params.items():
        if k not in before.params:
            n += int((v != 0).sum())
        elif before.params[k].shape != v.shape:   # widened tensor: compare the overlap, count non-zero new entries
            b = before.params[k]; sl = tuple(slice(0, s) for s in b.shape)
            n += int((v[sl] != b).sum()) + int((v != 0).sum() - (v[sl] != 0).sum())
        else:
            n += int((before.params[k] != v).sum())
    return n
