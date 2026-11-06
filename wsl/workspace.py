"""Workspace = base model + ordered patch history + an eval dataset.

  <ws>/ws.json        {"label": "is_fraud", "data": "path.csv", "flag_rate": 0.02, "slices": ["currency"]}
  <ws>/base/          the imported model (never modified)
  <ws>/patches.json   [{"id": 1, "message": ..., "ops": [...], "active": true, "before": {...}, "after": {...}}]
"""
from __future__ import annotations
import csv, json, os, shutil, time
import numpy as np
from .model import Model
from . import patches as P


# ------------------------------------------------------------------ data
def load_csv(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    cols = {}
    for k in rows[0]:
        vals = [r[k] for r in rows]
        try:
            cols[k] = np.array([float(v) for v in vals])
        except ValueError:
            cols[k] = np.array(vals)
    return cols


# ------------------------------------------------------------------ metrics
def auc(s, y):
    o = np.argsort(s); r = np.empty(len(s)); r[o] = np.arange(len(s))
    pos = y == 1
    return float((r[pos].mean() - (pos.sum() - 1) / 2) / (len(y) - pos.sum()))


def evaluate(m: Model, data, label, flag_rate, slices=()):
    if m.task == "regression":
        return _eval_reg(m, data, label, slices)
    return _eval_multi(m, data, label, slices) if m.multiclass else _eval_binary(m, data, label, flag_rate, slices)


def compare(before, after, tol=0.01):
    k = before.get("kind")
    if k == "regression":
        return _compare_reg(before, after)
    return _compare_multi(before, after, tol) if k == "multiclass" else _compare_binary(before, after, tol)


def headline(r):
    keys = {"multiclass": ("accuracy", "macro_f1"), "regression": ("mae", "rmse", "r2")}.get(r.get("kind"), ("auc", "recall", "false_flag_rate"))
    return {k: r[k] for k in keys}


def _eval_reg(m: Model, data, label, slices=()):
    y = np.asarray(data[label], float); p = m.forward(m.matrix(data)); e = p - y
    out = {"kind": "regression", "n": int(len(y)), "mae": float(np.abs(e).mean()), "rmse": float(np.sqrt((e ** 2).mean())),
           "r2": float(1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum()), "bias": float(e.mean()), "slices": {}}
    for col in slices:
        for v in np.unique(data[col]):
            mk = data[col] == v
            if mk.sum() >= 50:
                out["slices"][f"{col}={v}"] = {"n": int(mk.sum()), "mae": float(np.abs(e[mk]).mean()), "bias": float(e[mk].mean())}
    return out


def _compare_reg(before, after):
    lines, regs = [], []
    for k in ("mae", "rmse", "r2", "bias"):
        lines.append(f"{k:16s} {before[k]:.4f} -> {after[k]:.4f}  ({after[k] - before[k]:+.4f})")
    if after["mae"] > before["mae"] * 1.02: regs.append(f"overall MAE up {after['mae'] / before['mae'] - 1:.1%}")
    for name, b in before["slices"].items():
        a = after["slices"].get(name)
        if not a: continue
        mark = ""
        if a["mae"] > b["mae"] * 1.05 and a["mae"] - b["mae"] > 0.01 * before["mae"]:
            regs.append(f"{name}: MAE {b['mae']:.3f} -> {a['mae']:.3f}"); mark = "  <-- worse"
        lines.append(f"  {name:22s} MAE {b['mae']:.3f} -> {a['mae']:.3f}   bias {b['bias']:+.3f} -> {a['bias']:+.3f}{mark}")
    return lines, regs


def _labels(m: Model, y):
    if y.dtype.kind == "f":
        y = y.astype(int).astype(str)
    idx = {c: i for i, c in enumerate(m.classes)}
    unknown = sorted(set(y) - set(idx))
    if unknown:
        raise ValueError(f"labels {unknown[:5]} are not among the model's classes {m.classes}")
    return np.array([idx[v] for v in y])


def _eval_multi(m: Model, data, label, slices=()):
    X = m.matrix(data); y = _labels(m, data[label]); p = m.forward(X).argmax(1); K = len(m.classes)
    per = {}
    for k, c in enumerate(m.classes):
        tp = ((p == k) & (y == k)).sum()
        rec = tp / max((y == k).sum(), 1); prec = tp / max((p == k).sum(), 1)
        per[c] = {"n": int((y == k).sum()), "recall": float(rec), "precision": float(prec),
                  "f1": float(2 * rec * prec / (rec + prec)) if rec + prec else 0.0}
    out = {"kind": "multiclass", "n": int(len(y)), "accuracy": float((p == y).mean()),
           "macro_f1": float(np.mean([v["f1"] for v in per.values()])), "per_class": per, "slices": {},
           "confusion": [[int(((y == i) & (p == j)).sum()) for j in range(K)] for i in range(K)]}
    for col in slices:
        for v in np.unique(data[col]):
            mk = data[col] == v
            if mk.sum() < 50:
                continue
            out["slices"][f"{col}={v}"] = {"n": int(mk.sum()), "accuracy": float((p[mk] == y[mk]).mean()),
                "recall": {c: (float((p[mk & (y == k)] == k).mean()) if (mk & (y == k)).sum() >= 30 else None)
                           for k, c in enumerate(m.classes)}}
    return out


def _compare_multi(before, after, tol=0.01):
    lines, regs = [], []
    for k in ("accuracy", "macro_f1"):
        lines.append(f"{k:16s} {before[k]:.4f} -> {after[k]:.4f}  ({after[k] - before[k]:+.4f})")
        if before[k] - after[k] > tol: regs.append(f"overall {k} dropped {before[k] - after[k]:.4f}")
    for c, b in before["per_class"].items():
        a = after["per_class"][c]
        mark = "  <-- recall drop" if b["recall"] - a["recall"] > 0.05 else ""
        if mark: regs.append(f"class {c}: recall {b['recall']:.2f} -> {a['recall']:.2f}")
        lines.append(f"  class {c:14s} recall {b['recall']:.3f} -> {a['recall']:.3f}   precision {b['precision']:.3f} -> {a['precision']:.3f}{mark}")
    for name, b in before["slices"].items():
        a = after["slices"].get(name)
        if not a: continue
        drops = [f"{c} {b['recall'][c]:.2f}->{a['recall'][c]:.2f}" for c in b["recall"]
                 if b["recall"][c] is not None and a["recall"][c] is not None and b["recall"][c] - a["recall"][c] > 0.05]
        mark = ""
        if b["accuracy"] - a["accuracy"] > 0.02: regs.append(f"{name}: accuracy {b['accuracy']:.3f} -> {a['accuracy']:.3f}"); mark = "  <-- accuracy drop"
        if drops: regs.append(f"{name}: recall drop " + ", ".join(drops)); mark += "  <-- " + ", ".join(drops)
        lines.append(f"  {name:22s} accuracy {b['accuracy']:.3f} -> {a['accuracy']:.3f}{mark}")
    return lines, regs


def _eval_binary(m: Model, data, label, flag_rate, slices=()):
    X = m.matrix(data); y = data[label]; s = m.forward(X)
    thr = np.quantile(s, 1 - flag_rate); f = s > thr
    out = {"kind": "binary", "n": int(len(y)), "auc": auc(s, y), "recall": float(f[y == 1].mean()),
           "false_flag_rate": float(f[y == 0].mean()), "slices": {}}
    for col in slices:
        for v in np.unique(data[col]):
            mk = data[col] == v
            if mk.sum() < 50:
                continue
            out["slices"][f"{col}={v}"] = {"n": int(mk.sum()), "recall": float(f[mk & (y == 1)].mean()) if (mk & (y == 1)).any() else None,
                                           "false_flag_rate": float(f[mk & (y == 0)].mean())}
    return out


def _compare_binary(before, after, tol=0.01):
    """regression check: overall + every slice. Returns (lines, regressions)."""
    lines, regs = [], []
    for k in ("auc", "recall", "false_flag_rate"):
        b, a = before[k], after[k]
        lines.append(f"{k:16s} {b:.4f} -> {a:.4f}  ({a - b:+.4f})")
    worse = lambda k, b, a: (a - b > tol) if k == "false_flag_rate" else (b - a > tol)
    if worse("auc", before["auc"], after["auc"]): regs.append(f"overall AUC dropped {before['auc'] - after['auc']:.4f}")
    if worse("recall", before["recall"], after["recall"]): regs.append(f"overall recall dropped {before['recall'] - after['recall']:.3f}")
    for name, b in before["slices"].items():
        a = after["slices"].get(name)
        if not a:
            continue
        fb, fa = b["false_flag_rate"], a["false_flag_rate"]
        rb, ra = b["recall"], a["recall"]
        mark = ""
        if ra is not None and rb is not None and rb - ra > 0.05:
            regs.append(f"{name}: recall {rb:.2f} -> {ra:.2f}"); mark = "  <-- recall drop"
        lines.append(f"  {name:22s} false flags {fb:.3%} -> {fa:.3%}   recall {rb if rb is None else round(rb, 2)} -> {ra if ra is None else round(ra, 2)}{mark}")
    return lines, regs


# ------------------------------------------------------------------ workspace
class Workspace:
    def __init__(self, path):
        self.path = path
        self.cfg = json.load(open(os.path.join(path, "ws.json")))
        self.patches = json.load(open(os.path.join(path, "patches.json")))
        self._data = None

    @staticmethod
    def init(path, model_dir, data_csv, label, flag_rate=0.02, slices=()):
        os.makedirs(path, exist_ok=True)
        shutil.copytree(model_dir, os.path.join(path, "base"), dirs_exist_ok=True)
        json.dump({"label": label, "data": os.path.abspath(data_csv), "flag_rate": flag_rate, "slices": list(slices)},
                  open(os.path.join(path, "ws.json"), "w"), indent=1)
        json.dump([], open(os.path.join(path, "patches.json"), "w"))
        return Workspace(path)

    @property
    def data(self):
        if self._data is None:
            self._data = load_csv(self.cfg["data"])
        return self._data

    def base(self):
        return Model.load(os.path.join(self.path, "base"))

    def current(self):
        m = self.base()
        for p in self.patches:
            if p["active"]:
                m = P.apply(m, p["ops"])
        return m

    def eval(self, m=None):
        return evaluate(m or self.current(), self.data, self.cfg["label"], self.cfg["flag_rate"], self.cfg["slices"])

    def try_ops(self, ops):
        before_m = self.current(); after_m = P.apply(before_m, ops)
        b, a = self.eval(before_m), self.eval(after_m)
        lines, regs = compare(b, a)
        return dict(weights_changed=P.size(before_m, after_m), report=lines, regressions=regs, before=b, after=a)

    def commit(self, ops, message):
        r = self.try_ops(ops)
        pid = max([p["id"] for p in self.patches], default=0) + 1
        self.patches.append({"id": pid, "message": message, "ops": ops, "active": True, "time": time.strftime("%Y-%m-%d %H:%M"),
                             "weights_changed": r["weights_changed"], "regressions": r["regressions"],
                             "before": headline(r["before"]), "after": headline(r["after"])})
        self._save(); return pid, r

    def revert(self, pid):
        for p in self.patches:
            if p["id"] == pid:
                p["active"] = False; self._save(); return True
        return False

    def _save(self):
        json.dump(self.patches, open(os.path.join(self.path, "patches.json"), "w"), indent=1)
