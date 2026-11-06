"""A small, explicit model format: dense layers + optional residual "rule" blocks.

On disk (a directory):
  model.json   {"features": [...], "groups": {"currency": ["cur_USD", ...]},
                "layers": [{"act": "tanh"}, ..., {"act": "linear"}], "blocks": [...]}
  weights.npz  W0, b0, W1, b1, ...   (Wi has shape [out, in])
                + for each block k: V{k} [n, hidden+inputs], c{k} [n], A{k} [1, n], e{k} [1], U{k} [hidden, 1]

Blocks sit on the last hidden layer (before the output layer) and also see the raw inputs:
  g = tanh(V [h; x] + c)            one saturating gate per condition (~+1 true, ~-1 false)
  a = (tanh(A g + e) + 1) / 2       AND of the gates (~1 when all true, ~0 otherwise)
  h <- h + U a
With U = 0 a block does nothing, so adding one never changes behaviour until weights are written into it.
"""
from __future__ import annotations
import copy, hashlib, json, os
import numpy as np

ACTS = {"tanh": np.tanh, "relu": lambda z: np.maximum(z, 0), "linear": lambda z: z,
        "sigmoid": lambda z: 1 / (1 + np.exp(-z))}


class Model:
    def __init__(self, features, layers, params, groups=None, blocks=None, classes=None, task="classification"):
        self.task = task                          # "classification" | "regression"
        self.features = list(features)
        self.layers = layers                      # [{"act": ...}]
        self.params = params                      # {"W0": arr, "b0": arr, ...}
        self.groups = groups or {}
        self.blocks = blocks or []                # [{"name": ..., "units": n}]
        self.classes = list(classes) if classes else None   # None = binary (one logit)
        k = self.params[f"W{len(layers) - 1}"].shape[0]
        if task == "regression":
            if k != 1 or classes:
                raise ValueError("regression models must have exactly one output")
        elif self.classes is None and k > 1:
            self.classes = [str(i) for i in range(k)]
        if self.classes is not None and len(self.classes) != k:
            raise ValueError(f"{len(self.classes)} class names for {k} outputs")

    @property
    def multiclass(self):
        return self.classes is not None

    # ---------------------------------------------------------------- io
    @classmethod
    def load(cls, path):
        spec = json.load(open(os.path.join(path, "model.json")))
        w = np.load(os.path.join(path, "weights.npz"))
        return cls(spec["features"], spec["layers"], {k: w[k].astype(np.float64) for k in w.files},
                   spec.get("groups"), spec.get("blocks"), spec.get("classes"), spec.get("task", "classification"))

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        json.dump({"features": self.features, "groups": self.groups, "layers": self.layers, "blocks": self.blocks,
                   "classes": self.classes, "task": self.task}, open(os.path.join(path, "model.json"), "w"), indent=1)
        np.savez(os.path.join(path, "weights.npz"), **self.params)

    def copy(self):
        return Model(self.features, copy.deepcopy(self.layers), {k: v.copy() for k, v in self.params.items()},
                     copy.deepcopy(self.groups), copy.deepcopy(self.blocks), self.classes, self.task)

    def digest(self):
        h = hashlib.sha256()
        for k in sorted(self.params):
            h.update(k.encode()); h.update(np.ascontiguousarray(self.params[k]).tobytes())
        h.update(json.dumps(self.features).encode())
        return h.hexdigest()[:12]

    # ---------------------------------------------------------------- compute
    @property
    def n_layers(self):
        return len(self.layers)

    def forward(self, X, return_hidden=False):
        h = X; hidden = []
        for i, spec in enumerate(self.layers):
            if i == self.n_layers - 1:            # blocks act on the last hidden representation
                for k, _ in enumerate(self.blocks):
                    g = np.tanh(np.c_[h, X] @ self.params[f"V{k}"].T + self.params[f"c{k}"])
                    a = (np.tanh(g @ self.params[f"A{k}"].T + self.params[f"e{k}"]) + 1) / 2
                    h = h + a @ self.params[f"U{k}"].T
            h = ACTS[spec["act"]](h @ self.params[f"W{i}"].T + self.params[f"b{i}"])
            hidden.append(h)
        out = h if self.multiclass else h[:, 0]          # multiclass: [N, K] logits; binary: [N] logit
        return (out, hidden) if return_hidden else out

    def log_probs(self, X):
        z = self.forward(X); z = z - z.max(1, keepdims=True)
        return z - np.log(np.exp(z).sum(1, keepdims=True))

    def matrix(self, X_frame):
        """pull this model's feature columns out of a {name: column} mapping"""
        return np.c_[[np.asarray(X_frame[f], dtype=float) for f in self.features]].T

    def summary(self):
        rows = [f"inputs: {len(self.features)} features" + (f"  groups: {self.groups}" if self.groups else ""),
                f"outputs: {'one number (regression)' if self.task == 'regression' else 'classes ' + str(self.classes) if self.multiclass else 'one logit (binary)'}"]
        for i, spec in enumerate(self.layers):
            W = self.params[f"W{i}"]
            rows.append(f"layer {i}: {W.shape[1]} -> {W.shape[0]}  act={spec['act']}  |W|={np.linalg.norm(W):.2f}")
        for k, b in enumerate(self.blocks):
            rows.append(f"block {k} ({b.get('name', '')}): {b['units']} units, residual on last hidden layer")
        n = sum(v.size for v in self.params.values())
        rows.append(f"total weights: {n}   digest {self.digest()}")
        return "\n".join(rows)
