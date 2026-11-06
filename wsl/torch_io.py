"""Import PyTorch models into wsl and export edited models back to TorchScript.

Supported import sources
  - TorchScript file (torch.jit.save)            -> safe to load
  - state_dict (torch.save(model.state_dict()))   -> loaded with weights_only=True (safe); pass --acts
  - full pickled nn.Module (torch.save(model))    -> needs --trust: unpickling can run arbitrary code
Supported layers: Linear, ReLU, Tanh, Sigmoid, BatchNorm1d (folded into a Linear), Dropout / Identity / Flatten
(skipped at inference). A final Sigmoid / Softmax / LogSoftmax is dropped (wsl works in logits); a 2-class
output becomes one logit (class 1 minus class 0).
"""
from __future__ import annotations
import json
import numpy as np
from .model import Model

SKIP = {"Dropout", "Identity", "Flatten", "AlphaDropout"}
ACT = {"ReLU": "relu", "Tanh": "tanh", "Sigmoid": "sigmoid"}
FINAL_DROP = {"Sigmoid", "Softmax", "LogSoftmax"}


def _kind(mod):
    return getattr(mod, "original_name", None) or type(mod).__name__   # TorchScript keeps original_name


def _leaves(mod):
    """flatten a tree of Sequential containers into leaf layers, in execution order"""
    kids = list(mod.named_children())
    if not kids:
        return [mod]
    if _kind(mod) in ("Sequential", "ModuleList"):
        return [l for _, c in kids for l in _leaves(c)]
    if len(kids) == 1:                       # wrapper like self.net = nn.Sequential(...)
        return _leaves(kids[0][1])
    raise ValueError(f"can't infer execution order inside custom module {_kind(mod)!r} with children "
                     f"{[n for n, _ in kids]}; wrap the layers in an nn.Sequential")


def _np(t):
    return t.detach().double().cpu().numpy()


def _bn_affine(bn):
    g = _np(bn.weight) if bn.weight is not None else np.ones(bn.num_features if hasattr(bn, "num_features") else len(_np(bn.running_mean)))
    beta = _np(bn.bias) if bn.bias is not None else np.zeros_like(g)
    a = g / np.sqrt(_np(bn.running_var) + float(bn.eps))
    return a, beta - _np(bn.running_mean) * a          # BN(x) = a*x + c


def from_layers(layers_seq, features, groups=None, classes=None, task="classification"):
    """layers_seq: list of (kind, payload). payload for Linear = (W, b); for BatchNorm1d = (a, c)."""
    lin, acts, pending = [], [], None                  # pending: BN affine waiting to be folded into next Linear
    notes = []
    for i, (kind, p) in enumerate(layers_seq):
        if kind == "Linear":
            W, b = p
            if pending is not None:                    # BN came after an activation -> fold into this Linear's input
                a, c = pending; b = b + W @ c; W = W * a[None, :]; pending = None
            lin.append([W, b]); acts.append("linear")
        elif kind == "BatchNorm1d":
            a, c = p
            if lin and acts[-1] == "linear":           # directly after a Linear -> fold into its output
                W, b = lin[-1]; lin[-1] = [W * a[:, None], b * a + c]
            else:
                pending = (a, c) if pending is None else (pending[0] * a, pending[1] * a + c)
        elif kind in ACT:
            last = i == len(layers_seq) - 1 or all(k in SKIP | FINAL_DROP for k, _ in layers_seq[i + 1:])
            if last and kind in FINAL_DROP:
                notes.append(f"dropped final {kind}: outputs are logits"); continue
            if not lin or acts[-1] != "linear":
                raise ValueError(f"activation {kind} must follow a Linear layer")
            acts[-1] = ACT[kind]
        elif kind in FINAL_DROP:
            notes.append(f"dropped final {kind}: outputs are logits")
        elif kind in SKIP:
            continue
        else:
            raise ValueError(f"unsupported layer {kind!r}. Supported: Linear, ReLU, Tanh, Sigmoid, BatchNorm1d, "
                             f"Dropout, Identity, Flatten (+ final Sigmoid/Softmax)")
    if pending is not None:
        raise ValueError("BatchNorm at the very end is not supported")
    if acts[-1] != "linear":
        raise ValueError("the last Linear must be the output layer (no activation after it except a final Sigmoid/Softmax)")
    W, b = lin[-1]; K = W.shape[0]
    if task == "regression":
        if K != 1:
            raise ValueError(f"regression needs 1 output, model has {K}")
        notes.append("regression: output is the predicted value")
    elif classes:
        if len(classes) != K:
            raise ValueError(f"--classes has {len(classes)} names but the model has {K} outputs")
        notes.append(f"{K} classes: {classes}")
    elif K == 2:
        lin[-1] = [(W[1] - W[0])[None, :], np.array([b[1] - b[0]])]
        notes.append("2-class output -> single logit (class 1); pass --classes a,b to keep both outputs")
    elif K > 2:
        classes = [str(i) for i in range(K)]; notes.append(f"{K} classes, unnamed ({classes}); pass --classes to name them")
    if lin[0][0].shape[1] != len(features):
        raise ValueError(f"model takes {lin[0][0].shape[1]} inputs but {len(features)} feature names were given")
    params = {}
    for k, (W, b) in enumerate(lin):
        params[f"W{k}"], params[f"b{k}"] = W, b
    return Model(features, [{"act": a} for a in acts], params, groups, classes=classes, task=task), notes


def _register_code(code_path):
    """make the user's model classes importable for unpickling (pickles reference e.g. __main__.FraudNet)"""
    import importlib.util, sys
    spec = importlib.util.spec_from_file_location("wsl_user_model", code_path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    main = sys.modules["__main__"]
    for k, v in vars(mod).items():
        if isinstance(v, type):
            setattr(main, k, v)


def load_torch(path, features, groups=None, acts=None, trust=False, code=None, layout=None, classes=None, task="classification"):
    import torch, warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    if code:
        if not trust:
            raise ValueError("--code runs your Python file; pass --trust as well to confirm")
        _register_code(code)
    try:
        mod = torch.jit.load(path, map_location="cpu"); src = "torchscript"
    except Exception:
        obj = None
        try:
            obj = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as e:
            if not trust:
                raise ValueError(f"{path} is not TorchScript or a plain state_dict (it is probably a pickled nn.Module). "
                                 f"Loading pickles can execute code: re-run with --trust only if you trust this file. ({e.__class__.__name__})")
            try:
                obj = torch.load(path, map_location="cpu", weights_only=False)
            except AttributeError as e:
                raise ValueError(f"this pickle needs your model's class definitions ({e}). "
                                 f"Pass the file that defines them with --code path/to/model.py") from None
        if isinstance(obj, dict):
            return _from_state_dict(obj, features, groups, acts, layout, classes, task)
        mod = obj; src = "pickled module"
    mod.eval()
    seq = []
    for l in _leaves(mod):
        k = _kind(l)
        if k == "Linear":
            seq.append((k, (_np(l.weight), _np(l.bias) if l.bias is not None else np.zeros(l.weight.shape[0]))))
        elif k == "BatchNorm1d":
            seq.append((k, _bn_affine(l)))
        else:
            seq.append((k, None))
    m, notes = from_layers(seq, features, groups, classes, task)
    return m, [f"source: {src}"] + notes


LAYOUT_TOKENS = {"linear": "Linear", "bn": "BatchNorm1d", "relu": "ReLU", "tanh": "Tanh", "sigmoid": "Sigmoid",
                 "dropout": "Dropout", "softmax": "Softmax"}


def _from_state_dict(sd, features, groups, acts, layout=None, classes=None, task="classification"):
    seq, keys = [], list(sd.keys())
    prefixes = []
    for k in keys:
        p = k.rsplit(".", 1)[0] if "." in k else ""
        if p not in prefixes: prefixes.append(p)
    for p in prefixes:
        g = {k.rsplit(".", 1)[-1]: sd[k] for k in keys if (k.rsplit(".", 1)[0] if "." in k else "") == p}
        if "running_mean" in g:
            class _BN: pass
            bn = _BN(); bn.weight = g.get("weight"); bn.bias = g.get("bias"); bn.running_mean = g["running_mean"]
            bn.running_var = g["running_var"]; bn.eps = 1e-5
            seq.append(("BatchNorm1d", _bn_affine(bn)))
        elif "weight" in g and g["weight"].dim() == 2:
            W = _np(g["weight"]); seq.append(("Linear", (W, _np(g["bias"]) if "bias" in g else np.zeros(len(W)))))
    if layout:                                   # explicit order: map parameter groups onto linear/bn tokens
        toks = [LAYOUT_TOKENS[t.strip().lower()] for t in layout.split(",")]
        it = iter(seq); out = []
        for tk in toks:
            if tk in ("Linear", "BatchNorm1d"):
                k, p = next(it)
                if k != tk:
                    raise ValueError(f"--layout says {tk} but the next tensors in the state_dict are a {k}")
                out.append((k, p))
            else:
                out.append((tk, None))
        if next(it, None) is not None:
            raise ValueError("--layout has fewer linear/bn layers than the state_dict")
        m, notes = from_layers(out, features, groups, classes, task)
        return m, ["source: state_dict with explicit --layout"] + notes
    if any(k == "BatchNorm1d" for k, _ in seq):
        raise ValueError("this state_dict has BatchNorm layers, and a state_dict does not record whether each BatchNorm "
                         "comes before or after its activation. Guessing can silently produce a wrong model. Either save "
                         "TorchScript (torch.jit.save(torch.jit.script(model), path)) or pass the layer order, e.g. "
                         "--layout linear,bn,relu,dropout,linear,relu,bn,linear")
    n_hidden = sum(k == "Linear" for k, _ in seq) - 1
    acts = acts or ["relu"] * n_hidden
    if len(acts) != n_hidden:
        raise ValueError(f"state_dict has {n_hidden} hidden layers; pass {n_hidden} activations with --acts")
    out, li = [], 0
    for k, p in seq:
        out.append((k, p))
        if k == "Linear" and li < n_hidden:
            out.append(({"relu": "ReLU", "tanh": "Tanh", "sigmoid": "Sigmoid"}[acts[li]], None)); li += 1
    m, notes = from_layers(out, features, groups, classes, task)
    return m, [f"source: state_dict (activations: {acts})"] + notes


# ------------------------------------------------------------------ export
def to_torch(m: Model, path, fmt="export"):
    """fmt='export' -> torch.export .pt2 (load: torch.export.load(p).module());
       fmt='torchscript' -> .pt (load: torch.jit.load(p); deprecated in recent PyTorch)"""
    import torch, warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    net = _build(m)
    meta = json.dumps({"features": m.features, "groups": m.groups, "classes": m.classes, "task": m.task, "output": "value" if m.task == "regression" else "logits" if m.multiclass else "logit", "digest": m.digest(), "blocks": m.blocks})
    if fmt == "torchscript":
        torch.jit.save(torch.jit.script(net), path, _extra_files={"wsl.json": meta})
    else:
        ex = torch.export.export(net, (torch.zeros(2, len(m.features)),),
                                 dynamic_shapes={"x": {0: torch.export.Dim("batch")}})
        torch.export.save(ex, path, extra_files={"wsl.json": meta})
    return path


def _build(m: Model):
    import torch, torch.nn as nn
    acts = {"relu": nn.ReLU, "tanh": nn.Tanh, "sigmoid": nn.Sigmoid, "linear": nn.Identity}
    t = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32)

    class Block(nn.Module):
        def __init__(self, V, c, A, e, U):
            super().__init__()
            self.V = nn.Linear(V.shape[1], V.shape[0]); self.V.weight.data, self.V.bias.data = t(V), t(c)
            self.A = nn.Linear(A.shape[1], 1); self.A.weight.data, self.A.bias.data = t(A), t(e)
            self.U = nn.Linear(1, U.shape[0], bias=False); self.U.weight.data = t(U)
        def forward(self, h, x):
            g = torch.tanh(self.V(torch.cat([h, x], dim=1)))
            return h + self.U((torch.tanh(self.A(g)) + 1) / 2)

    class WSLNet(nn.Module):
        binary: bool

        def __init__(self):
            super().__init__()
            self.binary = not m.multiclass  # binary logit or regression value: one output
            body = []
            for i, spec in enumerate(m.layers[:-1]):
                W, b = m.params[f"W{i}"], m.params[f"b{i}"]
                lin = nn.Linear(W.shape[1], W.shape[0]); lin.weight.data, lin.bias.data = t(W), t(b)
                body += [lin, acts[spec["act"]]()]
            self.body = nn.Sequential(*body)
            self.blocks = nn.ModuleList([Block(*(m.params[f"{p}{k}"] for p in "VcAeU")) for k in range(len(m.blocks))])
            L = m.n_layers - 1; W, b = m.params[f"W{L}"], m.params[f"b{L}"]
            self.head = nn.Linear(W.shape[1], W.shape[0]); self.head.weight.data, self.head.bias.data = t(W), t(b)
        def forward(self, x):
            h = self.body(x)
            for blk in self.blocks:
                h = blk(h, x)
            out = self.head(h)
            return out.squeeze(1) if self.binary else out

    return WSLNet().eval()
