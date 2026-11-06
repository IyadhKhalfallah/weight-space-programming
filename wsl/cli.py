"""wsl — inspect, edit and version a model's weights.

  wsl init   --ws W --model DIR --data CSV --label COL [--slices a,b] [--flag-rate 0.02]
  wsl info   --ws W                      model summary + patch history
  wsl eval   --ws W                      metrics overall and per slice
  wsl effects --ws W                     how much each input moves the output
  wsl weights --ws W FEATURE...          first-layer weights for features
  wsl units  --ws W --col C --value V    hidden units that behave differently on a slice
  wsl ops                                list available edit operations
  wsl try    --ws W OPS.json             dry-run an edit: before/after + regression check
  wsl commit --ws W OPS.json -m MSG      apply an edit as a new patch
  wsl revert --ws W ID                   deactivate a patch (exact undo)
  wsl export --ws W OUT [--format wsl|torch|torchscript]   write the current model (base + active patches);
                   torch -> torch.export .pt2 (torch.export.load(p).module()), torchscript -> torch.jit.load
  wsl import-torch SRC --out DIR (--features a,b,c | --features-csv CSV --label COL)
                   [--group name=f1,f2,...] [--classes a,b,c] [--acts relu,relu] [--trust [--code model.py]]
"""
from __future__ import annotations
import argparse, json, sys
from .workspace import Workspace
from . import inspect_tools as T, patches as P


def _ops(path):
    o = json.load(open(path)) if path != "-" else json.load(sys.stdin)
    return o if isinstance(o, list) else [o]


def main(argv=None):
    ap = argparse.ArgumentParser(prog="wsl"); sub = ap.add_subparsers(dest="cmd", required=True)
    def cmd(name, *args):
        p = sub.add_parser(name)
        if name not in ("ops", "import-torch", "import-sklearn"): p.add_argument("--ws", required=True)
        for a in args: p.add_argument(*a[0], **a[1])
        return p
    cmd("init", (["--model"], dict(required=True)), (["--data"], dict(required=True)), (["--label"], dict(required=True)),
        (["--slices"], dict(default="")), (["--flag-rate"], dict(type=float, default=0.02)))
    cmd("info"); cmd("eval"); cmd("effects"); cmd("ops")
    cmd("weights", (["features"], dict(nargs="+")))
    cmd("units", (["--col"], dict(required=True)), (["--value"], dict(required=True)), (["--layer"], dict(type=int, default=0)))
    cmd("try", (["ops"], {})); cmd("commit", (["ops"], {}), (["-m"], dict(required=True)))
    cmd("revert", (["id"], dict(type=int)))
    cmd("export", (["dir"], {}), (["--format"], dict(default="wsl", choices=["wsl", "torch", "torchscript", "sklearn"])),
        (["--template"], dict(default=None)), (["--trust"], dict(action="store_true")))
    cmd("import-sklearn", (["src"], {}), (["--out"], dict(required=True)), (["--features"], dict(required=True)),
        (["--group"], dict(action="append", default=[])), (["--trust"], dict(action="store_true")))
    cmd("import-torch", (["src"], {}), (["--out"], dict(required=True)), (["--features"], dict(default="")),
        (["--features-csv"], dict(default="")), (["--label"], dict(default="")), (["--group"], dict(action="append", default=[])),
        (["--acts"], dict(default="")), (["--trust"], dict(action="store_true")), (["--code"], dict(default=None)), (["--layout"], dict(default=None)), (["--classes"], dict(default="")), (["--task"], dict(default="classification", choices=["classification", "regression"])))
    a = ap.parse_args(argv)

    if a.cmd == "ops":
        print(P.describe_ops()); return
    if a.cmd == "import-torch":
        from .torch_io import load_torch
        if a.features:
            feats = a.features.split(",")
        elif a.features_csv:
            import csv
            head = next(csv.reader(open(a.features_csv)))
            feats = [h for h in head if h != a.label]
            print(f"note: taking features from the CSV header in order ({len(feats)} columns, excluding label). "
                  f"Make sure this matches the model's input order.")
        else:
            ap.error("give --features or --features-csv")
        groups = {g.split("=")[0]: g.split("=")[1].split(",") for g in a.group}
        m, notes = load_torch(a.src, feats, groups, a.acts.split(",") if a.acts else None, a.trust, a.code, a.layout, a.classes.split(",") if a.classes else None, a.task)
        m.save(a.out); print("\n".join(notes)); print(m.summary()); return
    if a.cmd == "import-sklearn":
        if not a.trust:
            raise ValueError("scikit-learn models are pickles (joblib); loading one can run code. Re-run with --trust "
                             "only if you trust this file.")
        import joblib
        from .sklearn_io import from_sklearn
        groups = {g.split("=")[0]: g.split("=")[1].split(",") for g in a.group}
        m, notes = from_sklearn(joblib.load(a.src), a.features.split(","), groups)
        m.save(a.out); print("\n".join(notes)); print(m.summary()); return
    if a.cmd == "init":
        ws = Workspace.init(a.ws, a.model, a.data, a.label, a.flag_rate, [s for s in a.slices.split(",") if s])
        print(ws.current().summary()); return
    ws = Workspace(a.ws)
    if a.cmd == "info":
        print(ws.current().summary()); print("\npatches:")
        for p in ws.patches or []:
            st = "active" if p["active"] else "reverted"
            k = next(iter(p["before"]))
            print(f"  #{p['id']} [{st}] {p['time']}  {p['message']}  ({p['weights_changed']} weights; "
                  f"{k} {p['before'][k]:.3f}->{p['after'][k]:.3f}; regressions: {len(p['regressions'])})")
        if not ws.patches: print("  (none)")
    elif a.cmd == "eval":
        r = ws.eval()
        if r["kind"] == "regression":
            print(f"n={r['n']}  MAE {r['mae']:.4f}  RMSE {r['rmse']:.4f}  R2 {r['r2']:.4f}  bias {r['bias']:+.4f}")
            for k, v in r["slices"].items():
                print(f"  {k:22s} n={v['n']:6d}  MAE {v['mae']:.4f}  bias {v['bias']:+.4f}")
        elif r["kind"] == "multiclass":
            print(f"n={r['n']}  accuracy {r['accuracy']:.4f}  macro-F1 {r['macro_f1']:.4f}")
            for c, v in r["per_class"].items():
                print(f"  class {c:14s} n={v['n']:6d}  recall {v['recall']:.3f}  precision {v['precision']:.3f}")
            cls = list(r["per_class"]); w = max(len(c) for c in cls) + 2
            print("confusion (rows = true, cols = predicted):\n" + " " * w + "".join(f"{c[:9]:>10s}" for c in cls))
            for c, row in zip(cls, r["confusion"]):
                print(f"{c:{w}s}" + "".join(f"{x:10d}" for x in row))
            for k, v in r["slices"].items():
                rec = "  ".join(f"{c}:{x:.2f}" for c, x in v["recall"].items() if x is not None)
                print(f"  {k:22s} n={v['n']:6d}  accuracy {v['accuracy']:.3f}   recall {rec}")
        else:
            print(f"n={r['n']}  AUC {r['auc']:.4f}  recall {r['recall']:.3f}  false-flag rate {r['false_flag_rate']:.3%}")
            for k, v in r["slices"].items():
                print(f"  {k:22s} n={v['n']:7d}  false flags {v['false_flag_rate']:.3%}  recall {v['recall']}")
    elif a.cmd == "effects":
        print(json.dumps(T.feature_effects(ws.current(), ws.data), indent=1))
    elif a.cmd == "weights":
        print(json.dumps(T.input_weights(ws.current(), a.features), indent=1))
    elif a.cmd == "units":
        val = a.value
        try: val = float(val) if ws.data[a.col].dtype.kind == "f" else val
        except ValueError: pass
        print(json.dumps(T.unit_activity(ws.current(), ws.data, a.col, val, a.layer), indent=1))
    elif a.cmd == "try":
        r = ws.try_ops(_ops(a.ops))
        print(f"weights changed: {r['weights_changed']}"); print("\n".join(r["report"]))
        print("REGRESSIONS:\n  " + "\n  ".join(r["regressions"]) if r["regressions"] else "no regressions")
    elif a.cmd == "commit":
        pid, r = ws.commit(_ops(a.ops), a.m)
        print(f"committed patch #{pid}: {r['weights_changed']} weights changed")
        print("\n".join(r["report"]))
        if r["regressions"]: print("WARNING regressions:\n  " + "\n  ".join(r["regressions"]))
    elif a.cmd == "revert":
        print("reverted" if ws.revert(a.id) else "no such patch")
    elif a.cmd == "export":
        m = ws.current()
        if a.format == "sklearn":
            if not (a.template and a.trust):
                raise ValueError("sklearn export writes into a copy of your original fitted model: pass --template "
                                 "original.joblib --trust (it's a pickle)")
            import joblib
            from .sklearn_io import to_sklearn
            joblib.dump(to_sklearn(m, joblib.load(a.template)), a.dir)
            print(f"exported {m.digest()} to {a.dir} as a scikit-learn estimator (takes raw features: {m.features})")
            return
        if a.format in ("torch", "torchscript"):
            from .torch_io import to_torch
            to_torch(m, a.dir, "torchscript" if a.format == "torchscript" else "export")
            how = "torch.jit.load(p)" if a.format == "torchscript" else "torch.export.load(p).module()"
            print(f"exported {m.digest()} to {a.dir}  (load: {how}; input = features in order: {m.features}; output = fraud logit)")
        else:
            m.save(a.dir); print(f"exported {m.digest()} to {a.dir}")


if __name__ == "__main__":
    try:
        main()
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr); sys.exit(1)
