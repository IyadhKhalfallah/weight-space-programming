"""scikit-learn <-> wsl for weight-based models.

Import: LogisticRegression, LinearRegression/Ridge/Lasso, MLPClassifier, MLPRegressor — optionally inside a
Pipeline whose earlier steps are StandardScaler/MinMaxScaler (folded into the first layer, so the wsl model
takes raw features). Models are pickles (joblib): loading one can run code, so callers must pass trust=True.
Export: writes the edited weights into a copy of the original fitted estimator, so `est.predict(X_raw)` keeps
working in the user's code (any scaler is folded in, so the exported estimator takes raw features).
"""
from __future__ import annotations
import copy
import numpy as np
from .model import Model

ACT = {"relu": "relu", "tanh": "tanh", "logistic": "sigmoid", "identity": "linear"}


def _unwrap(obj):
    """returns (final estimator, scale a, shift c) such that preprocessing is x -> a*x + c"""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    if not isinstance(obj, Pipeline):
        return obj, None, None
    a = c = None
    for name, step in obj.steps[:-1]:
        if isinstance(step, StandardScaler):
            sa = 1 / step.scale_ if step.with_std else np.ones_like(step.mean_)
            sc = -step.mean_ * sa if step.with_mean else np.zeros_like(sa)
        elif isinstance(step, MinMaxScaler):
            sa, sc = step.scale_, step.min_
        else:
            raise ValueError(f"unsupported pipeline step {type(step).__name__}: only StandardScaler/MinMaxScaler before the model")
        a, c = (sa, sc) if a is None else (a * sa, c * sa + sc)
    return obj.steps[-1][1], a, c


def from_sklearn(obj, features, groups=None):
    est, a, c = _unwrap(obj); kind = type(est).__name__; notes = [f"source: sklearn {kind}" + (" (scaler folded in)" if a is not None else "")]
    classes = None; task = "classification"
    if kind in ("LogisticRegression", "LinearRegression", "Ridge", "Lasso", "ElasticNet", "SGDClassifier"):
        W = np.atleast_2d(np.asarray(est.coef_, float)); b = np.atleast_1d(np.asarray(est.intercept_, float))
        layers_W, layers_b, acts = [W], [b], ["linear"]
        if kind in ("LinearRegression", "Ridge", "Lasso", "ElasticNet"):
            task = "regression"
        elif W.shape[0] > 1:
            classes = [str(x) for x in est.classes_]
    elif kind in ("MLPClassifier", "MLPRegressor"):
        layers_W = [np.asarray(w, float).T for w in est.coefs_]; layers_b = [np.asarray(b, float) for b in est.intercepts_]
        acts = [ACT[est.activation]] * (len(layers_W) - 1) + ["linear"]
        if kind == "MLPRegressor":
            task = "regression"
            if layers_W[-1].shape[0] != 1:
                raise ValueError("multi-output regression is not supported yet")
        elif est.out_activation_ == "softmax":
            classes = [str(x) for x in est.classes_]
        elif est.out_activation_ != "logistic":
            raise ValueError(f"unsupported output activation {est.out_activation_}")
    else:
        raise ValueError(f"{kind} is not a weight-based model wsl can import (trees: use the tree tools)")
    if a is not None:                           # fold x' = a*x + c into the first layer
        W0 = layers_W[0]; layers_b[0] = layers_b[0] + W0 @ c; layers_W[0] = W0 * a[None, :]
    if layers_W[0].shape[1] != len(features):
        raise ValueError(f"model takes {layers_W[0].shape[1]} inputs but {len(features)} feature names were given")
    params = {}
    for i, (W, b) in enumerate(zip(layers_W, layers_b)):
        params[f"W{i}"], params[f"b{i}"] = W, b
    m = Model(features, [{"act": x} for x in acts], params, groups, classes=classes, task=task)
    return m, notes


def to_sklearn(m: Model, template):
    """copy of the user's fitted estimator (the final step if `template` is a Pipeline) carrying m's weights"""
    if m.blocks:
        raise ValueError("this model has rule blocks, which scikit-learn estimators can't represent; "
                         "export with --format torch instead")
    est, _, _ = _unwrap(template); est = copy.deepcopy(est); kind = type(est).__name__
    if kind in ("MLPClassifier", "MLPRegressor"):
        est.coefs_ = [m.params[f"W{i}"].T.copy() for i in range(m.n_layers)]
        est.intercepts_ = [m.params[f"b{i}"].copy() for i in range(m.n_layers)]
    else:
        W, b = m.params["W0"], m.params["b0"]; orig = _unwrap(template)[0]
        est.coef_ = W.copy() if np.ndim(orig.coef_) == 2 else W[0].copy()
        est.intercept_ = b.copy() if np.ndim(orig.intercept_) else float(b[0])
    est.n_features_in_ = len(m.features)
    if hasattr(est, "feature_names_in_"):
        del est.feature_names_in_
    return est
