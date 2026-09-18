"""LightGBM training, expanding-window time-series CV tuning, and JSON archiving.

Tuning uses training rows only. The training period (already in time order) is
cut into ``n_folds + 1`` contiguous blocks; fold k trains on blocks 0..k and
early-stops / scores on block k+1. The validation and test splits are never
seen by this module's tuning code.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def base_params(cfg: dict) -> dict:
    seed = cfg["seed"]
    return {
        "objective": "binary",
        "metric": "average_precision",
        "verbosity": -1,
        "seed": seed,
        "bagging_seed": seed,
        "feature_fraction_seed": seed,
        "data_random_seed": seed,
        "deterministic": True,
        "force_col_wise": True,
        "num_threads": cfg["n_threads"],
        **cfg["model"]["fixed_params"],
    }


def expanding_folds(n: int, n_folds: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding-window folds over rows 0..n-1 (assumed time-ordered)."""
    if n_folds < 1 or n < n_folds + 1:
        raise ValueError("not enough rows for the requested folds")
    b = np.linspace(0, n, n_folds + 2).astype(int)
    return [(np.arange(0, b[k + 1]), np.arange(b[k + 1], b[k + 2])) for k in range(n_folds)]


def _dataset(X: pd.DataFrame, y, categorical: list[str], reference=None) -> lgb.Dataset:
    cats = [c for c in categorical if c in X.columns]
    return lgb.Dataset(X, label=np.asarray(y), categorical_feature=cats or "auto",
                       free_raw_data=False, reference=reference)


def fit(X: pd.DataFrame, y, categorical: list[str], params: dict, num_rounds: int,
        X_val: pd.DataFrame | None = None, y_val=None, early_stopping: int | None = None) -> lgb.Booster:
    dtrain = _dataset(X, y, categorical)
    valid_sets, callbacks = [], []
    if X_val is not None:
        valid_sets = [_dataset(X_val, y_val, categorical, reference=dtrain)]
        if early_stopping:
            callbacks.append(lgb.early_stopping(early_stopping, verbose=False))
    return lgb.train(params, dtrain, num_boost_round=num_rounds, valid_sets=valid_sets,
                     callbacks=callbacks)


def tune(X: pd.DataFrame, y, categorical: list[str], cfg: dict, grid: dict | None = None,
         log=print) -> dict:
    """Grid search with expanding-window CV. Returns the chosen params and the CV table."""
    mcfg = cfg["model"]
    grid = grid if grid is not None else mcfg["tuning_grid"]
    y = np.asarray(y)
    folds = expanding_folds(len(X), mcfg["cv_folds"])
    rows = []
    keys = sorted(grid)
    for combo in itertools.product(*(grid[k] for k in keys)):
        p = {**base_params(cfg), **dict(zip(keys, combo))}
        for k, (tr, va) in enumerate(folds):
            bst = fit(X.iloc[tr], y[tr], categorical, p, mcfg["max_rounds"], X.iloc[va], y[va],
                      mcfg["early_stopping_rounds"])
            pred = bst.predict(X.iloc[va], num_iteration=bst.best_iteration)
            ap = average_precision_score(y[va], pred)
            rows.append({**dict(zip(keys, combo)), "fold": k, "n_train": len(tr), "n_valid": len(va),
                         "best_iteration": bst.best_iteration, "pr_auc": ap})
            log(f"[tune] {dict(zip(keys, combo))} fold {k}: PR-AUC {ap:.4f} @ {bst.best_iteration}")
    cv = pd.DataFrame(rows)
    summary = cv.groupby(keys).agg(pr_auc_mean=("pr_auc", "mean"), pr_auc_std=("pr_auc", "std"),
                                   best_iteration_mean=("best_iteration", "mean")).reset_index()
    best = summary.sort_values(["pr_auc_mean"] + keys, ascending=[False] + [True] * len(keys)).iloc[0]
    # cast back to the grid's own types (a mixed-dtype summary row turns ints into floats)
    chosen = {k: type(grid[k][0])(best[k]) for k in keys}
    return {"params": chosen, "num_rounds": int(round(best["best_iteration_mean"])),
            "cv_table": cv, "cv_summary": summary}


def cv_rounds(X: pd.DataFrame, y, categorical: list[str], cfg: dict, params: dict) -> tuple[int, pd.DataFrame]:
    """Number of rounds for fixed params: mean early-stopping iteration over the CV folds."""
    res = tune(X, y, categorical, cfg, grid={k: [v] for k, v in params.items()})
    return res["num_rounds"], res["cv_table"]


def save_json(bst: lgb.Booster, path: Path, extra: dict) -> None:
    """Archive the booster as JSON: LightGBM's native text model plus metadata. No pickle."""
    doc = {"format": "lightgbm-native-model-string", "lightgbm_version": lgb.__version__,
           "model_str": bst.model_to_string(), "feature_names": bst.feature_name(), **extra}
    Path(path).write_text(json.dumps(doc), encoding="utf-8")


def load_json(path: Path) -> tuple[lgb.Booster, dict]:
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if doc.get("format") != "lightgbm-native-model-string":
        raise ValueError("not a model archive written by save_json")
    bst = lgb.Booster(model_str=doc.pop("model_str"))
    return bst, doc
