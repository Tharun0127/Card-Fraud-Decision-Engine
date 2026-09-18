"""Smoothed target encoding with out-of-fold values inside the training period.

Encoding for category c: (sum_y(c) + m * prior) / (n(c) + m), where ``prior`` is
the fraud rate of the rows used to fit and ``m`` the smoothing pseudo-count.
Missing values are a category of their own. Categories never seen in the
fitting rows get exactly ``prior``.

- Training rows receive out-of-fold values: the training rows (in time order)
  are cut into ``n_folds`` contiguous blocks and each block is encoded with
  statistics from the other blocks only, so a row never contributes to its own
  encoding.
- Validation and test rows are encoded with statistics from all training rows.
  Their own labels are never read.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

NAN_TOKEN = "__nan__"


def _as_key(s: pd.Series) -> pd.Series:
    if pd.api.types.is_float_dtype(s):
        vals = s.to_numpy(dtype="float64")
        finite = vals[~np.isnan(vals)]
        if np.all(finite == np.floor(finite)):
            return s.astype("Int64").astype("string").fillna(NAN_TOKEN)
    return s.astype("string").fillna(NAN_TOKEN)


def fit_mapping(x: pd.Series, y: pd.Series, smoothing: float) -> tuple[dict, float]:
    if len(x) == 0:
        raise ValueError("cannot fit a target encoding on zero rows")
    y = pd.Series(np.asarray(y, dtype=float))
    if y.isna().any():
        raise ValueError("labels contain missing values")
    prior = float(y.mean())
    g = (pd.DataFrame({"k": _as_key(x).to_numpy(), "y": y.to_numpy()})
         .groupby("k")["y"].agg(["sum", "count"]))
    enc = (g["sum"] + smoothing * prior) / (g["count"] + smoothing)
    return enc.to_dict(), prior


def apply_mapping(x: pd.Series, mapping: dict, prior: float) -> np.ndarray:
    return _as_key(x).map(mapping).astype("float64").fillna(prior).to_numpy()


def oof_encode(x_train: pd.Series, y_train: pd.Series, n_folds: int, smoothing: float) -> np.ndarray:
    """Out-of-fold encoding. ``x_train`` must already be in time order."""
    n = len(x_train)
    if n_folds < 2 or n < n_folds:
        raise ValueError("need at least 2 folds and one row per fold")
    bounds = np.linspace(0, n, n_folds + 1).astype(int)
    out = np.empty(n, dtype=np.float64)
    for i in range(n_folds):
        te = np.arange(bounds[i], bounds[i + 1])
        tr = np.concatenate([np.arange(0, bounds[i]), np.arange(bounds[i + 1], n)])
        mapping, prior = fit_mapping(x_train.iloc[tr], y_train.iloc[tr], smoothing)
        out[te] = apply_mapping(x_train.iloc[te], mapping, prior)
    return out


def target_encode(df: pd.DataFrame, cols: list[str], is_train: np.ndarray, y: pd.Series,
                  n_folds: int, smoothing: float) -> pd.DataFrame:
    """Encode ``cols``. ``df`` must be sorted by time; ``y`` is only read on train rows."""
    is_train = np.asarray(is_train, dtype=bool)
    out = pd.DataFrame(index=df.index)
    y_tr = pd.Series(np.asarray(y)[is_train])
    for c in cols:
        x_tr = df.loc[is_train, c].reset_index(drop=True)
        vals = np.empty(len(df), dtype=np.float64)
        vals[is_train] = oof_encode(x_tr, y_tr, n_folds, smoothing)
        mapping, prior = fit_mapping(x_tr, y_tr, smoothing)
        vals[~is_train] = apply_mapping(df.loc[~is_train, c], mapping, prior)
        out[f"te_{c}"] = vals.astype(np.float32)
    return out
