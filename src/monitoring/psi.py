"""Population Stability Index.

PSI = sum_b (a_b - e_b) * ln(a_b / e_b), where e_b and a_b are the shares of the
reference (expected) and comparison (actual) populations in bin b.

- Numeric features: bins are the reference deciles (``n_bins`` quantile edges);
  missing values form their own bin.
- Categorical features: one bin per reference category, plus one bin for
  categories unseen in the reference, plus a missing bin.
- Empty bins are floored at ``eps`` so the log is finite.

Bands (config/pipeline.yaml): < warn = stable, warn..alert = moderate shift,
>= alert = significant shift.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-4


def _psi_from_counts(e: np.ndarray, a: np.ndarray, eps: float = EPS) -> float:
    e = np.asarray(e, dtype=float)
    a = np.asarray(a, dtype=float)
    if e.sum() == 0 or a.sum() == 0:
        return float("nan")
    e = np.maximum(e / e.sum(), eps)
    a = np.maximum(a / a.sum(), eps)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_numeric(ref: np.ndarray, cur: np.ndarray, n_bins: int = 10) -> float:
    ref = np.asarray(ref, dtype=float)
    cur = np.asarray(cur, dtype=float)
    r_ok, c_ok = ~np.isnan(ref), ~np.isnan(cur)
    if r_ok.sum() == 0:
        edges = np.array([])
    else:
        edges = np.unique(np.quantile(ref[r_ok], np.linspace(0, 1, n_bins + 1)[1:-1]))
    rb = np.searchsorted(edges, ref[r_ok], side="right")
    cb = np.searchsorted(edges, cur[c_ok], side="right")
    nb = len(edges) + 1
    e = np.append(np.bincount(rb, minlength=nb), (~r_ok).sum())
    a = np.append(np.bincount(cb, minlength=nb), (~c_ok).sum())
    return _psi_from_counts(e, a)


def psi_categorical(ref: pd.Series, cur: pd.Series) -> float:
    r = pd.Series(ref).astype("string")
    c = pd.Series(cur).astype("string")
    cats = r.dropna().unique().tolist()
    e = [int((r == k).sum()) for k in cats] + [0, int(r.isna().sum())]
    cc = c.value_counts(dropna=True)
    a = [int(cc.get(k, 0)) for k in cats] + [int(cc[~cc.index.isin(cats)].sum()), int(c.isna().sum())]
    return _psi_from_counts(np.array(e), np.array(a))


def band(psi: float, warn: float, alert: float) -> str:
    if np.isnan(psi):
        return "n/a"
    return "significant" if psi >= alert else "moderate" if psi >= warn else "stable"


def feature_psi(ref: pd.DataFrame, cur: pd.DataFrame, features: list[str], categorical: list[str],
                n_bins: int, warn: float, alert: float) -> pd.DataFrame:
    rows = []
    for f in features:
        if f in categorical:
            v = psi_categorical(ref[f], cur[f])
        else:
            v = psi_numeric(ref[f].to_numpy(dtype=float), cur[f].to_numpy(dtype=float), n_bins)
        rows.append({"feature": f, "psi": v, "band": band(v, warn, alert)})
    return pd.DataFrame(rows).sort_values("psi", ascending=False, na_position="last").reset_index(drop=True)
