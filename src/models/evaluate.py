"""Ranking metrics and paired bootstrap confidence intervals."""
from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def top_k_mask(score: np.ndarray, frac: float | None = None, k: int | None = None) -> np.ndarray:
    """Boolean mask of the k highest scores. Ties are broken by row position (stable)."""
    n = len(score)
    if k is None:
        k = int(np.ceil(frac * n))
    k = int(min(max(k, 0), n))
    order = np.argsort(-np.asarray(score, dtype=float), kind="stable")
    mask = np.zeros(n, dtype=bool)
    mask[order[:k]] = True
    return mask


def precision_recall_at(y: np.ndarray, score: np.ndarray, frac: float | None = None,
                        k: int | None = None) -> tuple[float, float]:
    y = np.asarray(y)
    m = top_k_mask(score, frac=frac, k=k)
    tp = float(y[m].sum())
    prec = tp / m.sum() if m.sum() else float("nan")
    rec = tp / y.sum() if y.sum() else float("nan")
    return prec, rec


def ranking_metrics(y: np.ndarray, score: np.ndarray, fracs=(0.01, 0.05)) -> dict:
    y = np.asarray(y)
    out = {"roc_auc": float(roc_auc_score(y, score)),
           "pr_auc": float(average_precision_score(y, score)),
           "n": int(len(y)), "n_pos": int(y.sum())}
    for f in fracs:
        p, r = precision_recall_at(y, score, frac=f)
        tag = f"{f * 100:g}pct"
        out[f"precision_at_{tag}"] = p
        out[f"recall_at_{tag}"] = r
    return out


def bootstrap_indices(n: int, n_boot: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=(n_boot, n), dtype=np.int32)


def paired_bootstrap(stat: Callable[[np.ndarray], float], n: int, n_boot: int, seed: int,
                     level: float = 0.95) -> dict:
    """Percentile bootstrap of ``stat(idx)``; ``stat`` must evaluate every compared
    quantity on the same resampled indices (that is what makes it paired)."""
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        vals[b] = stat(idx)
    vals = vals[np.isfinite(vals)]
    a = (1 - level) / 2
    lo, hi = np.quantile(vals, [a, 1 - a])
    return {"ci_low": float(lo), "ci_high": float(hi), "n_boot_valid": int(len(vals)),
            "excludes_zero": bool(lo > 0 or hi < 0)}


def metric_diff_ci(y, s_a, s_b, metric: str, n_boot: int, seed: int, level: float = 0.95) -> dict:
    """Paired bootstrap CI for metric(model a) - metric(model b) on the same rows."""
    y, s_a, s_b = map(np.asarray, (y, s_a, s_b))
    fn = {"roc_auc": roc_auc_score, "pr_auc": average_precision_score}[metric]

    def stat(idx):
        yy = y[idx]
        if yy.min() == yy.max():
            return np.nan
        return fn(yy, s_a[idx]) - fn(yy, s_b[idx])

    point = fn(y, s_a) - fn(y, s_b)
    return {"point": float(point), **paired_bootstrap(stat, len(y), n_boot, seed, level)}


def mean_diff_ci(x_a: np.ndarray, x_b: np.ndarray, n_boot: int, seed: int, level: float = 0.95,
                 scale: float = 1.0) -> dict:
    """Paired bootstrap CI for scale * (mean(x_a) - mean(x_b)); used for per-row money."""
    d = (np.asarray(x_a, dtype=float) - np.asarray(x_b, dtype=float)) * scale
    rng = np.random.default_rng(seed)
    n = len(d)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        vals[b] = d[rng.integers(0, n, size=n)].mean()
    a = (1 - level) / 2
    lo, hi = np.quantile(vals, [a, 1 - a])
    return {"point": float(d.mean()), "ci_low": float(lo), "ci_high": float(hi),
            "excludes_zero": bool(lo > 0 or hi < 0), "n_boot_valid": n_boot}
