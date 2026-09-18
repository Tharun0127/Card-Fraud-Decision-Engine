"""Backward-looking velocity features per card identity.

For a transaction at time ``t`` in group ``g`` a window of length ``w`` covers
earlier transactions of ``g`` with time in ``[t - w, t)``. The interval is open
at ``t``: the transaction itself, and any other transaction with the same
``TransactionDT`` second, are excluded, as is everything later. This is the
leakage guard; ``tests/test_velocity.py`` checks it on a hand-built fixture.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class GroupIndex:
    """Sorted (group, time) index supporting strictly-before-t range queries."""

    def __init__(self, codes: np.ndarray, times: np.ndarray, values: np.ndarray | None = None):
        times = np.asarray(times).astype(np.int64)
        if len(times) and times.min() < 0:
            raise ValueError("times must be non-negative")
        self.S = np.int64(times.max() + 1) if len(times) else np.int64(1)
        self.order = np.lexsort((times, codes))
        self.keys = np.asarray(codes)[self.order].astype(np.int64) * self.S + times[self.order]
        self.times_sorted = times[self.order]
        if values is not None:
            self.csum = np.concatenate([[0.0], np.cumsum(np.asarray(values)[self.order].astype(np.float64))])

    def pos(self, codes: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Number of indexed rows with (code, time) < (code, t): end of the strictly-before range."""
        t = np.minimum(np.asarray(t).astype(np.int64), self.S)  # times beyond the index clamp to group end
        return np.searchsorted(self.keys, np.asarray(codes).astype(np.int64) * self.S + t, side="left")


def velocity_features(df: pd.DataFrame, key: str, windows: dict[str, int],
                      time_col: str = "TransactionDT", amt_col: str = "TransactionAmt",
                      prefix: str | None = None) -> pd.DataFrame:
    """Rolling counts and amount sums per ``key`` over strictly prior windows.

    Also returns seconds since the previous transaction, the prior transaction
    count over all observed history, tenure in days (time since the first prior
    transaction; 0 when there is none) and the amount relative to the mean
    amount over the longest window.
    """
    prefix = prefix or key
    codes, _ = pd.factorize(df[key], use_na_sentinel=True)
    codes = codes.astype(np.int64)
    missing = codes < 0
    codes = np.where(missing, codes.max() + 1, codes)  # own bucket; outputs masked to NaN below
    t = df[time_col].to_numpy().astype(np.int64)
    amt = df[amt_col].to_numpy().astype(np.float64)
    idx = GroupIndex(codes, t, amt)
    hi = idx.pos(codes, t)
    start = idx.pos(codes, np.zeros_like(t))
    out = {}
    for name, w in windows.items():
        lo = idx.pos(codes, np.maximum(t - int(w), 0))
        out[f"{prefix}_cnt_{name}"] = (hi - lo).astype(np.float32)
        out[f"{prefix}_amt_{name}"] = (idx.csum[hi] - idx.csum[lo]).astype(np.float32)
    has_prev = hi > start
    prev_t = np.where(has_prev, idx.times_sorted[np.maximum(hi - 1, 0)], -1)
    first_t = np.where(has_prev, idx.times_sorted[np.minimum(start, len(t) - 1)], -1)
    out[f"{prefix}_secs_since_prev"] = np.where(has_prev, t - prev_t, np.nan).astype(np.float32)
    out[f"{prefix}_prior_cnt"] = (hi - start).astype(np.float32)
    out[f"{prefix}_tenure_days"] = np.where(has_prev, (t - first_t) / 86400.0, 0.0).astype(np.float32)
    longest = max(windows, key=windows.get)
    cnt, tot = out[f"{prefix}_cnt_{longest}"], out[f"{prefix}_amt_{longest}"]
    with np.errstate(divide="ignore", invalid="ignore"):
        out[f"{prefix}_amt_vs_mean_{longest}"] = np.where(cnt > 0, amt / (tot / cnt), np.nan).astype(np.float32)
    res = pd.DataFrame(out, index=df.index)
    if missing.any():
        res.loc[missing, :] = np.nan
    return res
