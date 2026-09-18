"""Reuse and linkage features: distinct counts of one entity per another.

``distinct_count(key, other)`` for a transaction at time ``t`` is the number of
distinct non-missing ``other`` values seen with the same ``key`` in
transactions strictly before ``t``, plus one if the current transaction's own
``other`` value is new (it is known at authorisation time). Later transactions
never contribute. Rows with a missing ``key`` get NaN.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.velocity import GroupIndex


def distinct_count(df: pd.DataFrame, key: str, other: str, time_col: str = "TransactionDT") -> np.ndarray:
    k_codes, _ = pd.factorize(df[key], use_na_sentinel=True)
    o_codes, _ = pd.factorize(df[other], use_na_sentinel=True)
    t = df[time_col].to_numpy().astype(np.int64)
    valid = (k_codes >= 0) & (o_codes >= 0)
    has_key = k_codes >= 0
    out = np.full(len(df), np.nan, dtype=np.float32)
    out[has_key] = 0.0
    if not valid.any():
        return out
    # first time each (key, other) pair is seen
    pairs = pd.DataFrame({"k": k_codes[valid], "o": o_codes[valid], "t": t[valid]})
    first = pairs.groupby(["k", "o"], sort=False)["t"].min().reset_index()
    idx = GroupIndex(first["k"].to_numpy().astype(np.int64), first["t"].to_numpy())
    kk = k_codes[has_key].astype(np.int64)
    out[has_key] = idx.pos(kk, t[has_key]) - idx.pos(kk, np.zeros(len(kk), dtype=np.int64))
    # +1 when the current pair was not seen strictly before t
    pair_first = pd.Series(first["t"].to_numpy(),
                           index=pd.MultiIndex.from_arrays([first["k"], first["o"]]))
    cur = pd.MultiIndex.from_arrays([k_codes[valid], o_codes[valid]])
    cur_first = pair_first.reindex(cur).to_numpy()
    out[np.flatnonzero(valid)] += (cur_first >= t[valid]).astype(np.float32)
    return out


def linkage_features(df: pd.DataFrame, specs: list[tuple[str, str, str]]) -> pd.DataFrame:
    """``specs`` = [(feature_name, key, other), ...]."""
    return pd.DataFrame({name: distinct_count(df, key, other) for name, key, other in specs},
                        index=df.index)
