"""Time-ordered train / validation / test split.

``TransactionDT`` is an offset in seconds from an undisclosed reference. It is
used only to order rows and to measure relative durations. Boundaries are set
on row counts and then pushed forward past any tied ``TransactionDT`` so a
timestamp never appears in two splits.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SPLITS = ("train", "valid", "test")


def _boundary(dt_sorted: np.ndarray, idx: int) -> int:
    """First index >= idx whose DT differs from the row before it."""
    if idx <= 0 or idx >= len(dt_sorted):
        return idx
    # move forward while the row at idx ties with the row at idx-1
    return int(np.searchsorted(dt_sorted, dt_sorted[idx - 1], side="right"))


def time_split(df: pd.DataFrame, train_frac: float, valid_frac: float,
               time_col: str = "TransactionDT") -> pd.Series:
    """Return a Series of 'train'/'valid'/'test' aligned to ``df.index``."""
    if not 0 < train_frac < 1 or not 0 < valid_frac < 1 or train_frac + valid_frac >= 1:
        raise ValueError("fractions must be in (0,1) and leave room for a test split")
    if df[time_col].isna().any():
        raise ValueError(f"{time_col} has missing values; cannot order rows")
    order = np.lexsort((df["TransactionID"].to_numpy(), df[time_col].to_numpy()))
    dt_sorted = df[time_col].to_numpy()[order]
    n = len(df)
    b1 = _boundary(dt_sorted, int(round(n * train_frac)))
    b2 = _boundary(dt_sorted, max(int(round(n * (train_frac + valid_frac))), b1))
    labels = np.empty(n, dtype=object)
    labels[order[:b1]] = "train"
    labels[order[b1:b2]] = "valid"
    labels[order[b2:]] = "test"
    out = pd.Series(labels, index=df.index, name="split")
    check_split(df.assign(split=out), time_col)
    return out


def check_split(df: pd.DataFrame, time_col: str = "TransactionDT") -> None:
    """Raise if the split is not strictly time-ordered or a split is empty."""
    g = df.groupby("split")[time_col].agg(["min", "max", "size"])
    for s in SPLITS:
        if s not in g.index or g.loc[s, "size"] == 0:
            raise AssertionError(f"split '{s}' is empty")
    if not g.loc["train", "max"] < g.loc["valid", "min"]:
        raise AssertionError("train overlaps validation in time")
    if not g.loc["valid", "max"] < g.loc["test", "min"]:
        raise AssertionError("validation overlaps test in time")
