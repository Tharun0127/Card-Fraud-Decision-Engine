"""Identity keys. Definitions and justification: DECISIONS.md D0.4."""
from __future__ import annotations

import numpy as np
import pandas as pd

CARD_COLS = ["card1", "card2", "card3", "card4", "card5", "card6"]


def _tok(s: pd.Series) -> pd.Series:
    """String token with an explicit marker for missing values."""
    if pd.api.types.is_float_dtype(s):
        vals = s.to_numpy(dtype="float64")
        finite = vals[~np.isnan(vals)]
        if np.all(finite == np.floor(finite)):
            # integer-valued floats: render 150.0 as "150" so keys match across dtypes
            return s.astype("Int64").astype("string").fillna("nan")
    out = s.astype("string")
    return out.fillna("nan")


def add_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``card_key``, ``D1n`` and ``uid`` columns (in place and returned)."""
    card = _tok(df[CARD_COLS[0]])
    for c in CARD_COLS[1:]:
        card = card + "|" + _tok(df[c])
    df["card_key"] = card
    day = np.floor(df["TransactionDT"].to_numpy() / 86400.0)
    df["D1n"] = (day - df["D1"].to_numpy()).astype("float32")
    df["uid"] = card + "|" + _tok(df["addr1"]) + "|" + _tok(df["D1n"])
    return df
