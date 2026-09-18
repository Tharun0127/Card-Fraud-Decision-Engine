"""Swap-set analysis: transactions two policies decide differently.

For two policies A and B on the same rows:
- ``A approves, B declines``
- ``A declines, B approves``
When both policies threshold the same score, one of the two sets is empty by
construction (the decline sets are nested). That is reported, not hidden.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.decision.costs import APPROVE, DECLINE


def swap_sets(dec_a: np.ndarray, dec_b: np.ndarray) -> dict[str, np.ndarray]:
    a, b = np.asarray(dec_a), np.asarray(dec_b)
    if a.shape != b.shape:
        raise ValueError("decision vectors differ in length")
    return {"a_approves_b_declines": (a == APPROVE) & (b == DECLINE),
            "a_declines_b_approves": (a == DECLINE) & (b == APPROVE)}


def characterise(mask: np.ndarray, frame: pd.DataFrame, name: str) -> dict:
    """Summary of the rows in ``mask``. ``frame`` needs amount, isFraud, tenure, ProductCD."""
    sub = frame.loc[mask]
    n = int(mask.sum())
    row = {"set": name, "n": n}
    if n == 0:
        return row
    amt = sub["amount"]
    row.update({
        "fraud_rate": float(sub["isFraud"].mean()),
        "n_fraud": int(sub["isFraud"].sum()),
        "amount_total": float(amt.sum()),
        "amount_p25": float(amt.quantile(0.25)), "amount_median": float(amt.median()),
        "amount_p75": float(amt.quantile(0.75)), "amount_p95": float(amt.quantile(0.95)),
        "tenure_days_median": float(sub["tenure_days"].median()),
        "share_new_card": float((sub["prior_cnt"] == 0).mean()),
    })
    for pcd, share in sub["ProductCD"].astype("string").fillna("nan").value_counts(normalize=True).items():
        row[f"product_{pcd}"] = float(share)
    return row
