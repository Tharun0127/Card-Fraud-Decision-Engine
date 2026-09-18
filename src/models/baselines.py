"""Pre-registered rules baseline (PRE_REGISTRATION.md §4).

Decline when the amount exceeds a fixed percentile of training-period amounts
AND the card (``uid``, DECISIONS.md D0.4) has no earlier transaction in the
observed history. Not tuned.
"""
from __future__ import annotations

import numpy as np


def amount_threshold(train_amounts: np.ndarray, percentile: float) -> float:
    a = np.asarray(train_amounts, dtype=float)
    if a.size == 0 or np.isnan(a).all():
        raise ValueError("no training amounts to take a percentile from")
    return float(np.nanpercentile(a, percentile))


def rules_decline(amount: np.ndarray, uid_prior_cnt: np.ndarray, amt_threshold: float) -> np.ndarray:
    """Boolean decline flag. A missing prior count is treated as 'not new' (conservative)."""
    amount = np.asarray(amount, dtype=float)
    prior = np.asarray(uid_prior_cnt, dtype=float)
    return (amount > amt_threshold) & (prior == 0)
