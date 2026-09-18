"""Threshold sweeps on a score.

Convention: a transaction is declined when ``score >= t`` (and, for the
three-way policy, reviewed when ``t_review <= score < t_decline``). A
threshold of +inf means "decline nothing".

All sweeps use cumulative sums over the score-sorted rows, so a grid of
thousands of thresholds (or a 2-D grid of threshold pairs) costs O(n log n)
plus O(grid size).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from src.decision.costs import APPROVE, DECLINE, REVIEW, CostParams


@dataclass
class CumStats:
    """Prefix sums over rows sorted by descending score. Index k = top-k rows."""
    asc_scores: np.ndarray
    n: int
    fraud_amt: np.ndarray   # sum of fraud amounts in top k
    fraud_cnt: np.ndarray
    legit_amt: np.ndarray
    legit_cnt: np.ndarray
    days: float

    @classmethod
    def build(cls, score, y, amount, days: float) -> "CumStats":
        score = np.asarray(score, dtype=float)
        y = np.asarray(y)
        amount = np.asarray(amount, dtype=float)
        if np.isnan(score).any():
            raise ValueError("scores contain NaN")
        order = np.argsort(-score, kind="stable")
        f = (y[order] == 1)
        a = amount[order]

        def cs(v):
            return np.concatenate([[0.0], np.cumsum(v, dtype=np.float64)])

        return cls(np.sort(score), len(score), cs(a * f), cs(f.astype(float)),
                   cs(a * ~f), cs((~f).astype(float)), float(days))

    def k_at(self, t: np.ndarray) -> np.ndarray:
        """Number of rows with score >= t."""
        t = np.asarray(t, dtype=float)
        return self.n - np.searchsorted(self.asc_scores, t, side="left")

    @property
    def total_fraud_amt(self) -> float:
        return float(self.fraud_amt[-1])


def threshold_grid(score, n_quantiles: int, n_linear: int) -> np.ndarray:
    """Candidate thresholds: score quantiles (uniform and upper-tail dense), an even
    grid on [0, 1], and +inf (decline nothing). Sorted ascending, unique."""
    score = np.asarray(score, dtype=float)
    q_uniform = np.quantile(score, np.linspace(0, 1, n_quantiles))
    q_tail = np.quantile(score, 1 - np.geomspace(1e-5, 1, n_quantiles))
    return np.unique(np.concatenate([q_uniform, q_tail, np.linspace(0, 1, n_linear), [np.inf]]))


def single_sweep(cs: CumStats, thresholds: np.ndarray, p: CostParams) -> pd.DataFrame:
    """Decline-if-score>=t policy for every t. No review queue."""
    k = cs.k_at(thresholds)
    missed_fraud = cs.total_fraud_amt - cs.fraud_amt[k]
    fd_cost = p.margin_rate * cs.legit_amt[k] + p.churn_penalty * cs.legit_cnt[k]
    cost = missed_fraud + fd_cost
    nb = cs.total_fraud_amt - cost
    n_legit = cs.legit_cnt[-1]
    n_fraud = cs.fraud_cnt[-1]
    return pd.DataFrame({
        "threshold": thresholds, "n_declined": k, "decline_rate": k / cs.n,
        "total_cost": cost, "net_benefit": nb, "net_benefit_per_1000": nb / cs.n * 1000,
        "tpr": cs.fraud_cnt[k] / n_fraud if n_fraud else np.nan,
        "fpr": cs.legit_cnt[k] / n_legit if n_legit else np.nan,
        "fraud_capture_rate_dollars": cs.fraud_amt[k] / cs.total_fraud_amt if cs.total_fraud_amt else np.nan,
    })


def profit_optimal_threshold(sweep: pd.DataFrame) -> float:
    """Threshold with maximum net benefit. Ties go to the highest threshold
    (fewest declines), so the choice is deterministic."""
    best = sweep["net_benefit"].max()
    cand = sweep.loc[np.isclose(sweep["net_benefit"], best, rtol=0, atol=1e-9), "threshold"]
    return float(cand.max())


def youden_threshold(score, y) -> float:
    """Cutoff maximising TPR - FPR (Youden's J) on the given data."""
    fpr, tpr, thr = roc_curve(np.asarray(y), np.asarray(score, dtype=float))
    j = tpr - fpr
    i = int(np.flatnonzero(j == j.max()).max())  # ties: highest-index = lowest threshold, deterministic
    t = float(thr[i])
    return t if np.isfinite(t) else np.inf


def three_way_grid(cs: CumStats, thresholds: np.ndarray, p: CostParams,
                   max_reviews_per_day: float) -> dict:
    """Search all (t_review <= t_decline) pairs subject to reviews/day <= cap.

    Returns the best pair, its expected net benefit on the data behind ``cs``,
    and the full net-benefit matrix (NaN where infeasible) for plotting.
    """
    t = np.asarray(thresholds, dtype=float)
    k = cs.k_at(t)
    kr = k[:, None]          # rows = t_review
    kd = k[None, :]          # cols = t_decline
    feasible = (t[:, None] <= t[None, :]) & (kr >= kd)
    n_rev = kr - kd
    if max_reviews_per_day is not None and np.isfinite(max_reviews_per_day):
        feasible &= n_rev <= max_reviews_per_day * cs.days
    F = cs.fraud_amt
    cost = ((cs.total_fraud_amt - F[kr])
            + p.margin_rate * cs.legit_amt[kd] + p.churn_penalty * cs.legit_cnt[kd]
            + p.review_cost * n_rev
            + (1 - p.review_catch_rate) * (F[kr] - F[kd]))
    nb = np.where(feasible, cs.total_fraud_amt - cost, np.nan)
    best = np.nanmax(nb)
    ii, jj = np.nonzero(np.isclose(nb, best, rtol=0, atol=1e-9))
    # deterministic tie-break: fewest reviews, then highest decline threshold
    order = np.lexsort((-t[jj], n_rev[ii, jj]))
    i, j = ii[order[0]], jj[order[0]]
    return {"t_review": float(t[i]), "t_decline": float(t[j]), "net_benefit": float(best),
            "net_benefit_per_1000": float(best / cs.n * 1000), "n_reviews": int(n_rev[i, j]),
            "reviews_per_day": float(n_rev[i, j] / cs.days) if cs.days else float("nan"),
            "matrix": nb, "thresholds": t}


def decide(score, t_decline: float, t_review: float | None = None) -> np.ndarray:
    """Apply thresholds to scores: 2 = decline, 1 = review, 0 = approve."""
    s = np.asarray(score, dtype=float)
    d = np.full(len(s), APPROVE, dtype=np.int8)
    if t_review is not None:
        if t_review > t_decline:
            raise ValueError("t_review must not exceed t_decline")
        d[s >= t_review] = REVIEW
    d[s >= t_decline] = DECLINE
    return d
