"""Policy selection on validation and evaluation on test.

Selection (validation) and evaluation (test) are kept in separate functions so
a threshold chosen with test labels cannot slip into a headline number.
"""
from __future__ import annotations

import numpy as np

from src.decision.costs import CostParams, policy_summary, row_benefit
from src.decision.threshold import (CumStats, decide, profit_optimal_threshold, single_sweep,
                                    three_way_grid, threshold_grid, youden_threshold)


def split_days(dt: np.ndarray) -> float:
    """Length of a split in days of relative time (TransactionDT is seconds)."""
    dt = np.asarray(dt)
    return float((dt.max() - dt.min()) / 86400.0)


def max_reviews_per_day(cost_cfg: dict, train_dt: np.ndarray, share: float | None = None) -> float:
    """Review capacity. ``auto`` = share * mean daily transaction count in training."""
    v = cost_cfg["max_reviews_per_day"]
    share = cost_cfg["review_capacity_share"] if share is None else share
    if v in (None, "auto") or share != cost_cfg["review_capacity_share"]:
        return float(share * len(train_dt) / split_days(train_dt))
    return float(v)


def pair_net_benefit(cs: CumStats, t_review: float, t_decline: float, p: CostParams) -> float:
    """Expected net benefit of the (t_review, t_decline) policy on the rows behind ``cs``."""
    kr, kd = cs.k_at(np.array([t_review, t_decline]))
    F = cs.fraud_amt
    cost = ((cs.total_fraud_amt - F[kr]) + p.margin_rate * cs.legit_amt[kd]
            + p.churn_penalty * cs.legit_cnt[kd] + p.review_cost * (kr - kd)
            + (1 - p.review_catch_rate) * (F[kr] - F[kd]))
    return float(cs.total_fraud_amt - cost)


def select_policies(val_score, val_y, val_amt, val_days: float, p: CostParams, cap_per_day: float,
                    grid_cfg: dict) -> dict:
    """Choose Youden-J, profit-optimal single threshold and constrained three-way pair on validation."""
    cs = CumStats.build(val_score, val_y, val_amt, val_days)
    grid = threshold_grid(val_score, grid_cfg["n_quantiles"], grid_cfg["n_linear"])
    sweep = single_sweep(cs, grid, p)
    t_profit = profit_optimal_threshold(sweep)
    t_youden = youden_threshold(val_score, val_y)
    pgrid = threshold_grid(val_score, grid_cfg["n_policy_grid"] // 2, 0)
    tw = three_way_grid(cs, pgrid, p, cap_per_day)
    return {"t_profit": t_profit, "t_youden": t_youden, "t_review": tw["t_review"],
            "t_decline": tw["t_decline"], "val_sweep": sweep, "val_three_way": tw, "cumstats": cs}


def evaluate_policies(sel: dict, score, y, amt, days: float, p: CostParams,
                      extra: dict[str, np.ndarray] | None = None) -> tuple[dict, dict]:
    """Decision vectors and summaries on an evaluation split for the selected policies."""
    decisions = {
        "approve_all": np.zeros(len(y), dtype=np.int8),
        "youden_single": decide(score, sel["t_youden"]),
        "profit_single": decide(score, sel["t_profit"]),
        "profit_three_way": decide(score, sel["t_decline"], sel["t_review"]),
    }
    decisions.update(extra or {})
    summaries = {k: policy_summary(d, y, amt, p, days) for k, d in decisions.items()}
    return decisions, summaries


def benefits(decisions: dict, y, amt, p: CostParams) -> dict:
    return {k: row_benefit(d, y, amt, p) for k, d in decisions.items()}
