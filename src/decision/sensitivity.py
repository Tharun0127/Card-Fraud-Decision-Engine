"""Sensitivity of the chosen policy to the (assumed) cost parameters.

For each point of the pre-registered grid, thresholds are re-selected on
validation exactly as for the headline result, then the test net benefit of
each competing policy is computed. The claim this supports is about the
*ranking* of policies, not about absolute dollar amounts.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from src.decision.costs import CostParams, row_benefit
from src.decision.policy import pair_net_benefit
from src.decision.threshold import (CumStats, profit_optimal_threshold, single_sweep,
                                    three_way_grid)

PARAMS = ("margin_rate", "churn_penalty", "review_cost", "review_catch_rate")


def run(grid: dict, base: CostParams, cs_val: CumStats, cs_test: CumStats, single_grid: np.ndarray,
        policy_grid: np.ndarray, cap_per_day: float, fixed_test_decisions: dict[str, np.ndarray],
        test_y: np.ndarray, test_amt: np.ndarray, t_youden: float) -> pd.DataFrame:
    """One row per grid point with selected thresholds and test net benefit per 1,000.

    ``fixed_test_decisions`` are policies whose decisions do not depend on the
    cost parameters (Youden-J, rules baseline); only their valuation changes.
    """
    rows = []
    for combo in itertools.product(*(grid[k] for k in PARAMS)):
        p = base.with_(**dict(zip(PARAMS, map(float, combo))))
        sweep = single_sweep(cs_val, single_grid, p)
        t_single = profit_optimal_threshold(sweep)
        tw = three_way_grid(cs_val, policy_grid, p, cap_per_day)
        n = cs_test.n
        row = dict(zip(PARAMS, combo))
        row.update({
            "t_profit_single": t_single, "t_review": tw["t_review"], "t_decline": tw["t_decline"],
            "val_reviews_per_day": tw["reviews_per_day"],
            "nb1000_profit_three_way": pair_net_benefit(cs_test, tw["t_review"], tw["t_decline"], p) / n * 1000,
            "nb1000_profit_single": pair_net_benefit(cs_test, t_single, t_single, p) / n * 1000,
            "nb1000_youden_single": pair_net_benefit(cs_test, t_youden, t_youden, p) / n * 1000,
        })
        for name, d in fixed_test_decisions.items():
            row[f"nb1000_{name}"] = float(row_benefit(d, test_y, test_amt, p).mean() * 1000)
        rows.append(row)
    return pd.DataFrame(rows)


def ranking_stability(sens: pd.DataFrame, tol: float = 1e-9) -> dict:
    """Share of grid points where each pre-registered ordering (H4) holds."""
    a = sens["nb1000_profit_three_way"] >= sens["nb1000_profit_single"] - tol
    b = sens["nb1000_profit_single"] > sens["nb1000_youden_single"]
    c = sens["nb1000_profit_three_way"] > sens["nb1000_rules"]
    all_ = a & b & c
    return {"n_cells": int(len(sens)),
            "share_three_way_ge_single": float(a.mean()),
            "share_single_gt_youden": float(b.mean()),
            "share_model_gt_rules": float(c.mean()),
            "share_all_orderings": float(all_.mean()),
            "share_three_way_positive": float((sens["nb1000_profit_three_way"] > 0).mean()),
            "share_youden_positive": float((sens["nb1000_youden_single"] > 0).mean())}
