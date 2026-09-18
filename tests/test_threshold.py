"""Threshold selection: Youden vs profit, tie handling, and sensitivity ranking."""
import numpy as np
import pandas as pd
import pytest

from src.decision.costs import CostParams, row_benefit
from src.decision.sensitivity import ranking_stability
from src.decision.swap import characterise, swap_sets
from src.decision.threshold import (CumStats, decide, profit_optimal_threshold, single_sweep,
                                    threshold_grid, youden_threshold)


def _synthetic(n=20_000, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.04).astype(int)
    score = np.clip(rng.normal(0.2 + 0.45 * y, 0.15), 0, 1)
    amt = rng.lognormal(4, 1, n)
    return score, y, amt


def test_profit_optimum_beats_youden_in_sample():
    s, y, a = _synthetic()
    p = CostParams()
    cs = CumStats.build(s, y, a, days=10)
    sw = single_sweep(cs, threshold_grid(s, 500, 101), p)
    t_p, t_y = profit_optimal_threshold(sw), youden_threshold(s, y)
    nb_p = row_benefit(decide(s, t_p), y, a, p).sum()
    nb_y = row_benefit(decide(s, t_y), y, a, p).sum()
    assert nb_p >= nb_y


def test_youden_threshold_matches_brute_force():
    s, y, _ = _synthetic(3_000, seed=4)
    t = youden_threshold(s, y)
    grid = np.unique(s)
    j = [((s >= g) & (y == 1)).sum() / y.sum() - ((s >= g) & (y == 0)).sum() / (y == 0).sum() for g in grid]
    j_t = ((s >= t) & (y == 1)).sum() / y.sum() - ((s >= t) & (y == 0)).sum() / (y == 0).sum()
    assert j_t == pytest.approx(max(j))


def test_grid_contains_decline_nothing_option():
    s, _, _ = _synthetic(1_000)
    g = threshold_grid(s, 50, 11)
    assert np.isinf(g[-1]) and np.all(np.diff(g) > 0)


def test_when_false_declines_are_ruinous_the_optimum_declines_no_legit_customer():
    s, y, a = _synthetic(5_000)
    p = CostParams(churn_penalty=1e6)
    cs = CumStats.build(s, y, a, days=1)
    t = profit_optimal_threshold(single_sweep(cs, threshold_grid(s, 200, 11), p))
    declined = decide(s, t) == 2
    assert (declined & (y == 0)).sum() == 0  # it may still decline a pure-fraud top tail


def test_tied_scores_are_all_on_the_same_side_of_a_threshold():
    s = np.array([0.5, 0.5, 0.5, 0.1])
    y = np.array([1, 0, 1, 0])
    cs = CumStats.build(s, y, np.ones(4), days=1)
    assert cs.k_at(np.array([0.5]))[0] == 3


def test_nan_scores_are_rejected():
    with pytest.raises(ValueError):
        CumStats.build(np.array([0.1, np.nan]), np.array([0, 1]), np.ones(2), days=1)


def test_swap_sets_are_disjoint_and_nested_for_a_shared_score():
    s, _, _ = _synthetic(2_000)
    low, high = decide(s, 0.3), decide(s, 0.6)
    sw = swap_sets(high, low)
    assert sw["a_approves_b_declines"].sum() == ((s >= 0.3) & (s < 0.6)).sum()
    assert sw["a_declines_b_approves"].sum() == 0  # nested by construction
    frame = pd.DataFrame({"amount": np.ones(2_000), "isFraud": 0, "tenure_days": 0.0, "prior_cnt": 0,
                          "ProductCD": "W"})
    assert characterise(sw["a_declines_b_approves"], frame, "x") == {"set": "x", "n": 0}


def test_ranking_stability_counts_cells():
    sens = pd.DataFrame({"nb1000_profit_three_way": [10, 5, 1], "nb1000_profit_single": [8, 6, 0],
                         "nb1000_youden_single": [1, 1, 2], "nb1000_rules": [0, 0, 0]})
    r = ranking_stability(sens)
    assert r["share_three_way_ge_single"] == pytest.approx(2 / 3)
    assert r["share_single_gt_youden"] == pytest.approx(2 / 3)
    assert r["share_all_orderings"] == pytest.approx(1 / 3)
