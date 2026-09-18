"""Cost-function unit tests with hand-computed expected values.

Fixture (margin 0.02, churn $10, review $5, catch rate 0.8):
  r0 fraud  $100  score 0.9
  r1 legit  $50   score 0.8
  r2 fraud  $200  score 0.3
  r3 legit  $1000 score 0.1
Approve-all cost = 100 + 200 = 300.
"""
import numpy as np
import pytest

from src.decision.costs import APPROVE, DECLINE, REVIEW, CostParams, policy_summary, row_benefit, row_costs
from src.decision.policy import pair_net_benefit
from src.decision.threshold import CumStats, decide, single_sweep, three_way_grid

Y = np.array([1, 0, 1, 0])
AMT = np.array([100.0, 50.0, 200.0, 1000.0])
S = np.array([0.9, 0.8, 0.3, 0.1])
P = CostParams(margin_rate=0.02, churn_penalty=10.0, review_cost=5.0, review_catch_rate=0.8)


def test_approve_all_costs_exactly_the_fraud_amount():
    c = row_costs(np.zeros(4, dtype=int), Y, AMT, P)
    np.testing.assert_allclose(c, [100, 0, 200, 0])
    assert row_benefit(np.zeros(4, dtype=int), Y, AMT, P).sum() == 0


def test_false_decline_cost_is_margin_times_amount_plus_churn():
    c = row_costs(np.array([APPROVE, DECLINE, APPROVE, DECLINE]), Y, AMT, P)
    np.testing.assert_allclose(c, [100, 0.02 * 50 + 10, 200, 0.02 * 1000 + 10])  # 11 and 30


def test_review_cost_includes_missed_fraud_at_catch_rate():
    c = row_costs(np.array([REVIEW, REVIEW, REVIEW, REVIEW]), Y, AMT, P)
    np.testing.assert_allclose(c, [5 + 0.2 * 100, 5, 5 + 0.2 * 200, 5])


def test_single_threshold_hand_computed():
    d = decide(S, t_decline=0.5)
    np.testing.assert_array_equal(d, [DECLINE, DECLINE, APPROVE, APPROVE])
    s = policy_summary(d, Y, AMT, P, days=1.0)
    assert s["total_cost"] == pytest.approx(211.0)      # 200 missed + 11 false decline
    assert s["net_benefit"] == pytest.approx(89.0)
    assert s["net_benefit_per_1000"] == pytest.approx(89.0 / 4 * 1000)
    assert s["good_customer_disruption_rate"] == pytest.approx(0.5)
    assert s["fraud_capture_rate_count"] == pytest.approx(0.5)
    assert s["fraud_capture_rate_dollars"] == pytest.approx(100 / 300)


def test_three_way_hand_computed():
    d = decide(S, t_decline=0.85, t_review=0.2)
    np.testing.assert_array_equal(d, [DECLINE, REVIEW, REVIEW, APPROVE])
    s = policy_summary(d, Y, AMT, P, days=2.0)
    assert s["total_cost"] == pytest.approx(0 + 5 + (5 + 40) + 0)
    assert s["net_benefit"] == pytest.approx(250.0)
    assert s["fraud_capture_rate_count"] == pytest.approx((1 + 0.8) / 2)
    assert s["fraud_capture_rate_dollars"] == pytest.approx((100 + 0.8 * 200) / 300)
    assert s["reviews_per_day"] == pytest.approx(1.0)
    assert s["good_customer_disruption_rate"] == 0.0
    assert s["good_customer_review_rate"] == pytest.approx(0.5)


def test_vectorised_sweep_matches_row_level_costs():
    cs = CumStats.build(S, Y, AMT, days=1.0)
    thr = np.array([0.05, 0.2, 0.5, 0.85, 0.95, np.inf])
    sw = single_sweep(cs, thr, P)
    for t, nb in zip(thr, sw["net_benefit"]):
        assert nb == pytest.approx(row_benefit(decide(S, t), Y, AMT, P).sum())
    assert pair_net_benefit(cs, 0.2, 0.85, P) == pytest.approx(250.0)


def test_three_way_grid_respects_capacity():
    cs = CumStats.build(S, Y, AMT, days=1.0)
    thr = np.array([0.05, 0.2, 0.5, 0.85, 0.95, np.inf])
    free = three_way_grid(cs, thr, P, max_reviews_per_day=np.inf)
    capped = three_way_grid(cs, thr, P, max_reviews_per_day=0)
    assert capped["n_reviews"] == 0
    assert free["net_benefit"] >= capped["net_benefit"]
    # with no reviews allowed, the best is declining r0, r1, r2: only r1 is a false decline
    # (cost 11) -> NB 300 - 11 = 289; declining r3 as well would add 30
    assert capped["net_benefit"] == pytest.approx(289.0)
    assert (capped["t_review"], capped["t_decline"]) == (0.2, 0.2)


def test_inputs_are_validated():
    with pytest.raises(ValueError):
        row_costs(np.zeros(4, dtype=int), Y, np.array([1.0, np.nan, 1.0, 1.0]), P)
    with pytest.raises(ValueError):
        row_costs(np.array([0, 3, 0, 0]), Y, AMT, P)
    with pytest.raises(ValueError):
        CostParams(review_catch_rate=1.5)
    with pytest.raises(ValueError):
        decide(S, t_decline=0.2, t_review=0.5)
