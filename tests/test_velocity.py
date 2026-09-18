"""Leakage guards for velocity features on a hand-built fixture.

Card A transactions (t in seconds, amount):
  r0 t=0      $10
  r1 t=100    $20
  r2 t=100    $40   (same second as r1)
  r3 t=3700   $80
  r4 t=90000  $160
Card B: r5 t=50  $1000 (must never leak into card A)
"""
import numpy as np
import pandas as pd
import pytest

from src.features.velocity import velocity_features

W = {"1h": 3600, "24h": 86400, "7d": 604800}


@pytest.fixture
def fx():
    return pd.DataFrame({
        "TransactionDT": [0, 100, 100, 3700, 90000, 50],
        "TransactionAmt": [10.0, 20.0, 40.0, 80.0, 160.0, 1000.0],
        "card": ["A", "A", "A", "A", "A", "B"],
    })


def test_first_transaction_sees_nothing(fx):
    v = velocity_features(fx, "card", W, prefix="v")
    assert v.loc[0, "v_cnt_7d"] == 0 and v.loc[0, "v_amt_7d"] == 0
    assert np.isnan(v.loc[0, "v_secs_since_prev"])


def test_transaction_never_sees_itself_or_same_second(fx):
    v = velocity_features(fx, "card", W, prefix="v")
    # r1 and r2 share t=100: each sees only r0
    for r in (1, 2):
        assert v.loc[r, "v_cnt_1h"] == 1
        assert v.loc[r, "v_amt_1h"] == pytest.approx(10.0)


def test_hand_computed_windows(fx):
    v = velocity_features(fx, "card", W, prefix="v")
    # r3 at 3700: 1h window [100, 3700) -> r1, r2 ; 24h -> r0, r1, r2
    assert v.loc[3, "v_cnt_1h"] == 2 and v.loc[3, "v_amt_1h"] == pytest.approx(60.0)
    assert v.loc[3, "v_cnt_24h"] == 3 and v.loc[3, "v_amt_24h"] == pytest.approx(70.0)
    # r4 at 90000: 24h window [3600, 90000) -> r3 only; 7d -> all four earlier
    assert v.loc[4, "v_cnt_24h"] == 1 and v.loc[4, "v_amt_24h"] == pytest.approx(80.0)
    assert v.loc[4, "v_cnt_7d"] == 4 and v.loc[4, "v_amt_7d"] == pytest.approx(150.0)
    assert v.loc[4, "v_secs_since_prev"] == 90000 - 3700
    assert v.loc[4, "v_tenure_days"] == pytest.approx(90000 / 86400)


def test_other_cards_do_not_leak(fx):
    v = velocity_features(fx, "card", W, prefix="v")
    assert v.loc[5, "v_cnt_7d"] == 0
    assert v.loc[1, "v_amt_7d"] == pytest.approx(10.0)  # card B's $1000 at t=50 not counted


def test_appending_future_rows_does_not_change_past_features(fx):
    before = velocity_features(fx, "card", W, prefix="v")
    future = pd.DataFrame({"TransactionDT": [90000, 95000, 10**6], "TransactionAmt": [5e4, 7e4, 9e4],
                           "card": ["A", "A", "B"]})
    after = velocity_features(pd.concat([fx, future], ignore_index=True), "card", W, prefix="v")
    pd.testing.assert_frame_equal(before, after.iloc[: len(fx)])


def test_row_order_does_not_matter(fx):
    a = velocity_features(fx, "card", W, prefix="v")
    perm = fx.sample(frac=1, random_state=7)
    b = velocity_features(perm, "card", W, prefix="v").loc[fx.index]
    pd.testing.assert_frame_equal(a, b)


def test_missing_key_gives_nan_not_a_shared_bucket():
    df = pd.DataFrame({"TransactionDT": [0, 10, 20], "TransactionAmt": [1.0, 2.0, 3.0],
                       "card": [None, None, "A"]})
    v = velocity_features(df, "card", W, prefix="v")
    assert v.loc[[0, 1]].isna().all().all()
    assert v.loc[2, "v_cnt_7d"] == 0
