"""Leakage guards and hand-computed values for reuse / linkage features."""
import numpy as np
import pandas as pd

from src.features.keys import add_keys
from src.features.linkage import distinct_count


def _fx():
    return pd.DataFrame({
        "TransactionDT": [0, 10, 10, 20, 30, 40],
        "device": ["d1", "d1", "d1", "d1", None, "d2"],
        "card": ["c1", "c2", "c1", "c3", "c9", "c1"],
    })


def test_hand_computed_cards_per_device():
    out = distinct_count(_fx(), "device", "card")
    # r0: {c1}=1; r1 (t=10): prior {c1} + new c2 = 2; r2 (t=10): prior {c1}, c1 not new = 1
    # r3: prior {c1,c2} + c3 = 3; r4: device missing -> NaN; r5: d2 first = 1
    np.testing.assert_array_equal(out[[0, 1, 2, 3, 5]], [1, 2, 1, 3, 1])
    assert np.isnan(out[4])


def test_future_rows_do_not_change_past_counts():
    fx = _fx()
    before = distinct_count(fx, "device", "card")
    future = pd.DataFrame({"TransactionDT": [50, 60, 70], "device": ["d1"] * 3, "card": ["c7", "c8", "c1"]})
    after = distinct_count(pd.concat([fx, future], ignore_index=True), "device", "card")
    np.testing.assert_array_equal(np.nan_to_num(before, nan=-1), np.nan_to_num(after[: len(fx)], nan=-1))


def test_missing_other_value_is_not_counted():
    df = pd.DataFrame({"TransactionDT": [0, 1, 2], "card": ["c", "c", "c"], "email": [None, "x", None]})
    np.testing.assert_array_equal(distinct_count(df, "card", "email"), [0, 1, 1])


def test_uid_key_is_stable_across_float_and_int_storage():
    base = {"TransactionDT": [86400 * 3], "card4": ["visa"], "card6": ["debit"], "D1": [2.0]}
    a = pd.DataFrame({**base, "card1": [1000], "card2": [111.0], "card3": [150.0], "card5": [226.0],
                      "addr1": [315.0]})
    b = pd.DataFrame({**base, "card1": [1000], "card2": pd.array([111], dtype="Int64"),
                      "card3": pd.array([150], dtype="Int64"), "card5": pd.array([226], dtype="Int64"),
                      "addr1": pd.array([315], dtype="Int64")})
    assert add_keys(a)["uid"].iloc[0] == add_keys(b)["uid"].iloc[0] == "1000|111|150|visa|226|debit|315|1"
