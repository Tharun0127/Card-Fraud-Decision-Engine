"""Leakage guards for out-of-fold target encoding."""
import numpy as np
import pandas as pd
import pytest

from src.features.target_encoding import apply_mapping, fit_mapping, oof_encode, target_encode

M = 10.0


def _fixture():
    # 20 training rows (time-ordered), 6 later rows
    cat = ["a"] * 10 + ["b"] * 10 + ["a", "b", "only_test", None, "a", "only_test"]
    y = [1, 0] * 5 + [0] * 10 + [1, 1, 1, 1, 1, 1]
    df = pd.DataFrame({"c": cat})
    is_train = np.array([True] * 20 + [False] * 6)
    return df, pd.Series(y), is_train


def test_category_only_in_test_gets_prior_not_nan():
    df, y, is_train = _fixture()
    out = target_encode(df, ["c"], is_train, y, n_folds=5, smoothing=M)["te_c"].to_numpy()
    prior = y[is_train].mean()
    assert not np.isnan(out).any()
    assert out[22] == pytest.approx(prior)
    assert out[25] == pytest.approx(prior)


def test_hand_computed_smoothed_value_for_test_rows():
    df, y, is_train = _fixture()
    out = target_encode(df, ["c"], is_train, y, n_folds=5, smoothing=M)["te_c"].to_numpy()
    prior = 5 / 20
    expected_a = (5 + M * prior) / (10 + M)  # 'a': 5 frauds in 10
    expected_b = (0 + M * prior) / (10 + M)
    assert out[20] == pytest.approx(expected_a, rel=1e-6)
    assert out[21] == pytest.approx(expected_b, rel=1e-6)


def test_test_labels_are_never_read():
    df, y, is_train = _fixture()
    a = target_encode(df, ["c"], is_train, y, n_folds=5, smoothing=M)
    y2 = y.copy()
    y2[~is_train] = 1 - y2[~is_train]
    b = target_encode(df, ["c"], is_train, y2, n_folds=5, smoothing=M)
    pd.testing.assert_frame_equal(a, b)


def test_oof_row_does_not_see_its_own_label():
    x = pd.Series(["a"] * 10)
    y = pd.Series([0] * 10)
    base = oof_encode(x, y, n_folds=5, smoothing=M)
    y_flip = y.copy()
    y_flip[0] = 1  # row 0 is in fold 0; its own encoding must not move
    flipped = oof_encode(x, y_flip, n_folds=5, smoothing=M)
    assert flipped[0] == pytest.approx(base[0])
    assert flipped[5] != pytest.approx(base[5])  # other folds do see it


def test_missing_values_are_their_own_category():
    mapping, prior = fit_mapping(pd.Series([None, None, "x", "x"]), pd.Series([1, 1, 0, 0]), smoothing=0)
    enc = apply_mapping(pd.Series([None, "x"]), mapping, prior)
    assert enc[0] == pytest.approx(1.0) and enc[1] == pytest.approx(0.0)


def test_numeric_float_categories_match_integer_tokens():
    mapping, prior = fit_mapping(pd.Series([315.0, 315.0, 204.0]), pd.Series([1, 1, 0]), smoothing=0)
    enc = apply_mapping(pd.Series([315.0, np.nan]), mapping, prior)
    assert enc[0] == pytest.approx(1.0)
    assert enc[1] == pytest.approx(prior)


def test_fit_rejects_missing_labels():
    with pytest.raises(ValueError):
        fit_mapping(pd.Series(["a", "b"]), pd.Series([1, np.nan]), smoothing=1)
