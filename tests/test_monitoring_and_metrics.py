"""PSI, metrics, bootstrap and rules baseline."""
import numpy as np
import pandas as pd
import pytest

from src.models.baselines import amount_threshold, rules_decline
from src.models.evaluate import mean_diff_ci, metric_diff_ci, precision_recall_at, ranking_metrics
from src.monitoring.psi import band, psi_categorical, psi_numeric


def test_psi_is_zero_for_identical_distributions():
    x = np.random.default_rng(0).normal(size=5_000)
    assert psi_numeric(x, x.copy()) == pytest.approx(0.0, abs=1e-12)


def test_psi_detects_a_shift_and_bands_it():
    rng = np.random.default_rng(1)
    ref, cur = rng.normal(size=20_000), rng.normal(1.0, 1, size=20_000)
    v = psi_numeric(ref, cur)
    assert v > 0.25 and band(v, 0.10, 0.25) == "significant"
    assert band(0.05, 0.10, 0.25) == "stable" and band(0.15, 0.10, 0.25) == "moderate"


def test_psi_counts_missingness_shift():
    ref = np.r_[np.arange(1000.0), np.full(10, np.nan)]
    cur = np.r_[np.arange(500.0), np.full(510, np.nan)]
    assert psi_numeric(ref, cur) > 0.25


def test_psi_categorical_flags_unseen_categories():
    ref = pd.Series(["a"] * 50 + ["b"] * 50)
    same = psi_categorical(ref, ref.copy())
    new = psi_categorical(ref, pd.Series(["a"] * 20 + ["zzz"] * 80))
    assert same == pytest.approx(0.0, abs=1e-12) and new > 0.25


def test_precision_recall_at_hand_computed():
    y = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 1])
    s = np.arange(10)[::-1] / 10.0  # row 0 highest
    p, r = precision_recall_at(y, s, k=3)
    assert p == pytest.approx(2 / 3) and r == pytest.approx(2 / 3)
    m = ranking_metrics(y, s, fracs=(0.1,))
    assert m["precision_at_10pct"] == 1.0


def test_paired_bootstrap_ci_contains_zero_for_identical_models():
    rng = np.random.default_rng(2)
    y = (rng.random(2_000) < 0.1).astype(int)
    s = rng.random(2_000) + y * 0.3
    ci = metric_diff_ci(y, s, s.copy(), "pr_auc", n_boot=50, seed=1)
    assert ci["point"] == 0 and not ci["excludes_zero"]


def test_mean_diff_ci_is_deterministic_and_detects_real_gap():
    rng = np.random.default_rng(3)
    a, b = rng.normal(1, 1, 5_000), rng.normal(0, 1, 5_000)
    c1, c2 = mean_diff_ci(a, b, 200, seed=9), mean_diff_ci(a, b, 200, seed=9)
    assert c1 == c2 and c1["excludes_zero"] and c1["ci_low"] > 0


def test_rules_baseline_needs_both_conditions():
    thr = amount_threshold(np.arange(1, 101, dtype=float), 90)
    amt = np.array([500.0, 500.0, 5.0, 500.0])
    prior = np.array([0, 3, 0, np.nan])
    np.testing.assert_array_equal(rules_decline(amt, prior, thr), [True, False, False, False])
    with pytest.raises(ValueError):
        amount_threshold(np.array([]), 90)
