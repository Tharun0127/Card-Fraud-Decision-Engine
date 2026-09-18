"""Time split: strict ordering, tie handling, and failure on bad input."""
import numpy as np
import pandas as pd
import pytest

from src.data.split import check_split, time_split


def _frame(dts):
    return pd.DataFrame({"TransactionID": np.arange(len(dts)) + 1000, "TransactionDT": dts})


def test_split_is_strictly_time_ordered():
    rng = np.random.default_rng(0)
    df = _frame(rng.integers(0, 10_000, size=2_000))
    s = time_split(df, 0.7, 0.15)
    tr, va, te = (df.loc[s == k, "TransactionDT"] for k in ("train", "valid", "test"))
    assert tr.max() < va.min()
    assert va.max() < te.min()
    assert tr.max() < te.min()


def test_split_fractions_are_approximately_respected():
    df = _frame(np.arange(10_000))
    s = time_split(df, 0.7, 0.15)
    assert (s == "train").sum() == 7_000
    assert (s == "valid").sum() == 1_500
    assert (s == "test").sum() == 1_500


def test_tied_timestamps_never_straddle_a_boundary():
    # 10 rows share DT=5 right at the 70% cut; they must land in one split
    dts = np.concatenate([np.arange(65), np.full(10, 100), np.arange(200, 225)])
    df = _frame(dts)
    s = time_split(df, 0.7, 0.15)
    assert s[df["TransactionDT"] == 100].nunique() == 1
    check_split(df.assign(split=s))


def test_split_does_not_depend_on_input_row_order():
    rng = np.random.default_rng(1)
    df = _frame(rng.integers(0, 5_000, size=1_000))
    a = time_split(df, 0.7, 0.15)
    shuffled = df.sample(frac=1, random_state=3)
    b = time_split(shuffled, 0.7, 0.15).reindex(df.index)
    assert (a == b).all()


def test_split_rejects_missing_time_and_bad_fractions():
    df = _frame(np.arange(100).astype(float))
    df.loc[3, "TransactionDT"] = np.nan
    with pytest.raises(ValueError):
        time_split(df, 0.7, 0.15)
    with pytest.raises(ValueError):
        time_split(_frame(np.arange(100)), 0.9, 0.15)


def test_check_split_detects_overlap():
    df = _frame(np.arange(10))
    df["split"] = ["train"] * 6 + ["valid", "train", "test", "test"]
    with pytest.raises(AssertionError):
        check_split(df)


def test_download_integrity_check_rejects_a_wrong_size(tmp_path):
    from src.data.download import verify_files
    (tmp_path / "a.csv").write_bytes(b"12345")
    assert verify_files(tmp_path, {"a.csv": 5})["a.csv"]["bytes"] == 5
    with pytest.raises(RuntimeError):
        verify_files(tmp_path, {"a.csv": 6})
    with pytest.raises(RuntimeError):
        verify_files(tmp_path, {"missing.csv": 1})
