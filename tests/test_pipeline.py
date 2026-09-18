"""Pipeline-level guards on a small synthetic IEEE-shaped dataset:
determinism under the fixed seed, label isolation, model archive round-trip."""
import copy

import numpy as np
import pandas as pd
import pytest

from src.config import pipeline_config
from src.data.split import time_split
from src.data.synthetic import make_raw
from src.decision.costs import CostParams
from src.decision.policy import select_policies
from src.features.build import OTHER, apply_categories, build_features, fit_categories
from src.models import lgbm


@pytest.fixture(scope="module")
def cfg():
    c = copy.deepcopy(pipeline_config())
    c["features"].update(v_sample_rows=5_000, v_max_keep=10)
    c["model"].update(max_rounds=60, early_stopping_rounds=10, cv_folds=2,
                      tuning_grid={"num_leaves": [7], "min_child_samples": [20], "feature_fraction": [0.8]})
    c["n_threads"] = 2
    return c


@pytest.fixture(scope="module")
def base(cfg):
    tx, ids = make_raw(n=6_000, seed=11, n_v=15)
    df = tx.merge(ids, on="TransactionID", how="left")
    df = df.sort_values(["TransactionDT", "TransactionID"], kind="mergesort").reset_index(drop=True)
    df["split"] = time_split(df, cfg["split"]["train_frac"], cfg["split"]["valid_frac"])
    return df


@pytest.fixture(scope="module")
def built(base, cfg):
    return build_features(base, cfg)


def _train_predict(frame, manifest, cfg):
    feats, cats = manifest["features"], manifest["categorical"]
    tr = frame["meta_split"] == "train"
    params = {**lgbm.base_params(cfg), "num_leaves": 7, "min_child_samples": 20}
    bst = lgbm.fit(frame.loc[tr, feats], frame.loc[tr, "meta_isFraud"], cats, params, 40)
    return bst, bst.predict(frame.loc[~tr, feats])


def test_feature_build_is_deterministic(base, cfg, built):
    frame2, manifest2 = build_features(base, cfg)
    pd.testing.assert_frame_equal(built[0], frame2)
    assert built[1]["features"] == manifest2["features"]


def test_training_and_policy_selection_are_deterministic(built, cfg):
    frame, manifest = built
    _, p1 = _train_predict(frame, manifest, cfg)
    _, p2 = _train_predict(frame, manifest, cfg)
    np.testing.assert_array_equal(p1, p2)
    va = frame[frame["meta_split"] != "train"]
    y, a = va["meta_isFraud"].to_numpy(), va["meta_TransactionAmt"].to_numpy()
    grid = {"n_quantiles": 200, "n_linear": 51, "n_policy_grid": 60}
    s1 = select_policies(p1, y, a, 10.0, CostParams(), 5.0, grid)
    s2 = select_policies(p2, y, a, 10.0, CostParams(), 5.0, grid)
    assert [s1[k] for k in ("t_profit", "t_youden", "t_review", "t_decline")] == \
           [s2[k] for k in ("t_profit", "t_youden", "t_review", "t_decline")]


def test_flipping_non_training_labels_changes_no_feature(base, cfg, built):
    flipped = base.copy()
    later = flipped["split"] != "train"
    flipped.loc[later, "isFraud"] = 1 - flipped.loc[later, "isFraud"]
    frame2, _ = build_features(flipped, cfg)
    feats = built[1]["features"]
    pd.testing.assert_frame_equal(built[0][feats], frame2[feats])


def test_no_label_or_identifier_in_feature_list(built):
    feats = set(built[1]["features"])
    assert not feats & {"isFraud", "TransactionID", "TransactionDT", "split", "uid", "card_key"}
    assert not any(f.startswith("meta_") for f in feats)


def test_feature_cap_and_every_family_present(built, cfg):
    manifest = built[1]
    assert manifest["n_features"] <= cfg["features"]["max_features"]
    assert {"raw", "velocity", "entity_risk", "linkage"} <= set(manifest["families"].values())
    assert all(manifest["families"][f] for f in manifest["features"])


def test_high_missing_column_is_dropped_with_a_reason(built):
    assert "id_07" in built[1]["dropped"] and "missing" in built[1]["dropped"]["id_07"]


def test_unseen_category_maps_to_other_not_nan():
    cats = fit_categories(pd.Series(["a"] * 30 + ["b"] * 30), min_count=20)
    out = apply_categories(pd.Series(["a", "zzz", None]), cats)
    assert out[0] == "a" and out[1] == OTHER and pd.isna(out[2])


def test_model_json_round_trip_is_exact(built, cfg, tmp_path):
    frame, manifest = built
    bst, pred = _train_predict(frame, manifest, cfg)
    lgbm.save_json(bst, tmp_path / "m.json", {"categorical": manifest["categorical"]})
    text = (tmp_path / "m.json").read_text(encoding="utf-8")
    assert text.lstrip().startswith("{")  # JSON, not a pickle
    bst2, meta = lgbm.load_json(tmp_path / "m.json")
    tr = frame["meta_split"] == "train"
    np.testing.assert_array_equal(pred, bst2.predict(frame.loc[~tr, manifest["features"]]))
    assert meta["categorical"] == manifest["categorical"]


def test_expanding_folds_never_train_on_the_future():
    for tr, va in lgbm.expanding_folds(1_000, 5):
        assert tr.max() < va.min()
        assert tr.min() == 0
