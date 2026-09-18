"""Assemble the model feature matrix.

Families:
- ``raw``: selected provider columns (numeric and low/medium-cardinality categorical).
- ``vesta``: a pruned subset of the V1..V339 block (selection fitted on train only).
- ``velocity``: backward-looking counts / sums per ``card1`` and per ``uid``.
- ``entity_risk``: out-of-fold smoothed target encodings of merchant proxies.
- ``linkage``: distinct-entity counts (cards per device, emails / addresses per card).

Every fitted step (V selection, target encoding, categorical vocabularies) uses
training rows only. The resulting manifest (kept and dropped columns with
reasons) is written to ``outputs/tables``.
"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.features.keys import add_keys
from src.features.linkage import linkage_features
from src.features.target_encoding import target_encode
from src.features.velocity import velocity_features

RAW_NUMERIC = (["TransactionAmt", "card1", "card2", "card3", "card5", "addr1", "addr2", "dist1", "dist2"]
               + [f"C{i}" for i in range(1, 15)] + [f"D{i}" for i in range(1, 16)]
               + ["id_01", "id_02", "id_03", "id_05", "id_06", "id_09", "id_11", "id_13", "id_14",
                  "id_17", "id_19", "id_20", "id_32"])
RAW_CATEGORICAL = (["ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain"]
                   + [f"M{i}" for i in range(1, 10)]
                   + ["DeviceType", "DeviceInfo", "id_12", "id_15", "id_16", "id_28", "id_29",
                      "id_30", "id_31", "id_33", "id_34", "id_35", "id_36", "id_37", "id_38"])
NEVER_FEATURES = {"TransactionID": "row identifier",
                  "TransactionDT": "ordering index; raw value encodes position in time and would not generalise",
                  "isFraud": "label",
                  "split": "split marker",
                  "card_key": "string identity key (used to build features, not a feature)",
                  "uid": "string identity key (used to build features, not a feature)",
                  "D1n": "quasi-identifier (implied account-open day); used only inside uid"}
LINKAGE_SPECS = [
    ("lnk_cards_per_deviceinfo", "DeviceInfo", "card_key"),
    ("lnk_cards_per_id30", "id_30", "card_key"),
    ("lnk_cards_per_id31", "id_31", "card_key"),
    ("lnk_cards_per_device_fp", "device_fp", "card_key"),
    ("lnk_pemails_per_card", "card_key", "P_emaildomain"),
    ("lnk_remails_per_card", "card_key", "R_emaildomain"),
    ("lnk_addr1_per_card", "card_key", "addr1"),
]
MISSING_DROP = 0.95
MIN_CATEGORY_COUNT = 20
OTHER = "__other__"


def select_v_columns(df: pd.DataFrame, train_pos: np.ndarray, y_train: np.ndarray, corr_threshold: float,
                     max_keep: int, sample_rows: int, seed: int) -> tuple[list[str], dict[str, str]]:
    """Prune the V block using training rows only.

    1. Drop columns with training missingness above ``MISSING_DROP``.
    2. Group columns by identical missingness rate (the V block comes in groups
       that share a NaN pattern); inside each group keep a column only if its
       |Pearson correlation| with every already-kept column is <= threshold.
    3. Rank survivors by univariate |AUC - 0.5| and keep the top ``max_keep``.
    """
    v_cols = sorted([c for c in df.columns if re.fullmatch(r"V\d+", c)], key=lambda c: int(c[1:]))
    reasons: dict[str, str] = {}
    rng = np.random.default_rng(seed)
    take = rng.choice(len(train_pos), size=min(sample_rows, len(train_pos)), replace=False)
    take.sort()
    sample = df.iloc[train_pos[take]][v_cols].astype("float32")  # sample rows only: the full V block is ~1 GB
    ys = np.asarray(y_train)[take]
    miss = sample.isna().mean()
    candidates = []
    for c in v_cols:
        if miss[c] > MISSING_DROP:
            reasons[c] = f"missing > {MISSING_DROP:.0%} in training"
        else:
            candidates.append(c)
    groups: dict[float, list[str]] = {}
    for c in candidates:
        groups.setdefault(round(float(miss[c]), 4), []).append(c)
    survivors = []
    for _, cols in sorted(groups.items()):
        block = sample[cols]
        corr = block.corr().abs().to_numpy()
        kept_idx: list[int] = []
        for j, c in enumerate(cols):
            clash = [cols[k] for k in kept_idx if corr[j, k] > corr_threshold]
            if clash:
                reasons[c] = f"|corr| > {corr_threshold} with {clash[0]} (same NaN group)"
            else:
                kept_idx.append(j)
        survivors += [cols[k] for k in kept_idx]
    score = {}
    for c in survivors:
        x = sample[c].fillna(-999).to_numpy()
        score[c] = abs(roc_auc_score(ys, x) - 0.5) if np.unique(x).size > 1 else 0.0
    ranked = sorted(survivors, key=lambda c: (-score[c], int(c[1:])))
    keep = sorted(ranked[:max_keep], key=lambda c: int(c[1:]))
    for c in ranked[max_keep:]:
        reasons[c] = f"outside top {max_keep} by univariate |AUC-0.5| (feature cap)"
    return keep, reasons


def fit_categories(train_col: pd.Series, min_count: int = MIN_CATEGORY_COUNT) -> list[str]:
    vc = train_col.astype("string").value_counts(dropna=True)
    cats = sorted(vc[vc >= min_count].index.tolist())
    return cats + [OTHER]


def apply_categories(col: pd.Series, cats: list[str]) -> pd.Categorical:
    s = col.astype("string")
    known = set(cats)
    s = s.where(s.isna() | s.isin(known), OTHER)
    return pd.Categorical(s, categories=cats)


def build_features(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Return (feature frame incl. metadata columns, manifest).

    ``df`` must be sorted by (TransactionDT, TransactionID) and carry ``split``.
    """
    fcfg = cfg["features"]
    seed = cfg["seed"]
    if not df["TransactionDT"].is_monotonic_increasing:
        raise ValueError("input must be sorted by TransactionDT")
    df = df.copy(deep=False)  # new columns only; pandas copy-on-write leaves the caller's frame untouched
    add_keys(df)
    is_train = (df["split"] == "train").to_numpy()
    y = df["isFraud"]
    dropped: dict[str, str] = {}
    families: dict[str, str] = {}

    # raw columns, with the training-missingness filter
    train_pos = np.flatnonzero(is_train)
    tr_miss = pd.Series({c: float(df[c].iloc[train_pos].isna().mean()) for c in df.columns
                         if not re.fullmatch(r"V\d+", c)})
    raw_num = [c for c in RAW_NUMERIC if tr_miss[c] <= MISSING_DROP]
    raw_cat = [c for c in RAW_CATEGORICAL if tr_miss[c] <= MISSING_DROP]
    for c in set(RAW_NUMERIC + RAW_CATEGORICAL) - set(raw_num + raw_cat):
        dropped[c] = f"missing > {MISSING_DROP:.0%} in training"
    listed = set(RAW_NUMERIC + RAW_CATEGORICAL) | set(NEVER_FEATURES)
    for c in df.columns:
        if c.startswith("id_") and c not in listed:
            dropped[c] = (f"missing > {MISSING_DROP:.0%} in training" if tr_miss.get(c, 0) > MISSING_DROP
                          else "near-duplicate of a kept identity column or free-text with no stable meaning")
    for c, why in NEVER_FEATURES.items():
        dropped[c] = why

    out = pd.DataFrame(index=df.index)
    for c in raw_num:
        out[c] = df[c].astype("float32")
        families[c] = "raw"
    cat_vocab = {}
    for c in raw_cat:
        cat_vocab[c] = fit_categories(df.loc[is_train, c])
        out[c] = apply_categories(df[c], cat_vocab[c])
        families[c] = "raw"
    out["amt_cents"] = ((df["TransactionAmt"] * 1000).round() % 1000 / 1000).astype("float32")
    out["hour_proxy"] = ((df["TransactionDT"] // 3600) % 24).astype("float32")
    families["amt_cents"] = families["hour_proxy"] = "raw"

    # V block
    v_keep, v_reasons = select_v_columns(df, train_pos, y.to_numpy()[is_train], fcfg["v_corr_threshold"],
                                         fcfg["v_max_keep"], fcfg["v_sample_rows"], seed)
    dropped.update(v_reasons)
    for c in v_keep:
        out[c] = df[c].astype("float32")
        families[c] = "vesta"

    # velocity
    windows = fcfg["velocity_windows_seconds"]
    for key in ("card1", "uid"):
        vf = velocity_features(df, key, windows, prefix=f"vel_{key}")
        for c in vf:
            out[c] = vf[c]
            families[c] = "velocity"

    # entity risk
    te = target_encode(df, fcfg["te_columns"], is_train, y, fcfg["te_folds"], fcfg["te_smoothing"])
    for c in te:
        out[c] = te[c]
        families[c] = "entity_risk"

    # linkage
    df["device_fp"] = (df["DeviceInfo"].astype("string") + "|" + df["id_30"].astype("string").fillna("nan")
                       + "|" + df["id_31"].astype("string").fillna("nan") + "|"
                       + df["id_33"].astype("string").fillna("nan"))
    lk = linkage_features(df, LINKAGE_SPECS)
    for c in lk:
        out[c] = lk[c]
        families[c] = "linkage"

    feature_names = list(out.columns)
    if len(feature_names) > fcfg["max_features"]:
        raise ValueError(f"{len(feature_names)} features exceed the cap of {fcfg['max_features']}")
    meta = df[["TransactionID", "TransactionDT", "TransactionAmt", "isFraud", "split", "ProductCD"]].copy()
    meta["card_tenure_days"] = out["vel_uid_tenure_days"]
    meta["uid_prior_cnt"] = out["vel_uid_prior_cnt"]
    frame = pd.concat([meta.add_prefix("meta_"), out], axis=1)
    manifest = {
        "features": feature_names,
        "families": families,
        "categorical": raw_cat,
        "category_vocab": cat_vocab,
        "dropped": dropped,
        "n_features": len(feature_names),
    }
    return frame, manifest


def manifest_tables(manifest: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    kept = pd.DataFrame({"feature": manifest["features"],
                         "family": [manifest["families"][f] for f in manifest["features"]],
                         "categorical": [f in manifest["categorical"] for f in manifest["features"]]})
    dropped = (pd.DataFrame(sorted(manifest["dropped"].items()), columns=["column", "reason"]))
    return kept, dropped


def save_manifest(manifest: dict, path) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1, sort_keys=True)
