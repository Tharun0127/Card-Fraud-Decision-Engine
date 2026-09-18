"""Small synthetic dataset with the IEEE-CIS schema.

Used only for tests and pipeline smoke runs (``python -m src.run_all --synthetic``).
It is never mixed with the real data: synthetic runs write to ``data/synthetic``
and ``outputs_synthetic``. Fraud is injected with a velocity-burst pattern and
an amount effect so that every feature family has something to find.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_raw(n: int = 30_000, seed: int = 0, n_v: int = 40) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    n_cards = max(n // 8, 50)
    card_id = rng.integers(0, n_cards, n)
    dt = np.sort(rng.integers(86_400, 86_400 * 60, n))
    card_risk = rng.random(n_cards) < 0.04
    # fraud: risky cards, burst in time, larger amounts
    p = np.where(card_risk[card_id], 0.45, 0.012)
    y = (rng.random(n) < p).astype(np.int8)
    amt = np.round(rng.lognormal(3.8, 1.0, n) * np.where(y == 1, 1.8, 1.0), 2)
    burst = y == 1
    dt[burst] = np.maximum(dt[burst] - rng.integers(0, 3_600, burst.sum()), 86_400)
    order = np.argsort(dt, kind="stable")
    dt, card_id, y, amt = dt[order], card_id[order], y[order], amt[order]
    prod = rng.choice(["W", "C", "R", "H", "S"], n, p=[0.7, 0.12, 0.07, 0.07, 0.04])
    emails = np.array(["gmail.com", "yahoo.com", "hotmail.com", "anonymous.com", "outlook.com", None], dtype=object)
    tx = pd.DataFrame({
        "TransactionID": np.arange(2_987_000, 2_987_000 + n),
        "isFraud": y,
        "TransactionDT": dt,
        "TransactionAmt": amt,
        "ProductCD": prod,
        "card1": 1000 + card_id,
        "card2": (100 + card_id % 400).astype(float),
        "card3": np.where(card_id % 7 == 0, 185.0, 150.0),
        "card4": np.where(card_id % 3 == 0, "mastercard", "visa"),
        "card5": (200 + card_id % 30).astype(float),
        "card6": np.where(card_id % 4 == 0, "credit", "debit"),
        "addr1": (100 + card_id % 200).astype(float),
        "addr2": 87.0,
        "dist1": np.where(rng.random(n) < 0.6, np.nan, rng.integers(0, 500, n)),
        "dist2": np.where(rng.random(n) < 0.9, np.nan, rng.integers(0, 500, n)),
        "P_emaildomain": emails[np.where(y == 1, rng.integers(2, 6, n), rng.integers(0, 6, n))],
        "R_emaildomain": np.where(rng.random(n) < 0.7, None, emails[rng.integers(0, 5, n)]),
    })
    for i in range(1, 15):
        tx[f"C{i}"] = rng.poisson(1 + 2 * y * (i % 3 == 0), n).astype(float)
    card_start = rng.integers(0, 400, n_cards)
    for i in range(1, 16):
        if i == 1:
            tx["D1"] = np.maximum(dt // 86_400 - card_start[card_id], 0).astype(float)
        else:
            tx[f"D{i}"] = np.where(rng.random(n) < 0.3 + 0.03 * i, np.nan, rng.integers(0, 600, n))
    for i in range(1, 10):
        tx[f"M{i}"] = np.where(rng.random(n) < 0.4, None, np.where(rng.random(n) < 0.5 + 0.2 * y, "T", "F"))
    base = rng.normal(size=(n, 4))
    for i in range(1, n_v + 1):
        v = base[:, i % 4] * (1 + 0.05 * i) + rng.normal(scale=0.3 + 0.1 * (i % 5), size=n) + y * (i % 6 == 0)
        miss = rng.random(n) < (0.1 * (i % 4))
        tx[f"V{i}"] = np.where(miss, np.nan, v)
    has_id = rng.random(n) < 0.25
    ids = tx.loc[has_id, ["TransactionID"]].copy()
    m = len(ids)
    devices = np.array(["Windows", "iOS Device", "MacOS", "SM-G960U", "Trident/7.0"], dtype=object)
    ids["id_01"] = rng.choice([-5.0, 0.0, -10.0], m)
    ids["id_02"] = rng.integers(1, 600_000, m).astype(float)
    for c in ["id_03", "id_05", "id_06", "id_09", "id_11", "id_13", "id_14", "id_17", "id_19", "id_20", "id_32"]:
        ids[c] = np.where(rng.random(m) < 0.4, np.nan, rng.integers(0, 100, m))
    for c in ["id_12", "id_15", "id_16", "id_28", "id_29", "id_34", "id_35", "id_36", "id_37", "id_38"]:
        ids[c] = rng.choice(["Found", "NotFound", "T", "F"], m)
    ids["id_30"] = rng.choice(["Windows 10", "iOS 11.2.1", "Android 7.0", None], m)
    ids["id_31"] = rng.choice(["chrome 63.0", "mobile safari 11.0", "edge 16.0"], m)
    ids["id_33"] = rng.choice(["1920x1080", "2208x1242", None], m)
    ids["DeviceType"] = rng.choice(["desktop", "mobile"], m)
    ids["DeviceInfo"] = devices[rng.integers(0, len(devices), m)]
    ids["id_07"] = np.nan  # near-empty column, should be dropped by the missingness filter
    return tx, ids
