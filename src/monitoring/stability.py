"""Score distribution stability across time windows of the test period."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from src.monitoring.psi import band, psi_numeric


def score_stability(ref_scores: np.ndarray, dt: np.ndarray, scores: np.ndarray, y: np.ndarray,
                    window_days: int, n_bins: int, warn: float, alert: float,
                    t_decline: float, t_review: float) -> pd.DataFrame:
    """Per window of ``window_days`` relative days: score PSI vs ``ref_scores``,
    mean score, flag rates at the deployed thresholds, observed fraud rate and PR-AUC."""
    dt = np.asarray(dt)
    w = ((dt - dt.min()) // (window_days * 86400)).astype(int)
    rows = []
    for k in np.unique(w):
        m = w == k
        s, yy = scores[m], y[m]
        v = psi_numeric(ref_scores, s, n_bins)
        rows.append({
            "window": int(k), "n": int(m.sum()),
            "days_from_test_start": float((dt[m].min() - dt.min()) / 86400),
            "mean_score": float(s.mean()), "p99_score": float(np.quantile(s, 0.99)),
            "decline_rate": float((s >= t_decline).mean()),
            "review_rate": float(((s >= t_review) & (s < t_decline)).mean()),
            "fraud_rate": float(yy.mean()),
            "pr_auc": float(average_precision_score(yy, s)) if 0 < yy.sum() < len(yy) else float("nan"),
            "score_psi_vs_valid": v, "band": band(v, warn, alert),
        })
    return pd.DataFrame(rows)
