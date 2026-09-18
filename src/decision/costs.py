"""Cost model for approve / review / decline decisions.

Per transaction with amount A (all parameters from ``config/costs.yaml``):

- fraud approved:      A
- legitimate declined: margin_rate * A + churn_penalty
- sent to review:      review_cost, plus (1 - review_catch_rate) * A if it is fraud
                       (a missed reviewed fraud is approved and lost; a reviewed
                       legitimate transaction is approved)
- fraud declined, legitimate approved: 0

Net benefit of a policy = cost(approve everything) - cost(policy). Costs of
review outcomes are expected values over the analyst's catch probability.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import numpy as np

APPROVE, REVIEW, DECLINE = 0, 1, 2


@dataclass(frozen=True)
class CostParams:
    margin_rate: float = 0.02
    churn_penalty: float = 10.0
    review_cost: float = 5.0
    review_catch_rate: float = 0.85

    def __post_init__(self):
        if not 0 <= self.review_catch_rate <= 1:
            raise ValueError("review_catch_rate must be in [0, 1]")
        if self.margin_rate < 0 or self.churn_penalty < 0 or self.review_cost < 0:
            raise ValueError("cost parameters must be non-negative")

    @classmethod
    def from_config(cls, cfg: dict) -> "CostParams":
        return cls(**{k: float(cfg[k]) for k in ("margin_rate", "churn_penalty", "review_cost",
                                                   "review_catch_rate")})

    def with_(self, **kw) -> "CostParams":
        return replace(self, **kw)

    def to_dict(self) -> dict:
        return asdict(self)


def _check(y, amount):
    y = np.asarray(y)
    amount = np.asarray(amount, dtype=float)
    if y.shape != amount.shape:
        raise ValueError("y and amount must have the same shape")
    if np.isnan(amount).any() or (amount < 0).any():
        raise ValueError("amounts must be non-negative and not missing")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("labels must be 0/1")
    return y.astype(np.int8), amount


def false_decline_cost(amount, p: CostParams) -> np.ndarray:
    return p.margin_rate * np.asarray(amount, dtype=float) + p.churn_penalty


def row_costs(decision, y, amount, p: CostParams) -> np.ndarray:
    """Expected cost of each transaction under ``decision`` (0 approve, 1 review, 2 decline)."""
    y, amount = _check(y, amount)
    d = np.asarray(decision)
    if d.shape != y.shape or not np.isin(d, (APPROVE, REVIEW, DECLINE)).all():
        raise ValueError("decision must be an array of 0/1/2 aligned with y")
    fraud = y == 1
    c = np.zeros(len(y))
    c[(d == APPROVE) & fraud] = amount[(d == APPROVE) & fraud]
    legit_dec = (d == DECLINE) & ~fraud
    c[legit_dec] = false_decline_cost(amount[legit_dec], p)
    rev = d == REVIEW
    c[rev] = p.review_cost + np.where(fraud[rev], (1 - p.review_catch_rate) * amount[rev], 0.0)
    return c


def row_benefit(decision, y, amount, p: CostParams) -> np.ndarray:
    """Per-row cost saved versus approving everything."""
    y, amount = _check(y, amount)
    baseline = np.where(y == 1, amount, 0.0)
    return baseline - row_costs(decision, y, amount, p)


def policy_summary(decision, y, amount, p: CostParams, days: float) -> dict:
    """Money and customer-impact summary of one decision vector."""
    y, amount = _check(y, amount)
    d = np.asarray(decision)
    fraud, legit = y == 1, y == 0
    cost = row_costs(d, y, amount, p)
    benefit = np.where(fraud, amount, 0.0) - cost
    n = len(y)
    caught_w = np.where(d == DECLINE, 1.0, np.where(d == REVIEW, p.review_catch_rate, 0.0))
    n_review = int((d == REVIEW).sum())
    return {
        "n": int(n),
        "total_cost": float(cost.sum()),
        "approve_all_cost": float(amount[fraud].sum()),
        "net_benefit": float(benefit.sum()),
        "net_benefit_per_1000": float(benefit.sum() / n * 1000) if n else float("nan"),
        "decline_rate": float((d == DECLINE).mean()),
        "review_rate": float((d == REVIEW).mean()),
        "good_customer_disruption_rate": float((d[legit] == DECLINE).mean()) if legit.any() else float("nan"),
        "good_customer_review_rate": float((d[legit] == REVIEW).mean()) if legit.any() else float("nan"),
        "fraud_capture_rate_count": float(caught_w[fraud].sum() / fraud.sum()) if fraud.any() else float("nan"),
        "fraud_capture_rate_dollars": (float((caught_w[fraud] * amount[fraud]).sum() / amount[fraud].sum())
                                       if fraud.any() else float("nan")),
        "fraud_loss_remaining": float(cost[fraud].sum()),
        "false_decline_cost": float(cost[legit & (d == DECLINE)].sum()),
        "review_cost_total": float(n_review * p.review_cost),
        "n_declined": int((d == DECLINE).sum()),
        "n_reviewed": n_review,
        "n_false_declines": int((legit & (d == DECLINE)).sum()),
        "reviews_per_day": float(n_review / days) if days > 0 else float("nan"),
    }
