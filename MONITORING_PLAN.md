# Monitoring plan

Scope: the full LightGBM model and the capacity-constrained three-way policy
(approve / review / decline) described in REPORT.md. Generated from
`outputs/results.json`; trigger values live in `config/pipeline.yaml` under
`monitoring.plan`.

## Owners

| Role | Owns |
|---|---|
| Fraud strategy lead | The decision policy: thresholds, cost parameters, capacity. Signs off any threshold change. |
| Model owner (data science) | Model, features, retraining, this plan's analyses. |
| Fraud operations manager | Review queue: staffing, daily volume, analyst outcomes (catch rate, analyst declines). |
| Model risk / validation | Independent review of retraining and of any change to the cost model. |
| ML engineering | Feature pipeline health, scoring latency, data contracts. |

## What to watch

| Signal | Frequency | Reference | Investigate when | Owner |
|---|---|---|---|---|
| Feature PSI, every model feature | Weekly | Training period | Any feature at or above 0.25, or at least 5 features at or above 0.10 | Model owner |
| Score distribution PSI | Weekly (7-day windows) | Validation-period scores | Above 0.10; retraining review above 0.25 | Model owner |
| Decline rate and review rate | Daily | Launch rates (7.02% declined, 1.33% reviewed on test) | Moves by more than 50% of the launch rate | Fraud strategy |
| Review queue load | Daily | Capacity 34.5 per day | Over capacity 3 days in a row | Fraud operations |
| Analyst catch rate and analyst declines | Monthly | Assumed 0.85 | Measured value outside the pre-registered range | Fraud operations |
| PR-AUC, capture rate, disruption rate on matured labels | Monthly, lagged 90 days | Test-period values | PR-AUC falls by more than 15% relative to launch | Model owner |
| Net benefit at current cost parameters | Monthly, lagged | Test-period value | Sign change, or policy ranking changes | Fraud strategy |
| Input data contracts (nulls, new categories, schema) | Every batch | Training vocabulary and missingness | New unseen-category share or missingness shifts materially | ML engineering |

Current status against these checks (test period vs training period):
17 features at or above 0.25,
6 between 0.10 and 0.25;
score PSI validation vs test 0.009; largest weekly
score PSI 0.023; 20
days over review capacity. At launch these would already trigger the feature-PSI
investigation; see FINDINGS.md.

## Actions

**Investigation** (any trigger above): the owner reviews within one working
cycle, records cause (data pipeline fault, population change, fraud pattern
change, seasonality) and decides: no action, threshold re-selection, or retrain.

**Threshold re-selection without retraining**: when queue load breaches capacity
for the configured run of days, when a cost parameter is re-measured, or when the
decline rate leaves its band while ranking metrics hold. Re-run the decision
stage on the latest matured-label window; fraud strategy signs off.

**Retraining**: when matured-label PR-AUC falls by more than the configured
relative drop, when score PSI exceeds the retraining level, or on a fixed
quarterly schedule, whichever comes first. Retraining follows the same
pre-registered procedure (time split, expanding-window CV inside training,
thresholds selected on a later validation window, report-only test window) and
is validated by model risk before promotion.

**Rollback**: keep the previous model archive (JSON) and thresholds. If a new
model's first matured-label month underperforms the previous model on net
benefit at current cost parameters, revert.

## Known gaps

Label delay means performance signals arrive
90 days late; the drift and volume signals above are
the early warnings. Chargeback labels are an imperfect proxy for fraud
(FINDINGS.md).
