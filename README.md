# Card Fraud Decision Engine

![Python](https://img.shields.io/badge/python-3.12-3776AB?logo=python&logoColor=white)
![LightGBM](https://img.shields.io/badge/model-LightGBM-2a78d6)
![Tests](https://img.shields.io/badge/tests-passing-1baf7a)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

**A fraud score is not a decision.** This project builds a LightGBM fraud model on
the IEEE-CIS card-not-present dataset and then does the part that usually gets
skipped: it turns the score into an **approve / review / decline policy** chosen to
maximise expected profit under a fixed manual-review capacity, and shows, on a
held-out time period, that the **profit-optimal cutoff is not the AUC-optimal
cutoff**. Every result is reported in money and customer impact, not only in AUC.

### At a glance

- **Net benefit of the selected policy:** $2,811.82 saved per
  1,000 test transactions versus approving everything, while
  declining 4.88% of legitimate customers and
  capturing 66.6% of fraud dollars.
- **AUC-optimal vs profit-optimal:** the Youden-J cutoff leaves
  $213.54 per 1,000
  transactions on the table  (interval excludes zero).
- **Against a hand-written rules baseline:** $2,559 (95% CI $2,264 to $2,833)
  more per 1,000 transactions.
- **Robustness to the cost assumptions:** beats the rules baseline in 100.0%
  of 750 cost scenarios; profit cutoff beats Youden-J in 80.0%;
  full pre-registered ordering in 60.7% (H4 **FAIL**).
- **Model:** test PR-AUC 0.552, ROC-AUC 0.903,
  on a strictly later time period than training.
- **Verified:** 63 automated tests passing
  (0 failing), including leakage guards and a determinism check;
  every number in this README is machine-checked against `outputs/results.json`.

The dollar figures depend on cost parameters that are **assumptions**, not
measurements, so no absolute dollar amount here is a forecast of real savings.
Even the **ranking** of policies is only partly robust to those assumptions: the
model-driven policy beats the rules baseline in 100.0% of the
pre-registered sensitivity grid, but the full pre-registered ordering holds in
60.7% of it, short of the 80% target (H4 failed).
See [Limitations](#limitations) directly below.

## Limitations

Read these before the results. Full list with severity ratings: [FINDINGS.md](FINDINGS.md).

- **Cost parameters are invented.** Margin rate, churn penalty, review cost and
  analyst catch rate are defaults chosen in [PRE_REGISTRATION.md](PRE_REGISTRATION.md),
  not estimated from any portfolio. Absolute dollar results move by a large factor
  across the pre-registered grid (below); only the policy ranking is claimed.
- **Labels are chargebacks, not confirmed fraud.** `isFraud` marks reported
  chargebacks and propagates the label to later transactions on the same account.
  Some "fraud" is friendly fraud or disputes; some fraud is never charged back.
- **No true timestamp.** `TransactionDT` is a seconds offset from an undisclosed
  reference. It is used only to order rows and measure relative durations. Days
  and "per day" figures are in relative time.
- **Drift between training and test.** 17 of
  148 model features exceed a PSI of 0.25
  between the training and test periods (6 more between
  0.10 and 0.25). Results are for one
  period of one merchant population.
- **Review queue over capacity on test.** The three-way policy was sized to
  34.5 reviews per day on validation; on test it averaged
  38.2 and exceeded capacity on
  20 of 31 relative days. Overflow handling is not modelled.
- **Data from a public mirror.** The official competition download was refused, so
  the files came from a public Kaggle re-upload, verified against the official byte
  sizes and row count (deviation logged in FINDINGS.md).
- **One test period, one data source.** The test split covers
  30.8 relative days and 88,581 transactions.
  Intervals below are bootstrap intervals over transactions within that period;
  they do not cover period-to-period variation.

## Headline results

Population: the final 15% of labelled
transactions in time order (the test split; 88,581 transactions,
3.48% labelled fraud, $469,609
of fraud by amount). All thresholds were chosen on the validation split, never on
test. Net benefit is cost saved relative to approving everything. Default cost
parameters: margin rate 0.02, churn penalty
$10, review cost $5,
review catch rate 0.85, review capacity
34.5 cases per day
(1% of mean daily training volume).

| Policy | Net benefit per 1,000 txns | Net benefit, test period | Good-customer disruption rate | Fraud captured (count) | Fraud captured ($) | Reviews per day | Declined |
|---|---:|---:|---:|---:|---:|---:|---:|
| Approve everything | $0 | $0 | 0.00% | 0.0% | 0.0% | 0.0 | 0.00% |
| Rules baseline (amount above training P90 and new card) | $253 | $22,421 | 3.64% | 5.3% | 21.0% | 0.0 | 3.69% |
| Full model, Youden-J cutoff (AUC-optimal) | $2,521 | $223,340 | 13.11% | 79.1% | 80.2% | 0.0 | 15.41% |
| Full model, profit-optimal cutoff | $2,735 | $242,256 | 5.87% | 68.6% | 66.4% | 0.0 | 8.05% |
| No-velocity model, three-way policy | $2,797 | $247,792 | 6.92% | 72.2% | 71.1% | 39.2 | 9.13% |
| **Full model, three-way policy (capacity-constrained)** | **$2,812** | **$249,073** | **4.88%** | **68.8%** | **66.6%** | **38.2** | **7.02%** |

Fraud captured counts a reviewed fraud case as caught with probability equal to
the catch rate. The three-way policy also sends
1.26% of legitimate customers
to review, which delays but does not decline them.

Paired bootstrap comparisons on test (1,000 resamples, net
benefit in $ per 1,000 transactions):

| Comparison | Difference | Reading |
|---|---|---|
| Profit-optimal cutoff minus Youden-J cutoff | $213.5 (95% CI $53.8 to $380.7) | interval excludes zero |
| Three-way policy minus rules baseline | $2,558.7 (95% CI $2,264.5 to $2,833.1) | interval excludes zero |
| Three-way policy minus profit-optimal single cutoff | $77.0 (95% CI $54.3 to $104.8) | interval excludes zero |
| Full model minus no-velocity model (both three-way) | $14.5 (95% CI -$137.7 to $160.4) | interval includes zero: not distinguishable from no difference |

Pre-registered hypotheses: H1 (profit cutoff beats Youden-J) **PASS**,
H2 (three-way policy beats rules) **PASS**,
H3 (velocity features raise PR-AUC) **PASS**,
H4 (ranking stable in at least 80% of the sensitivity grid) **FAIL**.
The pre-registered success rule (H1 and H2 both pass) is met.

### Profit curve

![Profit curve](outputs/figures/profit_curve.png)

Net benefit per 1,000 transactions as the single
decline threshold moves, on validation (where the threshold was chosen) and test.
At the validation-selected optimum the full model saves
$2,734.85 per 1,000
test transactions by declining 8.05% of them.
The test-period oracle optimum, chosen with test labels and therefore not
achievable, is $2,754.60; the gap is the
cost of choosing the threshold one period earlier.

### AUC-optimal versus profit-optimal cutoff

| | Youden-J (AUC-optimal) | Profit-optimal |
|---|---:|---:|
| Score threshold (chosen on validation) | 0.0014 | 0.0045 |
| Share of test transactions declined | 15.41% | 8.05% |
| Good-customer disruption rate | 13.11% | 5.87% |
| Legitimate customers declined (test) | 11,210 | 5,015 |
| Fraud captured ($) | 80.2% | 66.4% |
| Net benefit per 1,000 txns | $2,521.31 | $2,734.85 |

The two cutoffs differ by 0.0030 in score.
Using the Youden-J cutoff leaves $213.54 per
1,000 transactions on the table
($18,916 over the test period,
$615 per relative day), and declines 6,195 more legitimate customers.
Youden's J weights a
missed fraud and a false decline equally in rate terms; the cost model does not,
because a missed fraud costs the full amount and a false decline costs a margin
plus a churn penalty.

![ROC with cutoffs](outputs/figures/roc_cutoffs.png)

When both cutoffs threshold the same score, their decline sets are nested, so the
swap set runs in one direction only. Its composition (amounts, card tenure,
product mix) is in [REPORT.md](REPORT.md#swap-set) and `outputs/tables/swap_set.csv`.

### Model metrics

Test split, full model versus no-velocity model:

| Metric | Full model | No-velocity model | Difference (paired bootstrap) |
|---|---:|---:|---|
| PR-AUC (primary) | 0.5524 | 0.5458 | 0.0066 (95% CI 0.0018 to 0.0115); interval excludes zero |
| ROC-AUC | 0.9028 | 0.9040 | -0.0011 (95% CI -0.0038 to 0.0015); interval includes zero: not distinguishable from no difference |
| Precision in top 1% | 88.0% | 87.5% | 0.6 pp (95% CI -0.8 to 1.9 pp) |
| Recall in top 1% | 25.3% | 25.1% | 0.2 pp (95% CI -0.6 to 0.9 pp) |
| Precision in top 5% | 42.3% | 41.9% | 0.4 pp (95% CI -0.5 to 1.2 pp) |
| Recall in top 5% | 60.8% | 60.2% | 0.6 pp (95% CI -0.5 to 1.6 pp) |

Against the rules baseline, at the rules' own flag volume
(3.69% of test transactions): rules precision
5.0% and recall 5.3%;
the full model's top transactions at the same volume reach precision
52.0% and recall
55.2%
(precision difference 47.0 pp (95% CI 45.3 to 48.6 pp)).

### Sensitivity to the cost assumptions

Thresholds were re-selected on validation at each of 750 points of
the pre-registered grid. Test net benefit per 1,000
transactions of the three-way policy ranges from
$1,649 to
$4,670 across
the grid, so absolute dollar figures should not be quoted without their
assumptions. How often each ordering of policies survives:

| Ordering on test | Share of grid cells where it holds |
|---|---:|
| Three-way policy at least as good as the single profit-optimal cutoff | 74.9% |
| Profit-optimal cutoff better than Youden-J | 80.0% |
| Three-way policy better than rules baseline | 100.0% |
| All three at once | 60.7% |

![Optimal threshold across margin and churn assumptions](outputs/figures/sensitivity_threshold_margin_churn.png)

## How the work is verified

| Check | What it proves | Where |
|---|---|---|
| Pre-registration | Split rule, metrics, baselines, cost ranges and success criteria were fixed before any model existed | `PRE_REGISTRATION.md` |
| Time-split test | `max(train.TransactionDT) < min(validation) < min(test)`; ties never straddle a boundary | `tests/test_split.py` |
| Velocity leakage guards | On a hand-built fixture, a transaction never sees itself, same-second peers, other cards, or the future; appending future rows changes nothing | `tests/test_velocity.py` |
| Target-encoding guards | Unseen categories get the prior (not NaN, not a leaked value); a row never sees its own label; test labels are never read | `tests/test_target_encoding.py` |
| Pipeline label isolation | Flipping every validation and test label changes no feature value | `tests/test_pipeline.py` |
| Cost-function unit tests | Money is checked against hand-computed values for approve, decline, review and capacity-limited policies | `tests/test_costs.py` |
| Determinism | Features, model predictions and selected thresholds are bit-identical across runs under the fixed seed | `tests/test_pipeline.py` |
| Model archive | JSON archive (LightGBM native model string, no pickle) reloads to bit-identical predictions | `tests/test_pipeline.py` |
| Number provenance | Build fails if any number in README, REPORT, FINDINGS or MONITORING_PLAN is absent from `results.json` | `src/reporting/check_numbers.py` |

Latest run: **63 passed, 0 failed,
0 skipped** across 9 test files.

## Architecture

```mermaid
flowchart LR
    A[Kaggle IEEE-CIS<br/>transactions + identity] --> B[Typed load and join<br/>time-ordered split]
    B --> C[Features<br/>velocity, entity risk,<br/>linkage, raw, Vesta]
    C --> D[LightGBM<br/>expanding-window CV<br/>on train only]
    D --> E[Scores on<br/>validation and test]
    E --> F[Decision layer<br/>cost model, threshold sweep,<br/>three-way policy under capacity]
    F --> G[Sensitivity<br/>across cost assumptions]
    E --> H[Monitoring<br/>PSI, score stability]
    F --> I[results.json]
    G --> I
    H --> I
    I --> J[README, REPORT,<br/>FINDINGS, figures<br/>number-checked]
```

Thresholds are chosen on the validation period and applied once to the test
period. Nothing is tuned on test.

## Reproduce

Requires Python 3.12 and a Kaggle API token (`~/.kaggle/kaggle.json` or
`~/.kaggle/access_token`). The results in this README were produced from the
public Kaggle re-upload `lnasiri007/ieeecis-fraud-detection` of the competition files; each
file was verified byte-for-byte in size against the official listing and the
merged row count against the published total (see the deviation in
[FINDINGS.md](FINDINGS.md)). To use the official source instead, accept the
competition rules at <https://www.kaggle.com/c/ieee-fraud-detection/rules> and set
`kaggle.source: competition` in `config/pipeline.yaml`.

```bash
make setup     # virtualenv + pinned requirements
make all       # download, features, train, decide, monitor, report, tests
```

`make all` runs `python -m src.run_all` and then the test suite. Individual
stages: `make data`, `make features`, `make train`, `make decide`, `make report`,
`make test`. `make smoke` runs the whole pipeline on a small synthetic dataset
with the same schema, without Kaggle access. The seed is fixed in
`config/pipeline.yaml`; LightGBM runs with `deterministic=true` and a fixed thread
count. Raw and derived data are gitignored and rebuilt by the pipeline.

## Layout

| Path | Contents |
|---|---|
| `PRE_REGISTRATION.md` | Split rule, metrics, baselines, cost ranges, success rule; frozen before modelling |
| `DECISIONS.md` | Design choices and the reasoning behind each |
| `DATA_DICTIONARY.md` | Column meanings, missingness, identifier cardinality, class balance by period |
| `REPORT.md` | Full results in model-risk-report form |
| `FINDINGS.md` | Severity-rated limitations and deviations from pre-registration |
| `MONITORING_PLAN.md` | What to watch in production, triggers, ownership |
| `config/` | Pipeline and cost parameters |
| `src/data`, `src/features`, `src/models` | Data, feature families, model and baselines |
| `src/decision` | Cost model, threshold sweeps, three-way policy, swap sets, sensitivity |
| `src/monitoring` | PSI and score stability |
| `src/reporting` | Figures, document rendering, number check |
| `outputs/` | `results.json`, figures (PNG), tables (CSV) |
| `artifacts/models/` | Models archived as JSON (LightGBM native model string) |
| `tests/` | Leakage guards, split, cost functions, determinism, number check |

Licence: MIT.
