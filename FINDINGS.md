# Findings: limitations and deviations

Generated from `outputs/results.json`. Severity reflects how much a finding
could change a decision made on the basis of this repository: **High** = could
reverse or invalidate a headline conclusion; **Medium** = changes the size of a
result or its applicability; **Low** = worth knowing, unlikely to change a
conclusion.

## Summary

| Finding | Severity |
|---|---|
| Cost parameters are assumptions, not measurements | High |
| Labels are chargebacks with account-level propagation, not confirmed fraud | High |
| Feature drift between training and test periods | High |
| Pre-registered hypothesis H4 failed | Medium |
| No true timestamp; relative time only | Medium |
| Confidence intervals cover row sampling, not period-to-period variation | Medium |
| Label delay not modelled | Medium |
| Review capacity is a daily average, not enforced per day | Medium |
| Out-of-fold target encoding uses later training blocks for earlier rows | Low |
| Feature-fitting steps precede cross-validation | Low |
| Composite account key is a proxy | Low |
| Reviewed legitimate customers assumed approved; reviews assumed independent of queue load | Low |

## High

### Cost parameters are assumptions

**What.** Margin rate 0.02, churn penalty
$10, review cost $5 and catch
rate 0.85 were set in PRE_REGISTRATION.md from
general reasoning, not from any issuer's or merchant's data. Fraud loss is taken
as the full transaction amount, ignoring chargeback fees, recoveries and
liability shift.

**Impact.** Across the pre-registered grid the three-way policy's test net
benefit ranges from $1,649 to $4,670 per
1,000 transactions. No absolute dollar figure in
this repository should be quoted as an estimate of real savings. The ordering of
policies holds in 60.7% of cells (all three orderings
jointly), which is the supportable claim.

**Recommendation.** Replace each parameter with a measured value before use:
margin from finance, churn penalty from a holdout of declined-then-lost
customers, review cost and catch rate from the review team's case data.
Re-run `make decide`; the thresholds re-select automatically.

### Labels are chargebacks, not confirmed fraud

**What.** `isFraud` = 1 when a chargeback was reported on the card, and the
provider propagated the label to later transactions of the same account, email
or billing address. Friendly fraud and merchant disputes count as fraud; fraud
that was never disputed counts as legitimate.

**Impact.** The model learns chargeback propensity. Because labels are
propagated at account level, account-level features (the `uid` velocity family
in particular) can learn "this account was already labelled" rather than "this
transaction is fraud". This inflates measured performance in a way a production
system, which learns labels weeks later, would not see. The measured value of
the velocity family (H3 passed)should be read with this in mind.

**Recommendation.** In production, train only on labels matured by the
chargeback window, and evaluate with a label-delay simulation: features at
decision time, labels only as of their availability date.

### Feature drift between training and test

**What.** 17 of 148 features have PSI at
or above 0.25 between the training and test periods;
6 more lie between 0.10 and
0.25. Largest PSI 5.46.

| Feature breaching the alert band | Family | PSI |
|---|---|---:|
| `vel_card1_tenure_days` | velocity | 5.462 |
| `te_R_emaildomain` | entity_risk | 4.491 |
| `te_ProductCD` | entity_risk | 4.370 |
| `te_P_emaildomain` | entity_risk | 3.340 |
| `id_31` | raw | 1.080 |
| `lnk_cards_per_id30` | linkage | 0.771 |
| `id_13` | raw | 0.533 |
| `lnk_cards_per_deviceinfo` | linkage | 0.474 |
| `vel_card1_prior_cnt` | velocity | 0.337 |
| `D11` | raw | 0.331 |
| `M9` | raw | 0.331 |
| `M8` | raw | 0.330 |
| `M7` | raw | 0.330 |
| `vel_uid_tenure_days` | velocity | 0.288 |
| `M2` | raw | 0.280 |
| `M3` | raw | 0.279 |
| `M1` | raw | 0.279 |

Features in the investigate band: `id_30` (0.246), `te_addr1` (0.165), `lnk_cards_per_id31` (0.159), `lnk_pemails_per_card` (0.149), `lnk_remails_per_card` (0.124), `vel_uid_prior_cnt` (0.114).

**Impact.** Counters and time-deltas that grow with elapsed time (prior counts,
tenure, the D-columns) drift by construction because the test period has more
history behind it. The target-encoded features (`te_*`) show large PSI partly by
construction as well: training rows carry out-of-fold encodings computed from the
other four-fifths of the training period, so each category takes several slightly
different values in training but a single value in validation and test. Decile
binning on the training values turns that into a large PSI even without any change
in the population. It is nonetheless a real train/serve difference, because the
model's splits were learned on the out-of-fold values. Others reflect population change. Drifting features can
make the score's calibration and the chosen thresholds stale. Score-level PSI
between validation and test is 0.009.

**Recommendation.** Before deployment, replace unbounded cumulative counters
with windowed versions, or normalise them by account age. Monitor per
MONITORING_PLAN.md.

### Failed pre-registered hypotheses

- **H4 failed.** All three orderings held in 60.7%
  of sensitivity cells, below the pre-registered 80%.
  Individually: three-way at least single cutoff 74.9%,
  profit cutoff above Youden-J 80.0%, three-way above
  rules 100.0%. The ranking claim is therefore limited
  to the orderings that do hold broadly.

**Impact / recommendation.** A failed hypothesis is reported as failed and was
not re-tested under a different metric or split. Treat the corresponding claim
as unsupported by this analysis.

## Medium

### No true timestamp

**What.** `TransactionDT` is an offset in seconds from an undisclosed reference.
**Impact.** "Per day" figures (review capacity, reviews per day, money per day)
are per relative day. The `hour_proxy` feature assumes the reference is midnight,
which is unverified; if it is not, the feature is a shifted hour and still
internally consistent. No weekday or seasonal claims are made.
**Recommendation.** Use real timestamps and the review team's actual working
calendar in production.

### Intervals cover row sampling only

**What.** All intervals come from a paired bootstrap over transactions in one
test period (30.8 relative days).
**Impact.** They do not capture variation between periods, fraud-pattern shifts,
or repeated transactions from the same account (rows are not independent).
Intervals are likely too narrow.
**Recommendation.** Rolling-origin evaluation over several periods, with a
block bootstrap by account.

### Label delay not modelled

**What.** Entity-risk encodings use all training-period labels as if known at the
end of training. In practice chargebacks arrive with a delay of weeks.
**Impact.** Validation and test are scored with encodings that a deployed system
would have only partly. The effect on the headline is likely small (encodings are
fitted on training labels only) but unmeasured.
**Recommendation.** Fit encodings on labels matured by
90 days before the scoring date.

### Review capacity is an average

**What.** The three-way policy was selected so that validation reviews per
relative day do not exceed 34.5. Nothing enforces
a per-day cap.
**Impact.** On test, mean load was 38.2 per
day, busiest day 56;
20 of 31
days exceeded capacity. Overflow handling (auto-approve, auto-decline, or backlog)
changes the realised value and is not modelled.
**Recommendation.** Add an explicit overflow rule and value it; size capacity to a
high percentile of daily load rather than the mean.

## Low

### Out-of-fold target encoding uses later training blocks

**What.** Training rows are encoded with statistics from the other time blocks of
the training period, including later ones.
**Impact.** Training rows see a mildly "future-informed" encoding that validation
and test rows do not. This is the standard out-of-fold construction; it can make training-period CV slightly optimistic. Validation and test
encodings use training labels only and are unaffected.
**Recommendation.** An expanding-window encoding (earlier blocks only) is the
stricter alternative.

### Feature fitting precedes cross-validation

**What.** V-column selection, categorical vocabularies and target encodings were
fitted on the whole training period and then used inside the expanding-window CV.
**Impact.** Early CV folds are scored with features fitted partly on later
training data, so CV PR-AUC (0.6025) may be optimistic.
CV was used only to choose among 4 configurations and
the round count, not for any reported test metric.
**Recommendation.** Refit feature steps inside each fold if CV numbers are used
for anything beyond relative ranking of configurations.

### Composite account key is a proxy

**What.** `uid` combines card attributes, billing region and implied account-open
day (DECISIONS.md D0.4).
**Impact.** Collisions merge different customers; missing `addr1` or `D1` splits
one customer into several keys. Velocity counts are noisy as a result, and the
rules baseline's "new card" flag inherits the same noise.
**Recommendation.** Use the issuer's card or account token in production.

### Review assumptions

**What.** A reviewed legitimate transaction is assumed approved at the cost of
review alone; catch rate is assumed constant regardless of queue length.
**Impact.** Real queues produce analyst false declines and customer friction from
delay, and quality falls as load rises. The three-way policy's advantage is
likely overstated by an unknown amount.
**Recommendation.** Measure analyst outcomes and model catch rate as a function of
load.

## Deviations from the pre-registration

See the list below. Each deviation is also recorded in DECISIONS.md.

- **D-1 Data source (severity: Low for results, Medium for licensing).**
  PRE_REGISTRATION.md specifies downloading the competition files through the
  Kaggle competition API. That download was refused because the competition
  rules had not been accepted on the account used. The labelled files were instead
  taken from the public Kaggle dataset `lnasiri007/ieeecis-fraud-detection`, a third-party
  re-upload of the same competition files.
  *Integrity:* each file matched the byte size on the official competition
  listing exactly (`train_transaction.csv`: 683,351,067 bytes;
  `train_identity.csv`: 26,529,680 bytes), and the
  merged table has 590,540 rows, the expected count. SHA-256 hashes are
  recorded in `outputs/results.json` under `data.provenance`.
  *Impact:* none expected on any result, since the files match the official release in byte size
  and row count; a content-level comparison was not possible without the official
  download. The mirror is not an official
  distribution; the competition's data-use terms still apply to the data.
  *Recommendation:* anyone reproducing this should accept the competition rules and
  set `kaggle.source: competition` in `config/pipeline.yaml`; the same size checks
  then apply to the official download.
- No other deviations. The split rule, metrics, baselines, cost ranges, success rule
  and seed were used as pre-registered.
