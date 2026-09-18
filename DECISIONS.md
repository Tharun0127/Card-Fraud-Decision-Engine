# Decision log

Design choices, with the reasoning behind each. Newest decisions are appended
at the end of each section. Where a decision interacts with the
pre-registration, that is stated.

## Phase 0: scope and pre-registration

**D0.2 Validation split selects every threshold; test is report-only.** The
project compares the profit-optimal and Youden-J cutoffs. Choosing them on test
would report an in-sample optimum as if it were achievable. Thresholds are
therefore picked on validation and evaluated on test. Test-optimal ("oracle")
values are reported only as a reference for how much is lost to threshold
estimation error.

**D0.3 Review outcome for legitimate cases.** A reviewed legitimate transaction
is assumed approved after review (costs only `review_cost`). Modelling analyst
false declines would add a parameter with no data to set it.

**D0.4 Composite card key (fixed before feature code, per PRE_REGISTRATION §4).**
Two identities are used:

- `card1`: the issuer-side card identifier as given. Coarse: many real cards
  share a `card1` value.
- `uid` = `card1 | card2 | card3 | card4 | card5 | card6 | addr1 | D1n`, with
  `D1n = floor(TransactionDT / 86400) - D1`. `card1..card6` describe the card
  (issuer, bank, type, category), `addr1` is billing region, and `D1` is
  "days since the card/account relationship began", so `D1n` is an implied
  account-open day that stays constant for one account across transactions.
  Together they approximate one cardholder account. This construction is
  widely used on this dataset and is the most defensible proxy without true
  card numbers. It is still a proxy: collisions (two accounts sharing a key)
  and splits (one account whose `addr1` or `D1` is missing on some rows) both
  occur. Rows with missing `D1` get `D1n` = missing, which is kept as its own
  token rather than dropped.
- `card_key` = `card1 | card2 | card3 | card4 | card5 | card6` (no address or
  account-open day) is used for the reuse / linkage family, where the question
  is "how many distinct addresses or emails has this card been seen with", so
  the address cannot be part of the key.

**D0.5 Seed.** `20240917`, recorded in `config/pipeline.yaml` and
PRE_REGISTRATION §7.

## Phase 1: data

**D1.1 Typed load via pyarrow.** V/C/D/id numeric columns are read as float32,
the amount as float64 (cents exact), keys as integers. The merged frame then fits
comfortably in memory on a 16 GB machine. The identity join is `left`,
`validate="one_to_one"`, with an assertion that the row count is unchanged.

**D1.2 Tie handling at split boundaries.** Row-count boundaries are moved forward
past any tied `TransactionDT`, so the strict inequality
`max(train.DT) < min(test.DT)` holds by construction and is also asserted.

**D1.3 Synthetic dataset for tests.** `src/data/synthetic.py` generates a small
dataset with the IEEE-CIS schema. It exists so that the determinism, leakage and
end-to-end tests run in seconds without Kaggle access. Synthetic runs write under
`data/synthetic/` and never touch real outputs. No reported number comes from it.

## Phase 2: features

**D2.1 Strict windows.** Velocity windows are `[t - w, t)`. Transactions in the
same second as the current one are excluded (they are not "prior"). This slightly
undercounts genuine same-second bursts and cannot leak.

**D2.2 Linkage counts include the current row's own value.** "Distinct cards on
this device" counts cards seen strictly earlier plus the current card, since the
current card is known at authorisation. Later rows never contribute.

**D2.3 Target encoding.** Pseudo-count 50 toward the fold prior. Out-of-fold over
5 contiguous time blocks of the training period (standard out-of-fold scheme); the
stricter expanding-window alternative is noted in FINDINGS.md. Missing is its own
category. Unseen categories get the training prior.

**D2.4 V-column pruning.** The 339 V columns come in groups that share a NaN
pattern. Within each group, a column is dropped if |Pearson r| > 0.90 with an
already-kept column (on a 150,000-row training sample); survivors are ranked by
univariate |AUC - 0.5| on training rows and the top 35 kept. The cap is set so the
total stays at or below 150. Every dropped column is listed with its reason in
`outputs/tables/dropped_columns.csv`.

**D2.5 Missingness filter.** Raw columns with more than 95% missing in the
training period are dropped. Identity columns not in the explicit list are
dropped as near-duplicates or free text.

**D2.6 Categoricals.** No one-hot encoding. String columns become pandas
categoricals over a training-period vocabulary; categories with fewer than 20
training rows, and unseen categories, map to `__other__`. LightGBM handles them
natively. `card1` and `addr1` are numeric-coded identifiers and are kept numeric
(LightGBM splits them as ordered integers), plus target-encoded where listed.

**D2.7 Excluded from features.** `TransactionID`, raw `TransactionDT` (its value is
position in time and would not generalise), `D1n` (quasi-identifier), and the
string keys. `hour_proxy = floor(DT / 3600) mod 24` is kept with the caveat that
the reference may not be midnight.

## Phase 3: model

**D3.1 Final model trained on train only.** Not refitted on train + validation,
because validation is where thresholds are chosen; refitting would change the
score distribution the thresholds were chosen on.

**D3.2 Round count.** The mean early-stopping round across the five
expanding-window folds for the chosen configuration. The final fit uses no
early stopping (there is no untouched data left to stop on).

**D3.3 Modest grid.** `num_leaves` {63, 255} x `min_child_samples` {50, 200},
feature fraction 0.5, learning rate 0.05: 4 configurations x 5 folds. Tuning is
ranked by mean fold PR-AUC.

**D3.4 No-velocity model.** Same hyper-parameters as the full model (to isolate the
feature family), with its own round count from the same CV procedure.

**D3.5 Model archive.** JSON document wrapping LightGBM's native model string
(`model_to_string`), plus feature names, categorical list, parameters and round
count. Reloads with `lgb.Booster(model_str=...)`; a test checks predictions are
bit-identical after the round trip. No pickle anywhere.

**D3.6 Comparisons against the rules baseline.** A binary rule has no ranking
curve, so the comparison is (a) net benefit of the decisions, and (b) precision /
recall of the model's top-k at the rule's own flag volume k. Precision@k bootstrap
intervals hold the flagged set fixed and resample rows (stated in REPORT.md).

## Phase 4: decision layer

**D4.1 Threshold grid.** 2,000 uniform quantiles of the validation scores, 2,000
upper-tail-dense quantiles, a 1,001-point even grid on [0, 1], and +inf ("decline
nothing"). The two-threshold search uses a 400-point grid dense in the upper tail.
Costs are computed from prefix sums, so the grid is exhaustive, not sampled.

**D4.2 Tie-breaks.** Profit-optimal single threshold: among equal net benefits,
the highest threshold (fewest declines). Three-way: fewest reviews, then highest
decline threshold. Youden-J: among equal J, the lowest threshold. All deterministic.

**D4.3 Review capacity.** 1% of mean daily training-period volume, in relative
days. Enforced as an average over the validation period during selection; test
reports the realised daily distribution and the number of days over capacity.

**D4.4 Net benefit per 1,000.** Net benefit divided by transactions in the
evaluated split, times 1,000.

**D4.5 Swap set.** Compared between the profit-optimal and Youden-J single
cutoffs, which isolates the effect of the cutoff choice. Because both threshold one
score, one direction is empty by construction; that is stated, not hidden.

## Phase 5: sensitivity

**D5.1 Grid.** 5 margin rates x 5 churn penalties x 5 review costs x 6 catch
rates = 750 cells, all within the pre-registered ranges. At each cell the
thresholds are re-selected on validation and valued on test. Capacity is varied
separately at the default costs.

## Phase 6 / 7: monitoring and packaging

**D6.1 PSI.** Reference = training period, comparison = test period, 10 bins on
training deciles plus a missing bin; categoricals binned by training category plus
an unseen bin; floor 1e-4 per bin.

**D7.1 Documents are rendered.** README.md, REPORT.md, FINDINGS.md and
MONITORING_PLAN.md are rendered from `docs_templates/` with Jinja2 (added to
requirements for this). Numbers enter only through filters reading
`results.json`; `src/reporting/check_numbers.py` fails the build on any numeric
token not traceable to it. Monitoring trigger values live in config for the same
reason.

**D7.2 Outputs are committed, data is not.** `outputs/` (results.json, figures,
tables) and the JSON model archives are committed as deliverables. They contain
aggregates and model parameters, not row-level data.

**D7.3 Makefile on Windows.** The Makefile uses the venv interpreter path for the
host OS. On Windows without `make`, run the same `python -m src.run_all` commands
shown in each target.
