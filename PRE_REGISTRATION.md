# Pre-registration

Status: **frozen**. Committed before any modelling code was written and before any
data was inspected beyond the published facts about the IEEE-CIS competition
(590,540 labelled transactions, ~3.5% positive). This file is not edited after
the Phase 0 commit. Anything that reality forces us to change is logged as a
deviation in `FINDINGS.md`, with the reason, and the pre-registered version is
still reported alongside it.

## 1. Population and unit of analysis

- Source: IEEE-CIS Fraud Detection, `train_transaction.csv` left-joined to
  `train_identity.csv` on `TransactionID`. Only the labelled `train_*` files are
  used; the competition `test_*` files carry no labels and are not used at all.
- Unit: one card-not-present transaction. Label: `isFraud`.
- `TransactionDT` is a seconds offset from an undisclosed reference. It is used
  **only as an ordering index** and for relative durations (e.g. "prior 24h").
  No calendar interpretation (weekday, holiday, month) is claimed.

## 2. Train / validation / test split rule

- Rows are ordered by (`TransactionDT`, `TransactionID`).
- Earliest 70% of rows = **train**, next 15% = **validation**, final 15% =
  **test**. Boundaries are placed on row counts and then moved forward so that
  no `TransactionDT` value straddles two splits (ties go to the later split).
- Required invariants (enforced by tests):
  `max(train.DT) < min(validation.DT)` and `max(validation.DT) < min(test.DT)`.
- **Train** is used for feature fitting (target encoding, V-column selection),
  hyper-parameter tuning (expanding-window time-series CV, 5 folds, inside
  train only) and fitting the final model.
- **Validation** is used for choosing every decision threshold (profit-optimal,
  Youden-J, three-way policy) and the rules-baseline amount percentile is fixed
  in advance (below), so validation is not used to tune it.
- **Test** is touched once, for reporting. Nothing is tuned or selected on it.
  Any "oracle" number computed with test labels (e.g. the in-sample test-optimal
  threshold) is labelled as such and never used as a headline.

## 3. Primary metrics

- **Primary decision metric:** net benefit versus an approve-everything policy,
  in dollars per 1,000 test transactions, of the capacity-constrained three-way
  (approve / review / decline) policy whose thresholds were chosen on validation.
- **Primary model metric:** PR-AUC (average precision) on test. ROC-AUC is
  reported but is not primary because at a 3.5% base rate it is insensitive to
  the region of the score distribution where decisions are made.
- Customer-impact metrics reported beside money in every headline table:
  good-customer disruption rate (legitimate transactions declined / all
  legitimate transactions), fraud capture rate by count and by dollars, and
  review queue volume (per day and in total).
- Uncertainty: paired bootstrap over test transactions, 1,000 resamples, fixed
  seed, 95% percentile intervals. Where an interval includes zero, the result is
  reported as "not distinguishable from zero", not as a point estimate alone.

## 4. Baselines (both must be reported)

1. **Rules baseline** (what a fraud analyst would write on day one):
   decline if `TransactionAmt` > the **90th percentile of training-period
   amounts** AND the card is new, where "new" means the composite card key
   (defined in `DECISIONS.md` before feature code runs) has **no earlier
   transaction** in the observed history. Everything else is approved. Fixed in
   advance; not tuned.
2. **No-velocity model**: the same LightGBM configuration and training
   procedure with the velocity feature family removed. Isolates what the
   velocity engineering adds.

Comparisons against the rules baseline use money (net benefit per 1,000), and
precision / recall at the rules baseline's own flag volume, because a binary
rule has no meaningful ranking curve. Its ROC-AUC / PR-AUC are reported for
completeness and marked as degenerate.

## 5. Cost model and pre-registered parameter ranges

Per transaction with amount `A`:

| Outcome | Cost |
|---|---|
| Fraud approved | `A` (full loss; chargeback) |
| Legitimate declined | `margin_rate * A + churn_penalty` |
| Sent to review | `review_cost`; if fraud, it is caught with probability `review_catch_rate`, otherwise approved and lost (`A`) |
| Fraud declined, legitimate approved | 0 |

Reviewed legitimate transactions are assumed approved after review (analysts
do not decline them). Net benefit = cost(approve-all) − cost(policy).

| Parameter | Default | Sensitivity range | Rationale |
|---|---|---|---|
| `margin_rate` | 0.02 | 0.005 – 0.05 | Net merchant / issuer margin on a card sale: interchange-like economics at the low end, retail gross margin contribution at the high end. |
| `churn_penalty` | $10 | $0 – $50 | Lost future value from a customer who abandons after a false decline. $0 = no churn effect; $50 = a meaningful fraction of annual value for a low-frequency online shopper. |
| `review_cost` | $5 | $1 – $20 | Analyst time per case: ~2–15 minutes at loaded labour cost, plus tooling. |
| `review_catch_rate` | 0.85 | 0.50 – 0.98 | Analysts are imperfect; 0.5 is a weak queue, 0.98 a strong one with good tooling. |
| review capacity | 1% of mean daily training-period transaction volume | 0.5% – 2% (reported, not a success criterion) | Stand-in for a fixed analyst headcount. |

These values are **assumptions**, not measurements. The project does not claim
that any absolute dollar figure is correct for a real portfolio.

## 6. Decision rule for success

Evaluated on the test split with thresholds chosen on validation.

- **H1 (primary, decision layer).** The profit-optimal single threshold yields
  higher test net benefit than the Youden-J threshold, and the paired bootstrap
  95% interval of the difference excludes zero.
- **H2 (primary, model versus rules).** The full model's constrained three-way
  policy yields higher test net benefit than the rules baseline, 95% interval
  excluding zero.
- **H3 (secondary, feature engineering).** The full model's test PR-AUC exceeds
  the no-velocity model's, 95% interval excluding zero.
- **H4 (secondary, robustness).** Across the pre-registered sensitivity grid,
  the ranking *three-way profit policy ≥ single-threshold profit policy >
  Youden-J policy* and *full model > rules baseline* holds in at least 80% of
  grid cells.

The project is declared successful if H1 and H2 both hold. H3 and H4 are
reported with the same prominence whatever their outcome. A failed hypothesis is
reported as failed; it is not re-tested with a different metric, split or
parameter until it passes.

## 7. Fixed seed

Global seed `20240917` for all randomness (bootstrap, LightGBM, fold
assignment, sampling). LightGBM is run with `deterministic=true` and a fixed
thread count.
