# Decision log

Choices made without asking, with the reasoning. Newest decisions are appended
at the end of each section. Where a decision interacts with the
pre-registration, that is stated.

## Phase 0: scope and pre-registration

**D0.1 Use the directory the session was started in.** The setup note suggested
creating `fraud-decision-engine/`; the repo lives in the existing working
directory instead. Nothing depends on the directory name.

**D0.2 Validation split selects every threshold; test is report-only.** The
brief asks for the profit-optimal and Youden-J cutoffs. Choosing them on test
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
