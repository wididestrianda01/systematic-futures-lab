# ML variants — 6a `ml_defaults` and 6b `ml_tuned`

**Source.** M5 (Deflated Sharpe Ratio), M6 (CSCV / probability of backtest overfitting), M7
(López de Prado, purged and embargoed cross-validation — **partially gated**: the purchased chapters
are not yet read by the owner). The implementation follows the published CV mechanism rather than a
library splitter, so the leakage argument is inspectable.

**Rule.** Pooled LightGBM regression on the point-in-time feature panel, predicting the forward
5-session volatility-scaled return. Out-of-fold predictions become the signal, centered
cross-sectionally through the same `methods.ranking.centered_rank` the XS family uses. 6a
`ml_defaults` uses sensible LightGBM defaults and searches nothing (trials = 1); 6b `ml_tuned` runs a
pruned TPE search per fold on an inner validation block that is itself purged and embargoed from that
fold's test block (trials = folds x trials = 100, counted whether or not a fold's search was later
skipped for lack of data).

**Construction.** `ml.features` builds the tidy `(date, symbol)` feature table — trailing returns over
five windows, realized vol over two windows and their ratio, distance from two moving averages, the
cross-sectional rank of 12-month momentum, the basis level and its basis rank, and the seasonality
score — every value as-of t. `ml.features.forward_label` is the one deliberately forward-looking
object and is never a model input. `ml.cv.purged_walk_forward` builds expanding-train, contiguous-test
folds, purges training dates whose label window reaches into the test block, and embargoes a further
buffer after every earlier test block; `leakage_violations` re-derives both mechanisms from the fold
semantics so the harness asserts cleanliness instead of trusting the constructor. Signals are 0 where
no fold predicts, which is why coverage is reported with the numbers.

**Evidence** (frozen tables; 2 bps is the headline level):

| variant | window | Sharpe 0 bps | Sharpe 2 bps | DSR | turnover | trials | coverage |
|---|---|---|---|---|---|---|---|
| `ml_defaults` (6a) | develop+validate | 1.406 | 0.799 | 0.9989 | 2.107 | 1 | 60.5% |
| `ml_tuned` (6b) | develop+validate | 2.074 | 1.550 | 0.9999 | 1.622 | 100 | 46.5% |
| `tsmom` (benchmark) | develop+validate | 0.200 | -0.085 | 0.3726 | 1.654 | 1 | 100% |
| `ml_defaults` (6a) | out-of-sample | 0.759 | 0.072 | 0.5433 | 3.183 | 1 | 99.1% |
| `ml_tuned` (6b) | out-of-sample | 0.798 | 0.390 | 0.0261 | 2.077 | 100 | 99.1% |
| `tsmom` (benchmark) | out-of-sample | 0.780 | 0.551 | 0.7966 | 1.914 | 1 | 100% |

The pre-declared rule (`results/decision/DECISION_RULE.md`, written before these numbers existed): a
variant wins only if its decision-window DSR after costs beats the benchmark's. Applied in
develop+validate both clear it; applied to the single out-of-sample read both fail it
(`results/out_of_sample/OOT.md`). The robustness read — each variant re-measured on its own covered dates —
agrees with the verdict on both windows, and the run asserts that agreement before either read is
reported.

**Failure modes (documented, mechanical).**

1. **Cost.** These are the highest-turnover families in the project: 6a reaches 3.18/day out of time,
   and 2 bps removes 0.69 of its 0.76 gross Sharpe there.
2. **Coverage.** Folds that cannot be fitted leave flat days, which dilute Sharpe by roughly the
   square root of coverage (46.5% for 6b in develop+validate). This is why the primary read is
   reported beside the like-for-like read and why both are committed.
3. **Search.** 6b's DSR is deflated by 100 configurations, so its decision-read margin over 6a is not free
   — the deflation is the price of having searched at all.
4. **Deliberate gap.** M6's own diagnostic — the probability of backtest overfitting via CSCV — is
   *not* computed in this project. The multiple-testing discipline rests on the trial-count deflation
   and the touched-once out-of-sample window. Recorded as a gap, not papered over.
5. **Unread plumbing.** M7's CPCV variants (ch. 12) are unimplemented, and the gate on the purchased
   chapters is open, so the ML interpretation is not written here.

**Monitoring.** Rolling Sharpe of out-of-fold predictions; coverage; rolling turnover and the
cost-to-gross ratio; feature drift (basis-rank and momentum-rank dispersion are the cheapest
sentinels); and a hard retest trigger on any change to the feature set, label horizon, fold geometry,
search space or the model class — under RTS 6 framing each of those is a material change, and the
protocol has to be re-run rather than patched.

**Open questions (the memo's, canon-gated).** Why the decision-read winners fail out of time, and what the
purged/embargoed folds and the DSR deflation actually changed versus a naive split, are exactly the
questions the M7 gate exists for (tickets 32 and 34).
