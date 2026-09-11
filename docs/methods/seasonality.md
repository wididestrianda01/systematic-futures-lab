# Seasonality — standalone signal and trend tilt

**Source.** S4 (Baltas & Kosowski 2013) for commodity seasonality as a documented effect; the family
was added to the comparison set at ticket authoring (tickets 24–25) with the explicit expectation
that the standalone version dies after costs — that failure is a required finding of the spec.

**Rules.** Two variants share one score.

- `seasonality` (standalone): sign of the same-calendar-month score — long a historically strong
  month, short a weak one, flat elsewhere.
- `seasonal_tilt`: the same score used as a multiplier on the TSMOM ensemble, `1 ± 0.5`, leaving the
  trend position untouched where the score is still warming up.

**Construction.** `methods.seasonality.month_score` is the mean daily return in the same calendar
month over a trailing 60-month window, using **strictly prior years'** same-months only (the current,
incomplete month never enters), requiring at least three priors; the monthly score is then mapped onto
daily dates. Daily returns and the monthly mean are measured over each symbol's own observations, so a
quote hole does not become a zero-return day.

**Evidence** (frozen tables; 2 bps is the headline level):

| variant | window | Sharpe 0 bps | Sharpe 2 bps | DSR | turnover |
|---|---|---|---|---|---|
| `seasonality` | develop+validate | -0.027 | -0.088 | 0.3683 | 0.400 |
| `seasonality` | out-of-sample | 0.828 | 0.770 | 0.8793 | 0.556 |
| `seasonal_tilt` | develop+validate | 0.163 | -0.116 | 0.3284 | 1.537 |
| `seasonal_tilt` | out-of-sample | 0.644 | 0.407 | 0.7304 | 1.836 |

**Failure modes (documented, mechanical).** In develop+validate the standalone signal is negative
*before* costs (-0.027) while its turnover is *below* the trend benchmark's — so churn is not what
makes that window negative. The score's statistical base is thin by construction: a 60-month window
holds only five priors per calendar month, so the estimate is noisy and its sign can flip between
adjacent regimes (it does, out of time). The tilt inherits all of trend's turnover and adds the
score's noise on top; it never reduces the benchmark's exposure, only amplifies or dampens it.

**Monitoring.** Rolling Sharpe of each variant separately; score coverage (share of roots and days
with a score, which grows over the sample); the tilt-versus-trend return gap (a widening gap means the
tilt is doing work — in whichever direction); and turnover, since the tilt is the higher-turnover
variant.

**Open questions.** Whether the 0.5 tilt multiplier should be pre-registered differently — it was
fixed at authoring, not tuned, so the committed numbers are not a search result. Whether the
standalone variant's failure should be interpreted as evidence against commodity seasonality or as
evidence about this universe and window is a reading the memo owes (S4, read; the interpretation is
not written here).
