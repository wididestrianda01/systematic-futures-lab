# Cross-sectional momentum — 12-month winners against losers

**Source.** S3 (Asness, Moskowitz & Pedersen 2013) — value and momentum everywhere; the futures
cross-section as the venue.

**Rule.** Each day, rank the roots by trailing 252-session return and center the ranks so the
strongest sits at +1, the weakest at -1 and the median at 0. Direction only; the overlay owns sizing.
The family is dollar-neutral by construction, so its return is relative, not directional.

**Construction.** `methods.xs_momentum` calls `methods.ranking.centered_rank` — the same
implementation the ML feature panel uses for its momentum rank and the ML signal map uses to center
predictions. One cross-sectional language across the lab, or the comparison stops being
apples-to-apples. Ties average their ranks; a symbol without a full trailing window is excluded from
that day's ranking rather than ranked on a partial history.

**Evidence** (frozen tables; 2 bps is the headline level):

| window | Sharpe 0 bps | Sharpe 2 bps | DSR | turnover |
|---|---|---|---|---|
| develop+validate | 0.562 | 0.406 | 0.9397 | 0.789 |
| out-of-sample | 1.294 | 1.174 | 0.9611 | 0.737 |

**Failure modes (documented, mechanical).** With 16 roots a full long-short tilt spreads across
ranks, so the book is thin: a single root's idiosyncratic move can dominate a day's return. The
signal is computed from price alone and carries no risk adjustment, so it is exposed to momentum
crashes — S5 is the paper for that framing and is not yet read, so no claim is made here. Turnover is
roughly half the trend benchmark's, which is why its cost drag is smaller in both windows.

**Monitoring.** Rolling Sharpe and rolling turnover; cross-sectional rank dispersion (a flattening
cross-section means the tilt is spreading thin); the share of the book in the extreme ranks; and
average pairwise correlation, which is what makes a crowded cross-section crash.

**Open questions.** Whether rank weighting or magnitude weighting suits a 16-root commodity-heavy
universe (M3's notes argue ranks for commodities) is unresolved and was fixed at authoring rather
than tuned.
