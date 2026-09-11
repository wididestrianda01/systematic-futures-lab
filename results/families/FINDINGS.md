# the families read findings

Window: develop+validate (2010-01-01 → 2021-12-31), frozen panel, 10% vol target, 16 CME roots; OOT (2022+) untouched.

**Interpretation status: gated.** The MUST reading canon gates interpretation of any
backtest result (`docs/reading/README.md`), and M7 is still partially gated
(`docs/reading/notes.md`). Everything below the numbers is a *provisional* reading of
mechanical output, not a checked finding, until the canon closes (ticket 32); the
the out-of-sample read interpretation pass owns it.

## Seasonality standalone loses (expected finding)

Numbers: standalone seasonality Sharpe -0.03 before costs, -0.09 after 2 bps (turnover 0.02), against the TSMOM benchmark's 0.20 / -0.08 (turnover 0.10). Reported as a failed challenger.

Provisional reading: the failure is edge rather than churn — the signal is already negative before costs and its turnover is *below* the benchmark's, so cost drag alone does not explain it.

Full per-method, per-bps metrics in `tables.csv` (Sharpe, Sortino, max DD, turnover, DSR). The cross-sectional family is the other side of this story: it posts Sharpe 0.41 against the benchmark's -0.08 after 2 bps.

## Carry: the sign convention decides (failed challenger under the literature sign)

Numbers: the KMPV convention (long backwardated / short contangoed, i.e. carry = front−next over next) posts Sharpe -3.61 at 2 bps on develop+validate, while the inverted sign — long contangoed — posts 2.81, against the benchmark's -0.08. The literature convention stays in the tables.

Provisional reading: a sign flip of that size is not a measurement artifact to average away, and what it means is M3 (carry) / M4 (commodity term structure) territory — so it waits for the canon rather than being resolved by picking the profitable sign.
