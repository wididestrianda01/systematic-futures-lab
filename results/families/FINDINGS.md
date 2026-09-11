# the families read findings

Window: develop+validate (2010-01-01 → 2021-12-31), frozen panel, 10% vol target, 16 CME roots; OOT (2022+) untouched.

## Seasonality standalone loses (expected finding)

Standalone seasonality: Sharpe -0.26 before costs, -0.31 after 2 bps (turnover 0.35), against the TSMOM benchmark's 0.74 / 0.51 (turnover 1.24). Confirmed, and the failure is edge rather than churn: the signal is already negative before costs, and its turnover is *below* the benchmark's — so cost drag alone does not explain it. Reported as a failed challenger.

Full per-method, per-bps metrics in `tables.csv` (Sharpe, Sortino, max DD, turnover, DSR). The cross-sectional family is the other side of this story: it posts Sharpe 0.60 against the benchmark's 0.51 after 2 bps.

## Carry: the sign convention decides (failed challenger under the literature sign)

The KMPV convention (long backwardated / short contangoed, i.e. carry = front−next over next) posts Sharpe -0.73 at 2 bps on develop+validate, while the inverted sign — long contangoed — posts 0.41. A sign flip of that size is not a measurement artifact to average away: on this universe and window the literature sign loses and its inverse is positive, close to the benchmark. Keep the literature convention in the tables; the reading-canon interpretation pass (gated, the out-of-sample read) owns what the flip means (M3 carry, M4 commodity term structure).
