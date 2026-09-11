# the families read findings

Window: develop+validate (2010-01-01 → 2021-12-31), frozen panel, 10% vol target, 16 CME roots; OOT (2022+) untouched.

## Seasonality standalone dies after costs (expected finding)

At the 2 bps default, standalone seasonality posts Sharpe -0.24 (turnover 0.24) against the TSMOM benchmark's 0.51 (turnover 0.37). Confirmed: the seasonal flip is a high-turnover, low-edge strategy and it loses to the trend benchmark after costs — reported as a failed challenger.

Full per-method, per-bps metrics in `tables.csv` (Sharpe, Sortino, max DD, turnover, DSR).

## Carry is weak on this window (failed challenger candidate)

The KMPV convention (long backwardated / short contangoed, i.e. carry = front−next over next) posts Sharpe -0.14 at 2 bps on develop+validate. The inverted convention loses less (-0.06) but also fails — carry is weak on this universe/window under either sign, not a convention artifact. Keep the literature convention; the reading-canon interpretation pass (gated, the out-of-sample read) owns the explanation.
