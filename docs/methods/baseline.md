# Baseline — vol-targeted buy-and-hold (the null model)

**Source.** M4: Gorton & Rouwenhorst (2006) and Erb & Harvey (2006) — the collateralized-futures
return decomposition, and the diversification return a rebalanced multi-market book earns with no
signal at all.

**Rule.** Constant full-long signal (+1) for every root. The shared overlay supplies all sizing, so
the family's return is the equal-capital, vol-targeted return of the 16-root book.

**Construction.** `methods.baseline` returns a +1 panel on the close grid, and nothing else. It
exists to be beaten: a challenger that cannot out-earn it after costs is paying complexity for
nothing.

**Evidence** (frozen tables; 2 bps is the headline level):

| window | Sharpe 0 bps | Sharpe 2 bps | DSR | turnover |
|---|---|---|---|---|
| develop+validate | 0.419 | 0.408 | 0.9403 | 0.112 |
| out-of-sample | -0.685 | -0.694 | 0.1470 | 0.143 |

**Failure modes (documented, mechanical).** Turnover is an order of magnitude below the signal
families, so cost is not what moves this family: its out-of-sample reading is the book's return, not
its trading. It carries the market exposure of a long futures book with no protection in a
commodity/rates drawdown, and its develop+validate strength is the diversification return doing the
work — mechanically, a rebalanced equal-capital book of imperfectly correlated roots earns it
(M4, read).

**Monitoring.** Rolling Sharpe of the equal-capital book; rolling realized vol against the 10%
target; the diversification contribution (mean cross-sectional dispersion and average pairwise
correlation) — thinning diversification is the first sign that this baseline's free lunch is
shrinking. A change to the universe, the overlay's vol target or lookback, or the cost level read is
a material change under RTS 6 framing and invalidates this table until re-run.

**Open questions.** Whether daily vol targeting is the right comparison point for a monthly
literature baseline, and how much of the family's develop+validate Sharpe survives a monthly
rebalance, are for the memo's reading pass.
