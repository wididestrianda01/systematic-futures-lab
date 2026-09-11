# TSMOM / trend — six-horizon vol-scaled ensemble

**Source.** M1 (Moskowitz, Ooi & Pedersen 2012) for the signal; M2 (Hurst, Ooi & Pedersen 2017) for
the multi-lookback, vol-scaled ensemble specification; S4 (Baltas & Kosowski 2013) for futures
implementation detail.

**Rule.** For each lookback in (5, 10, 20, 21, 63, 252) trading days, a sleeve is
`sign(trailing return) × 0.40 / trailing annualized vol`, clipped to ±1. The family is the
equal-weight mean of the six sleeves. This is the **decision-rule benchmark**: the ML variants must
beat it on decision-window DSR after costs or be documented as failed challengers.

**Construction.** `methods.tsmom.horizon_signal` measures both the trailing return and the trailing
vol over each symbol's own observations (`data.panels`), so a quote hole neither shifts a horizon
nor breaks a window; the signal is NaN until the window fills, and the engine treats NaN as flat
rather than manufacturing a warm-up. The 5/10/20-day sleeves (adopted at ticket 26) sit in the same
family as the literature's 1/3/12-month core.

**Evidence** (frozen tables; 2 bps is the headline level):

| window | Sharpe 0 bps | Sharpe 2 bps | DSR | turnover |
|---|---|---|---|---|
| develop+validate | 0.200 | -0.085 | 0.3726 | 1.654 |
| out-of-sample | 0.780 | 0.551 | 0.7966 | 1.914 |

**Failure modes (documented, mechanical).** Cost is the first: turnover of 1.65/day flips
develop+validate Sharpe from +0.20 gross to -0.085 at 2 bps, and the same turnover takes 0.23 of
Sharpe out of the out-of-sample window. The second is the payoff shape M1 names — the strategy is
straddle-like, strongest in sustained moves and weakest in sharp reversals, so a sideways,
frequent-reversal regime is its worst case. Third, short sleeves are execution-sensitive: M2's
one-month signal largely disappears with a one-month execution lag while the 12-month sleeve barely
moves, so the sleeves adopted here raise turnover and lower robustness per unit of edge.

**Monitoring.** Rolling 12-month Sharpe and rolling turnover; sleeve dispersion (the long 252-session
sleeve empties the effective sample on a twelve-year window); average pairwise correlation of the
book, which M2 names as the performance driver; and the cost-to-gross ratio, which is what flips this
family's sign. Any change to the horizon set, the vol scalar, the overlay parameters or the universe
is a material change under RTS 6 framing and requires a re-run, not an adjustment.

**Open questions.** The 2010–2019 weakness is the family's most interesting number and it is a
reading, owed to the memo (M1/M2, plus S5 for crash framing once read). The suite of sleeves was
adopted at ticket authoring, before these numbers, and is not re-tuned here.
