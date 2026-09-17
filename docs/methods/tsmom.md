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
rather than manufacturing a warm-up. The 5/10/20-day sleeves (adopted before any results existed) sit in the same
family as the literature's 1/3/12-month core.

**Evidence** (frozen tables; 2 bps is the headline level):

| window | Sharpe 0 bps | Sharpe 2 bps | DSR | turnover |
|---|---|---|---|---|
| develop+validate | 0.200 | -0.085 | 0.3726 | 0.103 |
| out-of-sample | 0.780 | 0.551 | 0.7966 | 0.120 |

**Failure modes (documented, mechanical).** Cost is the first: book turnover of 0.103/day flips
develop+validate Sharpe from +0.20 gross to -0.085 at 2 bps, and the same turnover takes 0.23 of
Sharpe out of the out-of-sample window. The second is the payoff shape M1 names: the strategy is
straddle-like, strongest in sustained moves and weakest in sharp reversals, so a sideways,
frequent-reversal regime is its worst case. Third, short sleeves are execution-sensitive: M2's
one-month signal largely disappears with a one-month execution lag while the 12-month sleeve barely
moves, so the sleeves adopted here raise turnover and lower robustness per unit of edge.

**Monitoring.** Rolling 12-month Sharpe and rolling turnover; sleeve dispersion (the long 252-session
sleeve empties the effective sample on a twelve-year window); average pairwise correlation of the
book, which M2 names as the performance driver; and the cost-to-gross ratio, which is what flips this
family's sign. Any change to the horizon set, the vol scalar, the overlay parameters or the universe
is a material change under RTS 6 framing and requires a re-run, not an adjustment.

**External check — the source's own construction, on free data.** `results/replication/` runs M1's
Eq. (5) verbatim (EWMA volatility with a 60-day centre of mass scaled by 261, sign of the trailing
twelve months, 40% volatility per instrument, monthly rebalance) on the 52 instruments the free
snapshot can supply from M1's Table 1, and `results/replication/FINDINGS.md` carries the numbers.
Three of them matter here:

- The paper's risk and Sharpe claims reproduce on free data: the diversified factor comes out at
  13.7% annualised volatility and gross Sharpe 1.20 over 1985–2009, against the paper's stated 12%
  and "greater than one".
- The 2010–2016 sub-window, the regime this family's out-of-sample read sits inside, comes out at
  0.58 gross / 0.56 net, bracketing the 0.41 *net* figure Hurst, Ooi & Pedersen publish for the same
  regime. This family's weak OOT numbers are therefore the published regime and the cost model, not
  an implementation failure.
- **The published levels are construction-specific, and this repo's construction is a different one.**
  M1 chains the most liquid contract's daily return, so the price gap at each roll enters its series;
  this repo's ratio back-adjustment makes the return the held contract's own, roll days included.
  On the commodity class the difference is worth a factor of 2.6 in the mean monthly return (0.500%/mo
  under the paper's splice, 1.317%/mo under the roll-adjusted one, published 0.59%/mo). Quoting M1's
  numbers as a level for this lab's series therefore compares two different objects; they are a
  regime and Sharpe reference, not a level target.

**Open questions.** The 2010–2019 weakness is the family's most interesting number and it is a
reading, owed to the memo (M1/M2, plus S5 for crash framing once read). The suite of sleeves was
adopted before these numbers, and is not re-tuned here.
