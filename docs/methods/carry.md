# Carry — term-structure slope from raw front/deferred prices

**Source.** M3 (Koijen, Moskowitz, Pedersen & Vrugt 2018) — carry as the observable expected return
of holding a position unchanged, which for futures is the term-structure slope.

**Rule.** `basis = close(next) / close(front) - 1` on **raw** closes; signal `sign(-basis)` — long a
backwardated front, short a contangoed one. Direction only; the overlay owns sizing.

**Construction.** The family consumes the data layer's basis panel (`data.basis.compute_basis`, built from
the raw leg data, never from the adjusted continuous series), bound into the seam-compatible callable
so `run(method, closes)` needs no special case. Days missing either leg carry NaN basis and stay
flat — never zero-filled.

**Evidence** (frozen tables; 2 bps is the headline level):

| window | Sharpe 0 bps | Sharpe 2 bps | DSR | turnover |
|---|---|---|---|---|
| develop+validate | -3.214 | -3.614 | 0.0000 | 0.176 |
| out-of-sample | -1.513 | -1.756 | 0.0041 | 0.165 |

**Failure modes (documented, mechanical).** Under the literature's sign convention this is the worst
family in both windows **and** the highest-turnover classic (0.176/day), so cost amplifies an already
negative gross signal rather than causing it. M3 names two hazards that apply directly: the basis
series is seasonal, so a current-slope signal is noisy (M3's own robustness variant is a 12-month mean
carry), and extreme basis spreads are distress indicators rather than richer signals. The third, and
the one that dominates, is frequency: the same signal held from each month's first session earns
**+0.163** in develop+validate where the daily refresh earns -3.614, and inverting the daily rule
earns +2.807 — `results/out_of_sample/carry_frequency.csv` (built by
`scripts/carry_frequency_diagnostic.py`) is the committed evidence. No term-structure premium has a
Sharpe of 3.6 in either direction, so the committed row measures the churn of a basis-derived sign,
not carry.

**Open question — the largest one in the table.** Whether the sign convention is right for this
universe and window. The committed tables keep the literature sign; the inverted sign is *not*
adopted on the strength of its backtest, because choosing the profitable direction after seeing the
result is precisely the selection the Deflated Sharpe Ratio exists to catch. The frequency diagnostic
above explains why the number is so large without settling the premium, and M3's published
construction — slope magnitude, rank-weighted, monthly — is untested here. Resolving the premium is
the reading pass's job (M3, M4), not the backtest's.

**Monitoring.** Rolling Sharpe; the share of roots in backwardation (a regime statistic, and the
family's exposure); basis-rank dispersion; turnover; and the gap between current-slope and 12-month-mean
carry, which is the seasonal-noise check M3 recommends. Any change to the basis definition or the
sign convention is a material change under RTS 6 framing and requires the full protocol again.
