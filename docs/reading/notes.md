# Reading notes — one paragraph per paper, committed as read

> Provenance: agent-authored working notes (2026-09-09), committed at the maintainer's request.
> They compress what each paper claims and what this project adopts from it: they are scaffolding
> for interpretation, not a substitute for reading the papers. The gate stays honest: skim
> these notes, then at least skim each paper before interpreting the results.
> Order and links: [README.md](README.md).

## M8 — The Sharpe Ratio (Sharpe 1994)

Sharpe's own 1994 restatement: SR is the mean **differential** return D_t = R_F − R_B
(benchmark = risk-free or any reference portfolio) divided by the standard deviation **of
the differential**, not of total returns. Three load-bearing points for the lab: (1) the
ratio is only comparable when computed over the same measurement interval, and annualized
figures must annualize BOTH numerator and denominator consistently (returns compound, vol
scales with √t; sloppy annualization is the most common SR error in practice); (2) rf
treatment is a decision to make once and keep fixed; (3) SR says nothing about skew, which
motivates everything in M5/M6. Pinned here: rf = collateral T-bill on fully-collateralized
futures accounting (Gorton-Rouwenhorst style), daily excess returns, annualized as
mean×252 / (σ_daily×√252), applied identically to every method before any comparison.

## M4 — Facts & Fantasies / Strategic & Tactical Value (Gorton-Rouwenhorst 2006; Erb-Harvey 2006)

The pair that grounds this project's return mechanics. G&R: fully-collateralized EW futures
(1959–2004, 36 commodities, nearest contract rolled the last business day before expiry)
earned 9.98%/yr geometric with equity-like Sharpe (0.43 vs 0.38), POSITIVE skew vs stocks'
negative, negative stock/bond correlations, and futures payoff = risk premium + unexpected
spot deviation (the decomposition that matters): expected spot declines never enter the
futures return, and "rolling earns zero" when done at expiry-eve. Their basis sort
(High−Low basis: 10.04%/yr, t=5.15) is the ancestor of the carry factor. Erb-Harvey: the
average individual commodity excess return is ~zero; portfolio returns come from the
diversification return (½σ²(1−ρ)(1−1/K); this project's rebalanced 16-root baseline earns this
with zero alpha, so it must be accounted before crediting any signal) plus cross-sectional
roll return (91.6% R² of long-run cross-sectional returns explained by roll yield; but
spot vol dominates short-run variance and roll-return persistence is assumed, not proven).
Consequences for this project: collateralized accounting everywhere, never benchmark vs naive spot
indices, and treat basis as a cross-sectional tilt with long-run (not monthly) power.

## M1 — Time Series Momentum (Moskowitz-Ooi-Pedersen 2012)

Signal = **sign of the instrument's own past 12-month excess return** (not cross-sectional
rank), position 40%/σ_t−1 per instrument with EWMA(60-day) vol estimated at t−1 (explicit
no-look-ahead), monthly holding, equal-weight across 58 futures 1965–2009. Headline:
alpha 1.58%/mo (t=7.99) vs factor models, gross SR >1, all 58 instruments positive SR,
robust across lookbacks/holdings ≤12m, decaying at 24–48m. Two findings this project must internalize:
TSMOM nets to a vol-managed buy-and-hold (the exact baseline-vs-trend comparison this project runs),
and its payoff is straddle-like: best in extreme up AND down markets, worst in sharp
reversals (they lose Mar–May 2009). Caveats: all headline numbers gross of costs (2 bps/side
will bite the 1m-holding variants), no position caps, and the sample is a two-bull secular run.
this project's TSMOM benchmark = MOP spec mapped to the 16 roots; its 2010–2024Q1 window is the
honest out-of-sample question the paper never answered.

## M2 — A Century of Evidence on Trend-Following (Hurst-Ooi-Pedersen 2017)

The copyable spec: equal-weight 1M/3M/12M lookback blend, sign of past excess return,
per-market vol sizing, portfolio scaled to **10% ex-ante annualized vol** via 3-year rolling
equal-weight covariance, monthly rebalance, 67 markets 1880–2016, positive net-of-cost
return in all 14 decades (net-of-fees Sharpe 0.76). Two things worth more than the headline:
their cost ladder (one-way notional: commodities 10 bps 2003–16, ×6 pre-1992; this project's 2 bps
is deliberately milder), and the cold shower of 2010–2016 (Sharpe 0.41 net; post-2009
trend weakness overlaps this project's sample; a weak TSMOM benchmark there is a regime fact, not
an implementation failure). Also: a 1-month signal lag kills the 1M sleeve (1.38→0.45)
but not the 12M (1.32→1.04); signal-to-execution mechanics belong in this project's engine from
day one; and the performance driver worth logging is average pairwise correlation regime.

## M3 — Carry (Koijen-Moskowitz-Pedersen-Vrugt 2018)

Carry = expected return of holding the position unchanged: for futures, the term-structure
slope, C_t = (F1−F2)/(F2·ΔT) per month for commodities (no spot data needed: directly
computable from this project's 16 roots), and it is observable ex ante with no pricing model.
Long-high/short-low carry earns Sharpe 0.62–0.93 within each asset class (1.49 diversified,
2012 working-paper numbers), with ~zero passive beta and near-zero cross-class carry
correlations; predictability decays over ~1 year. Three implementation notes: rank-weight
(ranks beat magnitudes for commodities), use **carry1-12** (12m mean signal) as the
robustness variant because current carry is seasonal, and treat extreme carry spreads as
distress indicators, not richer signals. Caveat on the artifact: our PDF is the July 2012
preliminary version. Re-verify headline numbers against the published JFE version before
quoting them in the memo. For this project it doubles as the roll-selection mechanics reference:
carry IS the slope, and the legs data gives it directly.

## M5 — The Deflated Sharpe Ratio (Bailey & López de Prado 2014)

Any backtest SR is inflated twice: by picking the max of N trials and by non-normal
returns. The fix is a PSR whose rejection threshold is the **expected maximum Sharpe under
zero skill**: SR0 = √V[{SR_n}]·((1−γ)·Z⁻¹(1−1/N) + γ·Z⁻¹(1−1/(Ne))), γ = 0.5772…, with
N = number of (effectively independent) trials; then
DSR = Φ((SR̂−SR0)·√(T−1) / √(1 − γ₃·SR̂ + ((γ₄−1)/4)·SR̂²)); skew γ₃ and kurtosis γ₄ move
it both ways. The paper's own example: annualized SR 2.5 with N=100, V=0.5, skew −3,
kurtosis 10 → DSR≈0.90 and you'd need N≤46 for 95% confidence. Consequences for this project: the
trial log is not optional: every attempted configuration (including abandoned hyperparameter
searches) must be logged by the harness, because N is the crux and self-reporting is where
people cheat; and the 16-root × parameter-grid trials are heavily correlated → use the
paper's effective-N (constant-ρ replacement), watching for the degenerate case M > T.
Known limits: trials' SRs assumed iid normal, and DSR corrects only SR: drawdowns and
costs enter only through whatever SR you feed it.

## M6 — Probability of Backtest Overfitting (Bailey-Borwein-López de Prado-Zhu 2017)

CSCV: stack the T×N matrix of per-config return streams, slice time into S=16 equal
sub-periods (kept in time order), enumerate all C(16,8)=12,870 train/test half-splits; on
each, take the IS-optimal config n*, compute its OOS relative rank Ω and logit
λ = ln[(1−Ω)/Ω]; **PBO = frequency of λ ≤ 0**, the probability that the config you picked
in-sample ranks below-median out-of-sample. Their examples sting: an 8,800-config search
with IS SR 1.27 and PSR p<1% still has PBO 55%. Companion diagnostics (use them): the
performance-degradation plot (OOS vs IS Sharpe pairs: negative slope means overfit; prefer
the flat region over the max-IS config) and OOS probability of loss. Usage rules: PBO is an
audit statistic, never an objective (optimizing to minimize it re-overfits); PBO > 0.05
rejects; and it needs full trial disclosure, else it is biased low. Limits: it measures
selection reliability, not skill (a flat all-good landscape can show high PBO), and each
split uses half the sample, so with 14 years, lookbacks above ~12 months get shredded at
S=16, so the project may need S=8 for the long-lookback sleeves (state the choice, don't tune it).

## M7 — AFML ch. 7, 11–12 (López de Prado 2018)

**Status: partially gated.** This is a book (Wiley, ISBN 978-1119482086); the chapters
must be read from the purchased copy; this note is built from M6 (whose CSCV is the paper
form of ch. 11–12) and public material on ch. 7's purged/embargoed CV. What ch. 7 adds over
M6 for this project's ML variants: k-fold CV on overlapping-label series leaks by construction, so
(1) **purge** training observations whose label windows overlap the test set's label
windows (a label like "12m forward return" has an 11-month leakage tail), and (2) **embargo**
a further buffer after the test block (the paper suggests ~1% of the sample) because
serial correlation bleeds across the boundary anyway. Combined with walk-forward refits
inside develop+validate, this is the exact algorithm the ML variants implement, but the
book's worked examples and the CPCV variants (ch. 12) are not yet owned. Before the memo's
ML section is written: read the purchased chapters and extend this note.

## S5 — Momentum Crashes (Daniel & Moskowitz 2016, JFE 120(2), 221–254)

> **Scaffolding, not a read.** Agent-drafted 2026-09-11 under the same provenance as the notes
> above, so the read can be a skim-and-confirm. The gate stays open until that read
> happens, and the memo may not cite this note as read before then.

What it claims: momentum's payoff is crash-prone rather than symmetric: long stretches of small
gains punctuated by rare, severe drawdowns. The crashes cluster in **rebounds after bear markets
when volatility is high**, and the mechanism named is the short leg: the losers being shorted are
exactly the high-beta, distressed names that rally hardest when the market snaps back, so a static
long-short momentum book is short a call on the recovery. The remedy the paper quantifies is
**dynamic, vol-scaled exposure**, scaling the momentum book by the ratio of forecast momentum
return to its volatility roughly doubles risk-adjusted performance against the static book, at the
cost of lower average exposure.

Pinned here: the paper's crash mechanism is developed on equity cross-sectional momentum, so it
transfers to this project's futures XS family only as (a) the general claim that momentum
payoffs are regime-dependent and crash-prone, which is the framing this project needs for the
2022–2024Q1 out-of-sample window, and (b) the confirmation that vol-scaled exposure is the
documented mitigation, which this project's shared overlay already applies to every family. Use
it as risk framing in the memo and the brief; do not read it as an implementation instruction,
and do not claim it validates the futures cross-section.

## S9 — The Risk in Hedge Fund Strategies: Theory and Evidence from Trend Followers (Fung & Hsieh 2001, RFS 14(2), 313–341)

> **Scaffolding, not a read.** Same provenance and gate as S5 above.

What it claims: trend-following returns replicate a portfolio of **lookback straddles** on currency,
bond and commodity markets: the primitive strategies that pay off in large moves in either
direction. That convexity explains the industry's observed profile: trend followers earn their
returns in extreme markets (crises included) and bleed in range-bound, low-volatility ones, and the
payoff is not explained by standard equity/bond factors.

Pinned here: the trend family's payoff shape is a documented property of the style, not
something this project discovered, which is exactly the industry lineage the memo's buckets
(trend / short-term / carry / multi-style) and the self-test brief need, and the reason a
sideways market is the benchmark's documented weak regime rather than an implementation
failure. It also disciplines the claim this project makes when the trend benchmark loses money
in develop+validate: what the paper establishes is the payoff shape, not a promise about any
particular window.
