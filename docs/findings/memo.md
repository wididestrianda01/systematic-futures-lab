# Findings: a systematic futures method comparison under one honest protocol

**Scope.** Sixteen CME futures roots, daily contract-level data 2010 → 2024Q1, six classic method
families and two LightGBM variants, one shared vectorised engine, one volatility overlay, one cost
model, one evaluation protocol, and a single out-of-sample window read once at the end.

**Provenance of the interpretation in this memo.** The MUST reading canon is the gate for
interpreting any backtest result (`docs/reading/README.md`). At the time of writing, M1–M6 and M8 are
read and closed; M7 (AFML ch. 7, 11–12) and the two SHOULD papers S5/S9 exist as agent-drafted
scaffolding in `docs/reading/notes.md`, explicitly marked *not read*. Every claim below is traceable
to a committed table; the two places where a reading rests on the scaffolding rather than on a
completed read are flagged inline as **[scaffolded]**. The memo is written so that a reader can
strike those flags' claims without disturbing the rest.

---

## 1. Executive summary

The research cycle this lab demonstrates runs from signal generation to implementation: raw contract
quotes → observed rolls → back-adjusted continuous series → basis → signals → one shared sizing
overlay → one cost charge → metrics → a pre-declared selection rule → a single out-of-sample read.
Five conclusions survived it.

1. **Costs select strategies, they do not shade them.** Across every family the Sharpe at 10 bps is
   materially different in kind from the Sharpe at 0 bps: the trend benchmark goes from +0.20 to
   −1.20 in develop+validate, the ML variants from +1.41/+2.07 to −1.66/−0.62, while the
   low-turnover families (baseline, XS momentum) barely move. Turnover, not signal cleverness, is
   what separates these families in a 2 bps world.
2. **A null model is a real competitor.** The vol-targeted buy-and-hold book earns Sharpe 0.41 at
   2 bps in develop+validate on book turnover of 0.007 — beating the trend benchmark, the seasonal
   tilt and standalone seasonality after costs. Any signal has to beat the diversification return
   before it has demonstrated anything.
3. **The ML variants won the comparison that was declared and lost the window that was not.** 6a and
   6b beat the benchmark on the pre-declared decision rule inside develop+validate (DSR 0.9989 and
   0.9999 against 0.3726) and both fail it on the single out-of-sample read (0.5433 and 0.0261
   against 0.7966). This is the finding the protocol existed to produce, and it is reported as such
   rather than retuned away.
4. **Two classic findings are cheap and permanent.** Standalone seasonality is negative *before*
   costs with turnover well below the benchmark's — the failure is the signal, not churn. Carry, as
   implemented here, is the worst family in both windows; the diagnostic in section 7 shows that
   number belongs to rebalancing a basis-derived sign daily, not to the carry premium.
5. **What generalised out of time is what did not need to time the regime.** XS momentum is positive
   in every sub-period and both windows (0.41 / 1.17 at 2 bps); the passive book lost money
   out of time while the timing families earned it — 2022 and 2024Q1 paid the trend and momentum
   families, 2023 paid almost nothing except seasonality.

For a reviewer, the transferable craft on show is: a frozen, hash-gated dataset; one accounting path
with a test that fails on look-ahead; a trial log that feeds a multiple-testing correction; a
pre-declared rule; a robustness read that must agree with the primary read or the run stops; and a
result that went against the author and was published.

---

## 2. Question, data, protocol

**The question.** Under one identical accounting and evaluation protocol, which futures method
families survive after costs — and does a tuned ML variant earn its complexity?

**Data.** pysystemtrade's free `multiple_prices_csv` snapshot, pinned at commit `b4a25e6`, frozen
upstream at 2024-03-28. Raw per-contract prices for three legs (front, successor, prior) across 16
CME roots; rolls observed from the leg data; the continuous series built in-project by ratio
back-adjustment; the basis series computed from raw closes. No market data is redistributed: the
repository ships derived statistics only, and every derived file is SHA-256-gated by a committed
manifest.

**Windows.** Develop 2010–2019, validate 2020–2021 (COVID stress), used jointly as the decision
window for the ML rule; **out-of-sample 2022 → 2024Q1**, read exactly once.

**Protocol.** One engine (`engine.run`) for every family and every parameter search; a shared sizing
overlay that scales **each symbol** to a 10% annualized volatility target and clips it to ±1 notional;
a linear cost charge of `bps` per side on traded notional at 0/2/5/10 bps, 2 bps headline; Sharpe,
Sortino, max drawdown, turnover and the Deflated Sharpe Ratio reported per method per cost level.
Two reads of each table: the primary read on the identical window, and a like-for-like read
re-measuring each ML variant on its own covered dates, with the run asserting both return the same
verdict.

**Reading the numbers.** Turnover is book-level — mean daily traded notional per unit of capital,
which is the base the charge is taken on — so `turnover × bps × 252 × 1e-4` is the annual return a
cost level costs. Sharpe is annualized at √252 in both windows as a stated convention, not because
the panel has 252 sessions a year: the frozen data carries 309.6 sessions per year in
develop+validate (it includes Sunday bars through 2021) and 258.4 from 2022, so the develop+validate
Sharpes are ~11% higher annualized on their own session density, and the two windows' Sharpe
*levels* are not strictly comparable. The DSR the rule reads is a probability and is unaffected.
Sizing per symbol does not equalize book risk either: realized book volatility is 0.9–3.0% in
develop+validate and 1.5–4.6% out of time (baseline 4.6%, `ml_defaults` 1.5%), so the families are
apples-to-apples in rule and cost, not in delivered risk.

**Test discipline.** A regression test fails on a look-ahead implementation and passes on the correct
as-of convention (`signal(t)` drives the position held from t to t+1). The roll invariant is pinned
by test: the continuous series' daily return equals the held contract's own return on every session
including roll days. The manifest gate fails ingest on any drift. The ML fold splitter is hand-rolled
and its purge/embargo properties are re-derived from the fold semantics by `leakage_violations`
rather than trusted.

---

## 3. What was compared

| family | rule, in one line | source |
|---|---|---|
| Baseline | constant full-long, overlay-sized (the null model) | M4 |
| TSMOM | equal-weight ensemble of vol-scaled 5/10/20-day and 1/3/12-month horizon signs | M1, M2, S4 |
| XS momentum | centered cross-sectional rank of 12-month return | S3 |
| Carry | sign of the negative basis (long backwardated, short contangoed) | M3 |
| Seasonality | sign of the same-calendar-month trailing score; separately as a ±50% tilt on TSMOM | S4 |
| 6a `ml_defaults` | LightGBM defaults, purged/embargoed walk-forward, no search | M5, M6, M7 |
| 6b `ml_tuned` | same, with a pruned TPE search per fold | M5, M6, M7 |

Per-family construction, parameters, monitoring triggers and failure modes:
`docs/methods/`.

---

## 4. Results

### 4.1 Develop+validate (2010-01-01 → 2021-12-31), 2 bps/side

| family | Sharpe | DSR | turnover | notes |
|---|---|---|---|---|
| `baseline` | 0.408 | 0.9403 | 0.007 | the null model |
| `xs_momentum` | 0.406 | 0.9397 | 0.049 | |
| `tsmom` (benchmark) | -0.085 | 0.3726 | 0.103 | +0.200 gross |
| `seasonality` | -0.088 | 0.3683 | 0.025 | -0.027 gross |
| `seasonal_tilt` | -0.116 | 0.3284 | 0.096 | +0.163 gross |
| `carry` | -3.614 | 0.0000 | 0.176 | -3.214 gross |
| `ml_defaults` (6a) | 0.799 | 0.9989 | 0.132 | trials 1, coverage 60.5% |
| `ml_tuned` (6b) | 1.550 | 0.9999 | 0.101 | trials 100, coverage 46.5% |

The like-for-like read agrees: on 6a's own dates the benchmark's DSR is 0.4253, on 6b's it is
0.3651, and both variants still clear it — the verdict is not an artefact of the variants sitting
flat on days they do not cover.

### 4.2 Out-of-sample (2022-01-03 → 2024-03-28), the single read, 2 bps/side

| family | Sharpe | DSR | turnover |
|---|---|---|---|
| `xs_momentum` | 1.174 | 0.9611 | 0.046 |
| `seasonality` | 0.770 | 0.8793 | 0.035 |
| `tsmom` (benchmark) | 0.551 | 0.7966 | 0.120 |
| `seasonal_tilt` | 0.407 | 0.7304 | 0.115 |
| `ml_tuned` (6b) | 0.390 | 0.0261 | 0.130 |
| `ml_defaults` (6a) | 0.072 | 0.5433 | 0.199 |
| `baseline` | -0.694 | 0.1470 | 0.009 |
| `carry` | -1.756 | 0.0041 | 0.165 |

**The pre-declared rule, applied as written:** both ML variants fail it out of time. The decision
read's verdict stands as decided; this is evidence about it, and it is not retuned.

### 4.3 Cost sensitivity (Sharpe by cost level)

| family | 0 bps | 2 bps | 5 bps | 10 bps |
|---|---|---|---|---|
| `baseline` | 0.419 / -0.685 | 0.408 / -0.694 | 0.390 / -0.709 | 0.361 / -0.734 |
| `tsmom` | 0.200 / 0.780 | -0.085 / 0.551 | -0.508 / 0.208 | -1.203 / -0.358 |
| `xs_momentum` | 0.562 / 1.294 | 0.406 / 1.174 | 0.172 / 0.994 | -0.216 / 0.695 |
| `carry` | -3.214 / -1.513 | -3.614 / -1.756 | -4.203 / -2.119 | -5.147 / -2.718 |
| `seasonality` | -0.027 / 0.828 | -0.088 / 0.770 | -0.177 / 0.683 | -0.326 / 0.539 |
| `seasonal_tilt` | 0.163 / 0.644 | -0.116 / 0.407 | -0.530 / 0.054 | -1.208 / -0.529 |
| `ml_defaults` | 1.406 / 0.759 | 0.799 / 0.072 | -0.122 / -0.957 | -1.660 / -2.658 |
| `ml_tuned` | 2.074 / 0.798 | 1.550 / 0.390 | 0.746 / -0.220 | -0.616 / -1.232 |

(develop+validate / out-of-sample. Turnover is identical across a row by construction — only the
price per trade changes.)

Two families have a *sign-flipping* cost profile, both in develop+validate: the trend benchmark
(+0.200 gross → −0.085 at the headline) and the seasonal tilt (+0.163 → −0.116). Both ML variants
stay positive gross *and* net at the headline level in both windows — 6a 0.759 → 0.072 and 6b
0.798 → 0.390 out of time — so what cost takes from them is most of the signal, not its sign. At
5 bps the survivors differ by window: develop+validate leaves the baseline (0.390), XS momentum
(0.172) and `ml_tuned` (0.746) positive; out of time leaves XS momentum (0.994), seasonality (0.683),
the trend benchmark (0.208) and the tilt (0.054).

---

## 5. Multiple-testing discipline

**The trial log is the input, not the afterthought.** The DSR column is deflated by the number of
configurations each selection actually evaluated: 1 for every family that searched nothing, 100 for
6b (5 folds × 20 trials, counted whether or not a fold's search was later skipped for lack of data).
The count is conservative by construction — the deflation never flatters the variant.

**What was pre-registered and when.** The decision rule was written into
`results/decision/DECISION_RULE.md` before the decision read's numbers existed; the horizon-sleeve set, the
tilt multiplier, the cost ladder and the headline cost level were all fixed before any results existed.
Nothing in this memo is a search result across those choices.

**One decision surface, two reads.** The primary read is the identical-window read the rule is
applied to; the like-for-like read exists because a family flat by construction on part of the window
has its Sharpe diluted by roughly the square root of its coverage. They are not two chances to
decide: `decision.require_same_verdict` stops the run when they disagree, on both protocol runs.

**One touch.** The out-of-sample window is read once, by one script, at the end. The walk-forward
schedule is extended across the whole frozen panel so that the ML fold covering 2022 trains entirely
inside develop+validate — no fold's fit or search ever saw an out-of-sample date.

**Recorded gap.** M6's own diagnostic — the probability of backtest overfitting via CSCV — is *not*
computed anywhere in this project. The multiple-testing discipline here is the trial-count DSR
deflation plus the single out-of-sample read. Calling that a complete treatment would be wrong; it is
the treatment this project implemented, and the gap is stated rather than glossed.

---

## 6. Where the ML variants overfit, and where they did not

**In-window.** The decision rule's bar was the trend benchmark's decision-window DSR (0.3726), and the
benchmark earned a *negative* net Sharpe in that window. Beating it is therefore a low bar: the
correct statement is "the variants did not lose money where the benchmark did", not "the variants
demonstrated skill". The sub-period decomposition makes the concentration explicit:

| family | 2010–2014 | 2015–2019 | 2020–2021 |
|---|---|---|---|
| `tsmom` (benchmark) | 0.15 | -0.51 | 0.33 |
| `ml_defaults` (6a) | -0.80 | 1.50 | 1.17 |
| `ml_tuned` (6b) | no coverage | 1.97 | 2.22 |
| `xs_momentum` | 0.41 | 0.30 | 0.66 |

The variants' entire in-window advantage sits in 2015–2019 and 2020–2021 — the two periods the
benchmark lost money — while in 2010–2014 6a was -0.80 and 6b produced no coverage at all (its search
was skipped for lack of training data). A learned combination of trailing returns, realized vol,
moving-average distance, basis and seasonality scores is a non-linear re-expression of the same
premia the classic families trade; when the benchmark's regime turns, that re-expression has nothing
extra to lean on.

**Out of time.** 6a's gross Sharpe is 0.759 and its net Sharpe at 2 bps is 0.072 on book turnover of
0.199 per day — the cost charge consumes essentially the whole signal. 6b holds 0.390 net, with a DSR
of 0.0261 under its 100-trial deflation, below the benchmark's 0.7966.

**What the purged/embargoed protocol bought, and what it did not.** It bought a structural guarantee:
with a 5-session label and daily overlapping samples, a shuffled or plain k-fold split leaks by
construction, and this project's folds cannot — the property is re-derived from the fold semantics by
`leakage_violations` and asserted on every run. What it did *not* buy is generalization: purge and
embargo remove label overlap, not regime dependence. The 2010–2024 out-of-sample read is the
instrument that caught the failure, and it caught it in one shot. A deliberate gap is recorded here
too: the project did **not** run a naive-split counterfactual to measure how much the leakage would
have inflated the in-window numbers, so the size of what the protocol prevented is argued from
theory (M6, M7) rather than measured. **[scaffolded: M7]**

**The DSR's honest reading.** The Deflated Sharpe Ratio answers "how likely is a Sharpe this large
given this many trials and these return moments, under zero skill". 6b's deflation by 100
configurations was honoured and it still cleared the benchmark in-window — which is exactly the
failure mode a significance test does not prevent: a real in-sample improvement that does not
generalize. DSR is a discipline on selection bias, not a promise about the next window.

---

## 7. Two findings worth more than the losers

**Standalone seasonality fails on the signal, not on cost.** It is negative *before* costs in
develop+validate (-0.027) with book turnover of 0.025 — a quarter of the benchmark's 0.103 — so cost
drag is not the explanation. The score's statistical base is thin by construction: a 60-month window
holds five priors per calendar month, so the estimate is noisy and its sign is not stable across
windows (out of time the same family earns +0.828 gross). Its intended use in this project was always
the tilt on trend, and the tilt inherits trend's turnover and hence trend's cost problem. Reading: a
documented failed challenger, reported as the plan required, and not evidence against commodity
seasonality as such (S4).

**Carry's result belongs to the rebalancing frequency, not to the carry premium.** The published
construction (M3) is a slope *magnitude* — `(F1 − F2) / (F2 · ΔT)` per month — rank-weighted,
rebalanced monthly, with a 12-month-mean variant recommended precisely because current-slope carry is
seasonal. What this project ran is the *sign* of the current basis refreshed every session. That sign
moves with the front leg's own price as much as with the term structure, so at daily frequency the
family trades the basis's own noise — and the diagnostic in `results/out_of_sample/carry_frequency.csv`
measures exactly that, through the same seam, on the same windows:

| construction | dev+validate | out-of-sample | turnover (dev) |
|---|---|---|---|
| daily `sign(-basis)` — the committed row | -3.614 | -1.756 | 0.176 |
| the same rule, held from each month's first session | +0.163 | +0.001 | 0.016 |
| daily `sign(basis)` — the committed rule inverted | +2.807 | +1.269 | 0.176 |

Holding the identical signal, rebalanced monthly instead of daily, moves develop+validate from -3.614
to +0.163; inverting the daily rule earns +2.807. No term-structure premium has a Sharpe of 3.6, and
the literature's version of this rule — long positive carry, short negative, equal-weighted, monthly —
reports a positive Sharpe of order 0.5–1 (M3; the diversified carry factor in Koijen, Moskowitz,
Pedersen & Vrugt 2018 is 1.10). So the tables measure "naive daily sign carry": the family's losses
are the churn of a basis-derived sign, which is also why they concentrate in 2015–2019 (-5.85). The
sign convention is *not* resolved by preferring the profitable direction — adopting the inverted sign
after seeing it win is the selection bias the DSR exists to catch — but neither is it left hanging:
the frequency diagnostic explains the number's sign and size, and M3's recommended construction
remains untested here. The carry premium is neither confirmed nor refuted by these tables.

---

## 8. Out-of-sample attribution

The window (2022 → 2024Q1) is a commodity-dominant inflationary regime with rapid rate moves, and the
project's own numbers decompose it without speculation:

| family | 2022 | 2023 | 2024Q1 |
|---|---|---|---|
| `xs_momentum` | 1.59 | -0.32 | 5.25 |
| `tsmom` (benchmark) | 0.98 | -0.17 | 1.29 |
| `ml_tuned` (6b) | 0.84 | -0.95 | 3.81 |
| `seasonal_tilt` | 0.82 | -0.13 | 0.55 |
| `ml_defaults` (6a) | 0.10 | -0.43 | 2.57 |
| `seasonality` | -0.10 | 1.54 | 1.76 |
| `carry` | -2.10 | -2.08 | 0.79 |
| `baseline` | -1.28 | -0.14 | -0.20 |

Two observations, both mechanical. **2023 is the year nobody earned:** with one exception, every
family is at or below zero, which is the flat, mean-reverting regime that M1's analysis and S9's
trend-follower straddle description identify as the trend family's worst case **[scaffolded: S9]** —
and it is the year the pre-declared winners' margins evaporated. **The passive book lost money in
all three windows while the cross-sectional family made money in all three:** out of time,
identifying *which* roots to hold paid, and holding all of them did not. The diversification return
that made the baseline strong in develop+validate (0.41 at 2 bps) was not available in this regime.

---

## 9. Statistical foundations, in one box

- **Sharpe 0.408** means: mean daily excess return over its standard deviation, annualized by
  `mean/std × √252`. It says nothing about skew or the path — a family with two large gains and many
  small losses and one with the reverse can share a Sharpe.
- **DSR 0.9989 vs 0.3726** means: the probability that the observed Sharpe exceeds the *expected
  maximum* Sharpe of a stated number of trials under zero skill, adjusted for the return series'
  skew and kurtosis. It is a multiple-testing correction, and its input — the trial count — is
  self-reported work. Two corrections are therefore worth more than the number: the log is
  committed, and the out-of-sample window is read once.
- **Turnover 0.103/day** means: mean daily traded notional **per unit of book capital** — the book's
  traded notional, averaged across the sixteen equal-capital sleeves, which is the same base the
  `bps` charge is taken on. Cost drag is `turnover × bps × 252 × 1e-4` per year in return terms; at
  2 bps and turnover 0.103 that is 0.52%/yr, and against the benchmark's 1.8% book volatility it is
  the 0.28 Sharpe between its +0.20 gross and -0.085 net. Cost sensitivity is therefore the
  second-order term everyone checks and the first-order term that decides this comparison.
- **Coverage 46.5%** means: the share of the evaluation window on which the family holds a
  non-constructed position. A Sharpe measured on a partially flat series is diluted by roughly
  √coverage, which is why the like-for-like read exists and why both reads are committed.
- **Max drawdown -0.691** (carry, develop+validate) means: the deepest peak-to-trough fall of the
  compounded equity curve. It is the number that survives a bad Sharpe — and it is the one that ends
  strategies in practice.
- **What none of these capture:** market impact, borrow/financing beyond the collateral convention,
  intraday execution, and the possibility that the next regime differs from both windows. Those are
  the limitations a live decision would have to add, not rebuild.

---

## 10. Limitations and honest boundaries

- **One universe, one protocol, one out-of-sample window.** 16 CME roots, 2010 → 2024Q1, one read.
  The out-of-time reversal is one draw of a regime, not a law.
- **The external reference is construction-matched, not level-matched.** The paper this benchmark
  follows (M1) was replicated on the same free snapshot (`results/replication/`, `FINDINGS.md`): its
  Eq. (5) run verbatim on the 52 instruments the source can supply puts the diversified factor at
  13.7% annualised volatility and gross Sharpe 1.20 over 1985–2009 — the paper states 12% and
  "greater than one" — and 0.58 gross / 0.56 net over 2010–2016, against the 0.41 *net* figure the
  trend literature publishes for that regime. This project's weak out-of-time trend numbers are
  therefore the published regime and its own cost model, not an implementation failure. What is *not*
  matched is the level: M1 chains the most liquid contract's daily return, so its series carries the
  price gap at each roll, while this repo's continuous series is roll-adjusted — on the commodity
  class that is worth a factor of 2.6 in the mean monthly return (0.500%/mo under the paper's splice,
  1.317%/mo roll-adjusted, published 0.59%/mo). And the paper's own 58-instrument universe is not
  rebuildable on free data (no LME metals before 2023, no Bund before 2006, no DAX before 2000, no
  CAC/AEX before 2009, no euro before 1999), so no like-for-like class level exists at this breadth.
  Per-instrument volatility is the one quantity that *is* directly comparable, and it lands within
  ±30% of the paper's table for the twelve instruments whose free history overlaps its sample.
- **A linear, constant cost model.** 0/5/10 bps sensitivity is a ladder, not a market-impact model;
  no bid/ask, no volume-dependent slippage. Volume exists in the source data but is informational
  only.
- **No CSCV/PBO diagnostic** (section 5) and **no naive-split counterfactual** (section 6): the two
  places where the multiple-testing story rests on argument rather than measurement.
- **Risk is not equalized across families.** The overlay targets 10% annualized volatility *per
  symbol*, not per book, so realized book volatility runs 0.9–3.0% in develop+validate and 1.5–4.6%
  out of time. Sharpe comparisons are like-for-like in rule and cost; drawdown and return levels are
  not (section 2).
- **Annualization is a stated convention, not a property of the panel.** Sharpe is reported at √252
  in both windows while the data carries 309.6 sessions per year before 2022 and 258.4 after, so the
  two windows' Sharpe *levels* are not strictly comparable — the develop+validate column is ~11%
  higher on its own session density. The DSR the rule reads is a probability and is unaffected
  (section 2).
- **Two SHOULD readings are scaffolded, not read** (S5, S9), and the M7 book chapters are not read;
  the two claims flagged **[scaffolded]** are the ones a reader should hold loosely.
- **The ML result is specific.** It does not say "machine learning does not work on futures". It says
  this feature set, this label horizon, this fold geometry, this cost level and this window produced
  an in-sample winner that did not generalize.

---

## 11. Non-adopt boundary

Deliberate, with reasons — the boundary is a research judgement, not a gap in the toolchain:

- **Deep learning / sequence models** — no comparative advantage claimed on 16 daily series with 14
  years of history; the data budget cannot support the parameter budget.
- **C++ / Rust / kdb+** — nothing here is latency- or scale-bound; rewriting the engine in a faster
  language would cost the one property that matters (one readable accounting path).
- **Dashboards and MLOps tooling** — the deliverable is a findings memo, not a product; committed
  notebooks, manifests and CI cover reproducibility without a serving stack.
- **Microstructure and order-book work** — the data is daily contract closes; there is nothing to
  model.
- **VRP / short-vol**, **standalone mean-reversion**, **stat-arb / pairs** — research-skipped with
  cited reasons in the map's method-family research (data-impossible, dead in commodities after
  costs, and too thin on 16 roots respectively).
- **Live trading and order routing** — explicitly out of scope; this is a research lab, not an
  MiFID II algorithmic trading system.
- **Macro features in the ML panel** — the free-data posture has no redistributable point-in-time
  macro source, so the models learn from the panel the repo owns. Recorded in `docs/methods/ml_variants.md`.

---

## 12. Provenance

- Decision reasoning: `docs/adr/0001-one-accounting-path.md`; per-family governance docs in
  `docs/methods/`.
- Reading canon and notes: `docs/reading/`.
- Committed numbers: `results/families/`, `results/decision/`, `results/out_of_sample/`; decision
  records `results/decision/DECISION_RULE.md`, `DECISION.md`, `results/out_of_sample/OOT.md`; and the
  carry frequency diagnostic `results/out_of_sample/carry_frequency.csv`
  (`scripts/carry_frequency_diagnostic.py`).
- The executable version of this memo's story: `notebooks/analysis.ipynb`.

**Market facts used in the framing:** the Danish systematic shop is Alipes Capital (Copenhagen); Da
Vinci Trading is Amsterdam (a correction carried from the research phase). Regulation talking points
live in the self-test brief.
