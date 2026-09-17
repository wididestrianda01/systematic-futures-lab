# Self-test brief — the project as rehearsal material

**How to use this.** Answer each question aloud, in 60–90 seconds, with one number attached. The
numbers here are the committed ones; if you cannot say where a number comes from, the answer is not
ready. The regulation section is deliberately last in priority order: the method comparison is the
point, and regulation depth is supporting material, offered, not led with.

## The 90-second summary

"I built a method-comparison lab for systematic futures: 16 CME roots, 2010 to 2024Q1, contract-level
data with self-built roll calendars and back-adjusted continuous series, one vectorised engine, one
volatility overlay and one cost model for every family. Six classic families plus two LightGBM
variants run through the identical accounting path, with a pre-declared selection rule and a single
out-of-sample window read once. The result that matters: the tuned ML variant beat the trend
benchmark inside the decision window on deflated Sharpe (0.9999 against 0.3726) and failed the same
test out of time, 0.0261 against 0.7966. The family that held up in every sub-period was
cross-sectional momentum, and the family that beat half the signals after costs was the null model,
which is the diversification return doing its work. Everything is reproducible from a tagged
revision, and no market data is in the repository."

## Self-test questions

**1. Why is the out-of-sample window read only once?**
Because every additional look turns the test into a selection problem. The window's job is to be the
one number no decision consumed on the way in. In this project the rule was written before the
decision read's numbers existed, the window is read by one script, and it is extended-walk-forward so that
the fold covering 2022 trains entirely inside develop+validate. The pay-off is not theoretical: the
pre-declared winners failed that read, and if the window had been peeked at earlier the temptation to
retune (and the DSR inflation that follows) would have been enormous.

**2. A model beats its benchmark with DSR 0.9999 after 100 configurations. Why is that not enough?**
Two reasons. First, in this case the benchmark lost money in that window (−0.085 Sharpe at 2 bps), so
"beat the benchmark" was a low bar: the variants did not lose money where the benchmark did. Second,
the Deflated Sharpe Ratio corrects for how many trials you ran, not for whether the *feature set*
generalises: it is a significance test under zero skill, not a promise about the next regime. The
honest evidence is the sub-period decomposition: the variants' entire in-window advantage sits in
2015–2019 and 2020–2021, the two periods the benchmark lost money, while in 2010–2014 the defaults
variant was −0.80 and the tuned variant produced no coverage at all.

**3. What do purge and embargo remove that a shuffled k-fold does not, and what do they not fix?**
With a five-session forward label and daily samples, a training row five days before a test block
already contains part of that block's outcome; a shuffled or plain k-fold split splits *samples*, not
time, so it cannot see the overlap. Purge deletes training dates whose label window reaches into the
test block; embargo deletes a further buffer after each test block because serial correlation bleeds
across the boundary anyway. They do not fix regime dependence: they guarantee no label leakage, and
this project asserts that property on every run by re-deriving it from the fold semantics. What they
cannot do is create out-of-sample persistence that the underlying premia do not have, which is
exactly what the out-of-time read showed.

**4. Cost flips the trend benchmark's sign in develop+validate but barely touches the baseline. One
statistic explains it.**
Turnover. The benchmark trades 0.103 of book notional per day against the baseline's 0.007:
book-level, the traded notional of the equal-capital book per unit of capital, which is the base the
charge is taken on. Drag is `turnover × bps × 252 × 1e-4` in annual return terms: at 2 bps the
benchmark pays 0.52%/yr and the baseline 0.035%/yr. Against the benchmark's 1.8% book volatility that
drag is the 0.28 Sharpe between +0.200 gross and −0.085 net. The second half of the question is that
cost is not the only way a family fails: standalone seasonality is negative *before* costs (−0.027
gross on turnover of 0.025, a quarter of the benchmark's), so there the signal, not the churn, is the
problem. In monitoring terms I would track turnover and the cost-to-gross ratio per family, not just
Sharpe, and at 5 bps three of the eight families stay positive in develop+validate (baseline 0.390,
XS momentum 0.172, the tuned variant 0.746), while the out-of-sample window leaves four (XS momentum
0.994, seasonality 0.683, the trend benchmark 0.208, the tilt 0.054).

**5. Your carry family is the worst performer in both windows. Argue for and against "carry does not
work in commodities".**
For: −3.614 Sharpe at 2 bps in develop+validate, −1.756 out of time, worst in the table on both, with
the highest turnover of the classic set (0.176/day), and catastrophic in 2015–2019 (−5.85).
Against: the published construction (Koijen–Moskowitz–Pedersen–Vrugt) is a slope *magnitude*,
`(F1−F2)/(F2·ΔT)` per month, with rank weighting, monthly rebalancing and a 12-month-mean variant
recommended because current-slope carry is seasonal. What I ran is the *sign* of the current basis,
refreshed daily, and that sign moves with the front leg's own price, so the daily refresh trades the
basis's own noise. The measured tell is in `results/out_of_sample/carry_frequency.csv`: holding the identical
signal from each month's first session earns +0.163 where the daily refresh earns −3.614, and
inverting the daily rule earns +2.807. No term-structure premium has a Sharpe of 3.6 in either
direction, so the defensible conclusion is that the tables measure naive daily sign carry, not the
carry factor. And I would not resolve it by flipping the sign: adopting the profitable direction after
seeing it win is exactly the selection bias the deflated Sharpe exists to catch.

**6. Why is a null model in your comparison set, and what did it do here?**
Because a long-only, vol-targeted book mechanically earns the diversification return, and any signal
has to beat that before it has demonstrated anything. It earned Sharpe 0.408 at 2 bps on book
turnover of 0.007 in develop+validate, which beats the trend benchmark, the seasonal tilt and standalone
seasonality after costs. Out of time it lost money in all three years, which is the other half of the
lesson: the free lunch is regime-dependent, and in 2022–2024Q1 the cross-section paid while passive
exposure did not.

**7. How do you know the back-adjusted continuous series is right?**
There is an invariant and a test for it. The position holds `front(t)` from `t−1`'s close to `t`'s
close, so the true return is `close(front(t), t) / close(front(t), t−1) − 1` on every session,
including roll days. The builder must reproduce that series exactly: the notebook's synthetic demo
shows the maximum deviation at 2.2e-16 across 261 sessions with 12 observed rolls, against a mean
phantom return of 2.23% per roll day for an unadjusted stitch. The roll calendar is *observed* from
the leg data rather than estimated from last-trade-date guesses, so there is no external rule to be
wrong about.

**8. If you had another ten hours on this project, what would you do, and what would you refuse?**
Do: implement the published carry construction properly (monthly rebalance, rank weighting,
carry1-12 robustness) and see whether the family behaves as the literature says; add the CSCV
probability-of-backtest-overfitting diagnostic, which this project deliberately does not implement;
and test the same protocol on a second universe to separate the universe effect from the regime
effect. Refuse: retuning the ML variants against the out-of-sample window, adopting the inverted carry
sign because it backtests better, adding deep learning to a 16-series daily dataset, and turning this
into a live system. Each refusal is a research judgement, and I would say so before being asked.

## Regulation talking points

Grounded in the project's own research pass over the primary regulation and the ESMA briefing,
with the sources listed inline below.

1. **Scope line.** Algorithmic trading is defined in MiFID II Art. 4(1)(39) as trading where an
   algorithm automatically determines order parameters, and Art. 1(5) extends the Art. 17 / RTS 6
   obligations to members of regulated markets and MTFs that are not authorised investment firms;
   prop shops with exchange membership are in scope. Honest boundary: this project is *not* an algorithmic
   trading system, because it submits no orders to any venue; it is a research backtester. The moment
   it routed orders, RTS 6 would attach.
2. **RTS 6 Art. 15 pre-trade controls.** Price collars, maximum order value and volume, maximum
   message limits and repeated-execution throttles, implemented as hard blocks a trader cannot
   override; the ESMA briefing's 2024 Common Supervisory Action (a follow-up to the 2022 Nordic flash
   crash) found most firms had integrated them but with divergent, sometimes fragile governance. The
   research analogue here is the ±1 notional position cap applied identically to every family, plus
   the cost charge as a pre-trade viability filter.
3. **Kill functionality and attribution (Art. 2, Recital 9).** The firm must be able to withdraw all
   or some orders immediately and always know which algorithm, trader or client is responsible. The
   research analogue is the tag ↔ manifest ↔ results chain: any committed number can be traced to the
   exact revision and the exact frozen data it came from, and the whole pipeline re-runs from one
   command.
4. **Annual self-assessment (Art. 9) and stress testing (Art. 10).** Documented validation reviewed by
   internal audit, and the ability to withstand twice the highest message and trade volume of the
   previous six months. Analogue: the test suite plus the manifest gate, which fails ingest on any
   data drift. Honest gap: no message-flow stress test exists here because there is no message flow.
   Saying that plainly is better than implying coverage.
5. **"Material change" retesting.** The ESMA briefing's own table includes retrained or modified ML
   components and changed data feeds. This is where an Optuna search stops being a notebook habit and
   becomes a governance event: `docs/methods/` names, per family, what counts as a material change
   (feature set, label horizon, fold geometry, search space, cost level, universe), and the decision
   rule was committed *before* the numbers existed.
6. **MAR and the AI Act.** MAR covers commodity derivatives and benchmarks, not just equities, and
   the canonical manipulation patterns are spoofing, layering and momentum ignition; RTS 6 requires
   monitoring that systems cannot be used contrary to MAR. The AI Act (Reg. 2024/1689) interplay per
   the ESMA briefing: AI-based algo trading is currently not classified as high-risk, the
   classification is under annual review, and firms must cover AI usage in the Art. 9 self-assessment
   and be able to explain how AI affects decisions. This project's ML section is that explanation in miniature:
   trial-count deflation, coverage, purged folds, and an out-of-time failure published as a finding.
7. **EMIR and the Swedish layer.** All derivative contracts, listed futures included, are reported to
   trade repositories (REFIT: ISO 20022 XML with UTI/LEI pairing); the 2024 clearing reform adds an
   active-account requirement at EU CCPs. Nationally, Finansinspektionen supervises via lag (2007:528)
   and FFFS 2017:2, on top of the directly applicable EU regulations: ESMA writes the rulebook, FI
   supervises locally. The one-liner: "every simulated position here would be an EMIR report within
   T+1."

## Licensing posture (a differentiator worth one sentence)

No market data is in the repository, private or public: the frozen manifest carries checksums, schema
and date ranges, and the committed artifacts are derived statistics (backtest results, signals,
equity curves) which sit inside the derived-data carve-outs of the vendor terms surveyed (Norgate
explicitly retains backtest statistics after subscription expiry; Nasdaq Data Link requires written
approval for derived-data publication, which is why the data source here is a code-and-checksum
pattern rather than a licensed feed). That boundary is a design decision made before the first
backtest, not a cleanup afterwards.

## Non-adopt boundary

Deep learning, C++/Rust/kdb+, dashboards, MLOps tooling, microstructure, live trading: each skipped
with a stated reason (see `docs/findings/memo.md` §11). The point of naming them is that a reader
should see judgement rather than absence, and be able to interrogate it.

## Market facts to get right

- The Danish systematic shop is **Alipes Capital** (Copenhagen); **Da Vinci Trading** is Amsterdam.
- Reading provenance as of this revision: M1–M6 and M8 read and closed; M7 (AFML ch. 7, 11–12) and
  S5/S9 are scaffolded, not read; do not claim them as read until they are.
