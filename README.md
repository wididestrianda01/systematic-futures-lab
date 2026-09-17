# Systematic futures: a method-comparison lab

[![License: GPL-3.0](https://img.shields.io/badge/license-GPL--3.0-blue.svg)](LICENSE)

Do six classic futures method families, and a machine-learned challenger, survive after costs on real
CME contract data? Sixteen CME roots, 2010 to 2024Q1, one accounting engine, one cost model, one
evaluation protocol, and one out-of-sample window read once. The primary artifact is a findings memo:
where each method wins, loses, and why, supported by an executed walkthrough, per-method governance
docs and a condensed report.

Read: [findings memo](docs/findings/memo.md) ·
[executed walkthrough](notebooks/analysis.ipynb) ·
[condensed report](docs/report/report.pdf)

Market data never enters this repository: only derived artifacts (code, configs, checksums,
signals, statistics) are committed.

## What it is for

**The decision it supports.** Whether to run a systematic futures book, and with which method
families. That is a capital-allocation question, so the lab is built the way a desk would build it:
one accounting path, a frozen dataset, a pre-declared selection rule, and an out-of-sample window
spent once, at the end.

**What trading the book costs.** Cost level decides this comparison before signal quality does. At
2 bps a side the trend benchmark pays 0.52% of capital a year and gives up its gross edge (-0.085 net
against +0.20 gross), while the tuned variant pays 0.51% and keeps 1.550. Raise the charge to 10 bps
and the trend benchmark is at -1.20 while the low-turnover families barely move. Out of time the same
mechanism shows up on `ml_defaults`: 0.759 gross becomes 0.072 net on a 1.00% a year drag. Every
family is reported across the 0/2/5/10 bps ladder.

**Two jobs, deliberately bundled.** Learn the craft, and prove it. Futures are where systematic
trading is most explicit, since rolls, term structure, carry, volatility targeting, costs and
multiple testing all appear in the same object, so a lab built here exercises the whole research
cycle: research from signal generation to implementation, calibration, validation and monitoring,
cost and turnover discipline, and overfitting as a quantity you measure.

## Headline results

Sharpe at the headline cost level (2 bps/side) with the Deflated Sharpe Ratio, on the decision window
(develop+validate, 2010-01-01 → 2021-12-31) and on the single out-of-sample read
(2022-01-03 → 2024-03-28):

| family | dev+validate Sharpe | dev+validate DSR | out-of-sample Sharpe | out-of-sample DSR |
|---|---|---|---|---|
| Baseline (null model) | 0.408 | 0.9403 | -0.694 | 0.1470 |
| TSMOM / trend (benchmark) | -0.085 | 0.3726 | 0.551 | 0.7966 |
| Cross-sectional momentum | 0.406 | 0.9397 | **1.174** | 0.9611 |
| Carry | -3.614 | 0.0000 | -1.756 | 0.0041 |
| Seasonality (standalone) | -0.088 | 0.3683 | 0.770 | 0.8793 |
| Seasonality (tilt on trend) | -0.116 | 0.3284 | 0.407 | 0.7304 |
| ML 6a `ml_defaults` | 0.799 | 0.9989 | 0.072 | 0.5433 |
| ML 6b `ml_tuned` | **1.550** | **0.9999** | 0.390 | 0.0261 |

Four of the eight families are positive after costs on the decision window and six are positive out
of time, but only three are positive in both: cross-sectional momentum, and the two ML variants.
Cross-sectional momentum is the only family positive in every sub-period and in both windows.

The pre-declared rule (an ML variant wins only if its decision-window deflated Sharpe after costs
beats the benchmark's) was written before the decision read's numbers existed. Both variants clear it
inside the decision window and both fail it out of time. That is reported as the finding, not retuned
away; `results/out_of_sample/OOT.md` records the read, and `results/decision/DECISION.md` keeps the
decision read's verdict as decided.

## Reading the results

The table is the arithmetic; this section is what it means. Every number below is read from the
committed tables under `results/`, and the long form with citations is `docs/findings/memo.md`.

**Costs select strategies rather than shading them.** Ten basis points a side changes which family
wins, and turnover decides it: the ladder charges the same turnover at a different price, so what it
sorts is turnover. Book turnover runs 0.007 of capital a day for the baseline, 0.025 for seasonality,
0.049 for cross-sectional momentum, 0.096 for the tilt, 0.103 for the trend benchmark and 0.176 for
carry. The benchmark travels +0.200 gross → -0.085 at the headline level → -1.203 at 10 bps; the tilt
+0.163 → -0.116 → -1.208; the low-turnover families barely move (baseline 0.419 → 0.361,
cross-sectional momentum 0.562 → -0.216). At 5 bps the survivors differ by window: develop+validate
leaves the baseline (0.390), cross-sectional momentum (0.172) and the tuned variant (0.746) positive,
while the out-of-sample window leaves cross-sectional momentum (0.994), seasonality (0.683), the trend
benchmark (0.208) and the tilt (0.054).

**The null model is the bar, and four families fail to clear it.** Vol-targeted buy-and-hold earns
Sharpe 0.408 at 2 bps on 0.007 of daily book turnover, a charge of 3.5 bps a year, and that number is
the diversification return rather than a signal. On the decision window the trend benchmark, carry,
standalone seasonality and the tilt all finish below it after costs. Out of time the same book loses
money in all three years (-1.28 in 2022, -0.14 in 2023, -0.20 in 2024Q1), which is the other half of
the lesson: the free lunch is regime-dependent, and in that window the cross-section paid while
passive exposure did not.

**The ML verdict is the honest one.** The rule was pre-declared before these numbers existed: a
variant wins only if its decision-window deflated Sharpe after costs beats the benchmark's. Both
variants clear it in develop+validate (DSR 0.9989 and 0.9999 against 0.3726) and both fail it on the
single out-of-sample read (0.5433 and 0.0261 against 0.7966). The in-window edge is also
concentrated where the bar was lowest, since the benchmark earned a negative net Sharpe in that
window: clearing it means the variants did not lose where the benchmark did, not that skill was
demonstrated.

| family | 2010–2014 | 2015–2019 | 2020–2021 |
|---|---|---|---|
| `tsmom` (benchmark) | 0.15 | -0.51 | 0.33 |
| `ml_defaults` (6a) | -0.80 | 1.50 | 1.17 |
| `ml_tuned` (6b) | no coverage | 1.97 | 2.22 |
| `xs_momentum` | 0.41 | 0.30 | 0.66 |

Out of time the cost charge consumes 6a's signal (0.759 gross, 0.072 net on 0.199 a day of book
turnover, a 1.00% a year charge), and 6b holds 0.390 with a DSR of 0.0261 under its 100-configuration
deflation. Purged, embargoed folds and the trial-count deflation guarantee no label leakage and an
honest search count. Neither can create persistence the features do not have: trailing returns,
realized vol, moving-average distance, basis and seasonality are a non-linear re-expression of the
same premia the classic families trade, so when the benchmark's regime turns the recombination has
nothing extra to lean on.

**Two failures worth more than the losers.** Standalone seasonality is negative *before* costs
(-0.027 gross) at book turnover of 0.025, a quarter of the benchmark's, so churn is not the
explanation; the score rests on five priors per calendar month, and out of time the same family earns
+0.828 gross. Carry's number belongs to the rebalancing frequency, not to the carry premium:

| construction | develop+validate | out-of-sample | turnover |
|---|---|---|---|
| daily `sign(-basis)`, the committed row | -3.614 | -1.756 | 0.176 |
| the same rule held from each month's first session | +0.163 | +0.001 | 0.016 |
| daily `sign(basis)`, the committed rule inverted | +2.807 | +1.269 | 0.176 |

The published construction is a monthly, rank-weighted slope *magnitude*, and no term-structure
premium has a Sharpe of 3.6 in either direction, so the committed row measures naive daily sign
carry. The sign convention is not resolved by adopting the profitable direction; that is the
selection bias the deflated Sharpe exists to catch. Evidence:
`results/out_of_sample/carry_frequency.csv`.

**What generalised was what did not need to time the regime.** Cross-sectional momentum is positive
in every sub-period of the decision window (0.41 / 0.30 / 0.66) and in both windows (0.406 at 2 bps
in, 1.174 out). Out of time it earned 1.59 in 2022 and 5.25 in 2024Q1 against -0.32 in 2023, and 2023
is the year nobody earned: with one exception every family sits at or below zero, which is the flat,
mean-reverting regime the trend literature names as the family's worst case. The passive book lost
money in all three of those years while the timing and selection families earned it.

**How to read these numbers.** Sharpe is annualized at √252 in both windows as a stated convention,
while the frozen panel carries 309.6 sessions a year before 2022 and 258.4 after, so the two windows'
Sharpe levels are not strictly comparable and the develop+validate column sits about 11% above its
own session density; the DSR the rule reads is a probability and is unaffected. Sizing targets
volatility per symbol rather than per book, so realized book volatility runs 0.9–3.0% in
develop+validate and 1.5–4.6% out of time: the families are like-for-like in rule and cost, not in
delivered risk. Coverage dilutes a Sharpe by roughly √coverage, which is why each ML variant is also
read on its own covered dates (benchmark DSR 0.4253 on 6a's dates, 0.3651 on 6b's, both verdicts
unchanged). The DSR bounds selection bias under zero skill; it is not a promise about the next
window, and it did not save these variants.

## The dataset

Raw per-contract daily prices for **16 CME roots** (ES, NQ, YM, ZN, ZB, ZF, GC, SI, HG, CL, NG, ZC,
ZW, 6E, 6J, 6B) from pysystemtrade's free `multiple_prices_csv` snapshot, pinned at commit `b4a25e6`
and frozen upstream on 2024-03-28. Each row carries three legs with contract identifiers (front,
successor and prior), so rolls, back-adjustment and basis are built in this project rather than
inherited from a vendor's adjusted series. KC/SB/CC (ICE softs, no free feed) and BZ (Brent legs start
in 2020-08) are excluded, recorded in code with their reasons.

**Licensing posture.** The repository ships no market data, private or public: `data/` is gitignored,
and everything committed under `results/` is a derived statistic (metrics, return series, signals,
checksums). A committed manifest carries SHA-256, schema and date range per derived file, and ingest
fails on any drift. Details: `docs/data/pst-source.md`.

## The pipeline

```
raw contract legs ──▶ observed roll calendar ──▶ per-contract closes ──▶ ratio back-adjusted
continuous series ──▶ basis (raw front/next) ──▶ signals ──▶ shared vol overlay ──▶ costs ──▶ metrics
```

- **Roll calendar**: observed from the leg data. The front contract flips exactly when the data says
  it flips, with no last-trade-date estimate and no vendor rule.
- **Continuous series**: ratio back-adjustment, applied backwards from the newest contract, with one
  invariant pinned by test: the continuous series' daily return equals the held contract's own
  return on every session, rolls included.
- **Basis**: `close(next)/close(front) − 1` on raw closes; days missing a leg are dropped, never
  zero-filled.

## Methodology and the decisions behind it

- **One accounting path** (ADR 0001): every method is a callable `closes → signals`; the engine owns
  sizing, P&L, costs and metrics. The arithmetic that tunes a model and the arithmetic that publishes
  a table cannot drift apart, and a test pins the equivalence.
- **Shared overlay**: signals scaled to a 10% annualized volatility target per symbol on trailing
  realized vol, clipped to ±1 notional, identically for every family. Per-symbol sizing is not
  book-level sizing: realized book volatility is 0.9–3.0% in develop+validate and 1.5–4.6% out of
  time, so the families are like-for-like in rule and cost, not in delivered risk.
- **Costs**: a linear charge of `bps` per side on traded notional, reported at 0/2/5/10 bps with 2 bps
  the headline level a decision is read at. Turnover is book-level (`turnover × bps × 252 × 1e-4` is
  the annual return a cost level costs), measured as mean daily traded notional per unit of capital,
  which is the same object the charge is taken on. At 2 bps that runs from 0.25%/yr for
  cross-sectional momentum (turnover 0.049) to 0.89%/yr for carry (0.176) and 1.00%/yr for
  `ml_defaults` out of time (0.199). The trend benchmark's 0.52%/yr, against its 1.8% realized book
  volatility, is the 0.28 Sharpe between its +0.20 gross and -0.085 net.
- **Reported units**: Sharpe is annualized at √252 in both windows as a stated convention, not
  because the panel has 252 sessions a year. The frozen data carries 309.6 sessions per year in
  develop+validate (Sunday bars included) and 258.4 from 2022, so the two windows' Sharpe *levels* are
  ~9% apart on equal annualization (√(309.6/258.4)); the develop+validate column printed at √252 sits
  ~11% below its own session density. The DSR the rule reads is a probability and is unaffected.
- **No look-ahead, structurally**: a signal formed with data through day `t` drives the position held
  from `t` to `t+1`; a regression test fails on a look-ahead implementation and passes on the correct
  one.
- **Split contract**: develop 2010–2019, validate 2020–2021, out-of-sample 2022 → 2024Q1 read exactly
  once at the end. Walk-forward refits stay inside develop+validate.
- **Two reads, one decision surface**: the primary identical-window read, plus a like-for-like read
  re-measuring each ML variant on its own covered dates; the run stops if they disagree.
- **Multiple testing**: the Deflated Sharpe Ratio is deflated by the configurations each selection
  evaluated (1 for every classic family, 100 for the tuned variant), and the ML folds are purged and
  embargoed by hand, with `leakage_violations` re-deriving the property from the fold semantics on
  every run. M6's CSCV probability-of-backtest-overfitting diagnostic is not implemented; the gap is
  recorded rather than glossed.

## Design and architecture

- `src/systematic_futures/data/`: universe map, observed roll calendar, ratio-adjusted continuous
  builder, basis, the manifest gate, a DuckDB/Parquet query layer.
- `src/systematic_futures/engine/`: the seam: `run(method, data) → metrics table`, the sizing
  overlay, the held-position path, the cost charge, metrics, DSR, and `curve()` for plotted series.
- `src/systematic_futures/methods/`: one callable per family, each with a hand-calculated fixture
  pinning its sign and scale at the seam.
- `src/systematic_futures/ml/`: point-in-time features, the hand-rolled purged/embargoed splitter,
  the two LightGBM variants.
- `protocol.py`: the comparison's two contracts: the protocol (fold geometry, horizon, search budget,
  cost level, overlay) and the method (a signal source plus the trial count its selection evaluated).
- `comparison.py`: the comparison table's shape and every read of it: assembly, the headline read,
  coverage and trial metadata, the like-for-like mask. It imports no family, so the shape costs nothing.
- `catalogue.py`: the comparison set (classics plus 6a/6b) and the walk-forward entry point.
- `reporting.py`: the rendered outcome: the markdown body of the decision records, from the same read.
- `decision.py`: the pre-declared rule and its guard, written once so the phases cannot drift apart.
- `scripts/`: thin runners; `notebooks/`: explanations. Logic lives in the package, arguments live
  in the runners.
- CI runs the test suite (123 tests) on every push.

## Where the interpretation lives

`docs/findings/memo.md` is the argument: the regime the trend benchmark lived through, why the ML
variants' in-window edge was concentrated in the periods the benchmark lost money, what purge and
embargo bought and what they cannot buy, and the two honest-failure findings. `docs/reading/notes.md`
records the reading behind the interpretation.

## Reproduce

Prerequisites: Python 3.12, [uv](https://docs.astral.sh/uv/), and network access for the first step
only. The dataset is fetched from the pinned upstream commit and verified against the committed
manifest; re-runs without network work off the frozen local store.

```bash
git checkout replication.1
uv sync
uv run python scripts/fetch_raw.py        # writes data/raw/, manifest-gated
uv run python scripts/build_derived.py    # writes data/derived/, manifest-verified
uv run python scripts/build_families.py       # results/families/
uv run python scripts/build_decision.py       # results/decision/
uv run python scripts/build_out_of_sample.py  # the single out-of-sample read
uv run python scripts/sweep_tsmom.py      # results/trend_sweep/tsmom_sweep.csv
uv run python scripts/carry_frequency_diagnostic.py  # results/out_of_sample/carry_frequency.csv
uv run python scripts/build_report_figures.py        # docs/report/figures/equity.pdf
uv run python scripts/fetch_replication.py           # data/replication/, the MOP subset
uv run python scripts/build_mop_replication.py       # results/replication/ - external checks
uv run pytest                             # 123 tests
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/analysis.ipynb
(cd docs/report && rm -f report.aux report.log report.out && \
   pdflatex -interaction=nonstopmode report.tex)   # report.pdf, reproducible from a clean build
```

The phase scripts write the same tables that are committed. Regeneration is deterministic by
construction: seeded fixture demos, frozen inputs, committed outputs, and no wall-clock anywhere in
the tables. `diff -r` against the committed `results/` tree is the check: the tables, decision
records and return series rebuild byte-for-byte, as do the report figure and the report PDF. The
executed notebook is deterministic in *content*, since every printed number matches, but the kernel
stamps cell timings and may split one stdout stream into two, so compare its outputs and not its
bytes.

`results/replication/` is the external check on all of that: MOP (2012) Eq. (5) run verbatim on the
52 instruments the free snapshot can supply from the paper's Table 1, with `FINDINGS.md` stating what
reproduces (13.7% volatility and Sharpe 1.20 for the diversified factor against the paper's 12% and
"greater than one") and what cannot (the paper's own 58-instrument universe, since free data carries
no LME metals, Bund, DAX, CAC or euro before 2000).

## Repository layout

| path | what it holds |
|---|---|
| `docs/findings/memo.md` | the primary artifact: findings, readings, limitations, non-adopts |
| `notebooks/analysis.ipynb` | the executed walkthrough: context, data, pipeline, methodology, results, readings |
| `docs/methods/` | per-family governance docs: construction, evidence, failure modes, monitoring |
| `docs/report/report.tex`, `report.pdf` | the condensed report |
| `results/families/`, `results/decision/`, `results/out_of_sample/` | committed tables, decision records, derived return series, the carry frequency diagnostic |
| `results/trend_sweep/` | the TSMOM horizon sweep, one sleeve per lookback |
| `docs/adr/` | decision reasoning (one accounting path) |
| `docs/data/pst-source.md` | data provenance and licensing posture |
| `LICENSE` | GPL-3.0 |

## Boundaries

- **Not claimed**: live trading, order routing, intraday execution, market impact beyond the constant
  cost ladder, and any statement about machine learning on futures in general, since the ML result is
  specific to this feature set, label horizon, fold geometry, cost level and window.
- **Out of scope**: deep learning, C++/Rust/kdb+, dashboards, MLOps tooling, microstructure,
  VRP/short-vol, standalone mean-reversion, stat-arb/pairs, each with a stated reason in the memo's
  non-adopt boundary.
- **One universe, one window**: 16 roots and a single out-of-sample read; the out-of-time reversal is
  one draw of a regime, not a law.

## License

GPL-3.0-or-later. Copyright (C) 2026 wididestrianda01. The full text is in [LICENSE](LICENSE); the
data source and its licensing posture are documented in `docs/data/pst-source.md`.
