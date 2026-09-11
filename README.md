# this project — Systematic futures: a method-comparison lab

One engine, one cost model, one evaluation protocol, and a single out-of-sample window read once:
six classic futures method families and two LightGBM variants compared on real CME contract data from
2010 to 2024Q1. The primary artifact is a findings memo — where each method wins, loses, and why —
supported by an executable walkthrough, per-method governance docs, a condensed report and an
interview brief.

**Status:** complete — Phases 0–5 built, results committed, tagged `results.1`. Market data never
enters this repository: only derived artifacts (code, configs, checksums, signals, statistics) are
committed.

## What it is for

**The decision it supports.** Whether to run a systematic futures book, and with which method
families. That is a capital-allocation decision, so the lab is built the way a desk would build it:
one accounting path, a frozen dataset, a pre-declared selection rule, and an out-of-sample window that
was spent once, at the end.

**Two jobs, deliberately bundled.** First, learn the craft — futures are where systematic trading is
most explicit, since rolls, term structure, carry, volatility targeting, costs and multiple testing
all appear in the same object. Second, prove the craft: the vocabulary throughout (research cycle from
signal generation to implementation; calibration, validation, monitoring; cost and turnover
discipline; overfitting as a measured quantity) is the vocabulary systematic employers screen for.

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

The pre-declared rule — an ML variant wins only if its decision-window deflated Sharpe after costs
beats the benchmark's — was written before the decision-read numbers existed. Both variants clear it inside
the decision window and **both fail it out of time**. That is reported as the finding, not retuned
away; `results/out_of_sample/OOT.md` records the read, and `results/decision/DECISION.md` keeps the decision-read
verdict as decided.

Four further readings, argued in the memo:

- **Costs select strategies rather than shading them.** At 5 bps only cross-sectional momentum, the
  baseline and out-of-sample seasonality remain positive; trend, the seasonal tilt and both ML
  variants flip sign between gross and headline cost.
- **The null model is a real competitor.** Vol-targeted buy-and-hold earns 0.408 at 2 bps on turnover
  of 0.112 — the diversification return, which a signal has to beat before it has shown anything.
- **What generalised is what did not need to time the regime.** Cross-sectional momentum is positive
  in every sub-period and both windows; the passive book lost money in all three out-of-sample years
  while the timing families earned it.
- **Two honest-failure results.** Standalone seasonality is negative before costs at turnover below
  the benchmark's (so churn is not the explanation), and carry's result traces to the implementation
  under test — a daily sign of the current basis where the published construction is a monthly,
  rank-weighted slope magnitude — so the carry premium is neither confirmed nor refuted here.

## The dataset

Raw per-contract daily prices for **16 CME roots** (ES, NQ, YM, ZN, ZB, ZF, GC, SI, HG, CL, NG, ZC,
ZW, 6E, 6J, 6B) from pysystemtrade's free `multiple_prices_csv` snapshot, pinned at commit `b4a25e6`
and frozen upstream on 2024-03-28. Each row carries three legs with contract identifiers — front,
successor and prior — so rolls, back-adjustment and basis are built in this project rather than
inherited from a vendor's adjusted series. KC/SB/CC (ICE softs, no free feed) and BZ (Brent legs start
in 2020-08) are excluded, recorded in code with their reasons.

**Licensing posture.** The repository ships no market data, private or public: `data/` is gitignored,
and everything committed under `results/` is a derived statistic — metrics, return series, signals,
checksums. A committed manifest carries SHA-256, schema and date range per derived file, and ingest
fails on any drift. Details: `docs/data/pst-source.md`.

## The pipeline

```
raw contract legs ──▶ observed roll calendar ──▶ per-contract closes ──▶ ratio back-adjusted
continuous series ──▶ basis (raw front/next) ──▶ signals ──▶ shared vol overlay ──▶ costs ──▶ metrics
```

- **Roll calendar**: observed from the leg data. The front contract flips exactly when the data says
  it flips — no last-trade-date estimate, no vendor rule.
- **Continuous series**: ratio back-adjustment, applied backwards from the newest contract, with one
  invariant pinned by test — the continuous series' daily return equals the held contract's own
  return on every session, rolls included.
- **Basis**: `close(next)/close(front) − 1` on raw closes; days missing a leg are dropped, never
  zero-filled.

## Methodology and the decisions behind it

- **One accounting path** (ADR 0001): every method is a callable `closes → signals`; the engine owns
  sizing, P&L, costs and metrics. The arithmetic that tunes a model and the arithmetic that publishes
  a table cannot drift apart, and a test pins the equivalence.
- **Shared overlay**: signals scaled to a 10% annualized volatility target on trailing realized vol,
  clipped to ±1 notional per symbol — identically for every family.
- **Costs**: a linear charge of `bps` per side on traded notional, reported at 0/2/5/10 bps with 2 bps
  the headline level a decision is read at; turnover comes from the same object the cost is charged on.
- **No look-ahead, structurally**: a signal formed with data through day `t` drives the position held
  from `t` to `t+1`; a regression test fails on a look-ahead implementation and passes on the correct
  one.
- **Split contract**: develop 2010–2019, validate 2020–2021, out-of-sample 2022 → 2024Q1 read exactly
  once at the end. Walk-forward refits stay inside develop+validate.
- **Two reads, one decision surface**: the primary identical-window read, plus a like-for-like read
  re-measuring each ML variant on its own covered dates; the run stops if they disagree.
- **Multiple testing**: the Deflated Sharpe Ratio is deflated by the configurations each selection
  evaluated (1 for every classic family, 100 for the tuned variant), and the ML folds are purged and
  embargoed by hand — `leakage_violations` re-derives the property from the fold semantics on every
  run. M6's CSCV probability-of-backtest-overfitting diagnostic is deliberately *not* implemented;
  the gap is recorded rather than glossed.

## Design and architecture

- `src/systematic_futures/data/` — universe map, observed roll calendar, ratio-adjusted continuous
  builder, basis, the manifest gate, a DuckDB/Parquet query layer.
- `src/systematic_futures/engine/` — the seam: `run(method, data) → metrics table`, the sizing
  overlay, the held-position path, the cost charge, metrics, DSR, and `curve()` for plotted series.
- `src/systematic_futures/methods/` — one callable per family, each with a hand-calculated fixture
  pinning its sign and scale at the seam.
- `src/systematic_futures/ml/` — point-in-time features, the hand-rolled purged/embargoed splitter,
  the two LightGBM variants.
- `harness.py` / `decision.py` — the comparison set and the pre-declared rule, written once so phases
  cannot drift apart.
- `scripts/` — thin runners; `notebooks/` — explanations. Logic lives in the package, arguments live
  in the runners.
- CI runs the test suite (100 tests) on every push.

## Where the interpretation lives

`docs/findings/memo.md` is the argument: the regime the trend benchmark lived through, why the ML
variants' in-window edge was concentrated in the periods the benchmark lost money, what purge and
embargo bought and what they cannot buy, and the two honest-failure findings. `docs/reading/notes.md`
records the canon behind it — M1–M6 and M8 are read and closed; M7 and the two SHOULD papers are
scaffolded and marked *not read*, and the memo flags the claims that rest on them.

## Reproduce

Prerequisites: Python 3.12, [uv](https://docs.astral.sh/uv/), and network access for the first step
only. The dataset is fetched from the pinned upstream commit and verified against the committed
manifest; re-runs without network work off the frozen local store.

```bash
git checkout results.1
uv sync
uv run python scripts/fetch_raw.py        # writes data/raw/, manifest-gated
uv run python scripts/build_derived.py    # writes data/derived/, manifest-verified
uv run python scripts/build_families.py
uv run python scripts/build_decision.py
uv run python scripts/build_out_of_sample.py # the single out-of-sample read
uv run pytest                             # 100 tests
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/analysis.ipynb
```

The phase scripts write the same tables that are committed. Regeneration is deterministic by
construction: seeded fixture demos, frozen inputs, and committed outputs.

## Repository layout

| path | what it holds |
|---|---|
| `docs/findings/memo.md` | the primary artifact — findings, readings, limitations, non-adopts |
| `notebooks/analysis.ipynb` | the executed walkthrough: context, data, pipeline, methodology, results, readings |
| `docs/methods/` | per-family governance docs: construction, evidence, failure modes, monitoring |
| `docs/report/report.tex`, `report.pdf` | the condensed report |
| `docs/handoff/brief.md` | self-test questions, regulation talking points, non-adopt boundary |
| `results/phase3|4|5/` | committed tables, decision records, derived return series |
| `docs/adr/` | decision reasoning (one accounting path) |
| `docs/data/pst-source.md` | data provenance and licensing posture |
| `.scratch/lab/` | the spec and phase tickets (local tracker, not committed) |

## Boundaries

- **Not claimed**: live trading, order routing, intraday execution, market impact beyond the constant
  cost ladder, and any statement about machine learning on futures in general — the ML result is
  specific to this feature set, label horizon, fold geometry, cost level and window.
- **Deliberately skipped**: deep learning, C++/Rust/kdb+, dashboards, MLOps tooling, microstructure,
  VRP/short-vol, standalone mean-reversion, stat-arb/pairs — each with a stated reason in the memo's
  non-adopt boundary, and regulation talking points in the interview brief rather than in the code.
- **One universe, one window**: 16 roots and a single out-of-sample read; the out-of-time reversal is
  one draw of a regime, not a law.
