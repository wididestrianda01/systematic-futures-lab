# this project — Systematic Futures Method-Comparison Lab

One identical, industry-grade backtest harness runs six method families (baseline,
trend/TSMOM, cross-sectional momentum, carry, seasonality — standalone and as a tilt on
trend — and ML with and without tuning) on real futures data, under an honest evaluation
protocol: purged walk-forward CV, a single touched-once OOT window, Deflated Sharpe Ratio,
cost sensitivity. Primary artifact is a findings memo: where each method wins, loses, and why.

Status: the decision read (ML variants through the walk-forward seam) — Phases 0–4 built and green;
the out-of-sample read (memo, condensed report, interview brief, tagged run) outstanding. Market data never
enters this repo — only derived artifacts (code, configs, checksums, signals, stats).

## Layout

- `src/systematic_futures/` — package-first source; `harness.py` defines the comparison set
  and the window contract every phase script runs on
- `scripts/` — thin runners: ingest, derived build, phase tables, sweeps, query demo
- `notebooks/` — thin experiment notebooks over the package
- `results/` — committed derived output (tables, findings, decision docs); no market data
- `docs/reading/` — gated reading canon and per-paper notes
- Tests run on pushes to `main` and on pull requests via GitHub Actions
