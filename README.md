# this project — Systematic Futures Method-Comparison Lab

One identical, industry-grade backtest harness runs four method families (baseline,
trend/TSMOM, cross-sectional momentum + carry, ML with and without tuning) on real
futures data, under an honest evaluation protocol: purged walk-forward CV, a single
touched-once OOT window, Deflated Sharpe Ratio, cost sensitivity. Primary artifact is
a findings memo: where each method wins, loses, and why.

Status: the scaffold (foundations). Market data never enters this repo — only derived
artifacts (code, configs, checksums, signals, stats).

## Layout

- `src/systematic_futures/` — package-first source
- `docs/reading/` — gated reading canon and per-paper notes
- Tests run on every push via GitHub Actions
