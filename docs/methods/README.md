# Method research docs

One document per family in the comparison set: what the signal is and where it comes from, how it is
constructed here, what it cost to trade, what the committed evidence shows on both evaluation
windows, how it fails, and what would trigger a retest. These are the model-governance artifacts of
the lab — the document a desk keeps for a live signal, written for the signals this project ran.

Shared machinery, stated here once and true of every doc below:

- **Overlay**: every family's signal is scaled to a **10% annualized volatility target** by
  `engine.sizing.vol_target_positions` using trailing 30-session realized vol computed as-of t, then
  clipped to ±1 notional per symbol. No family carries its own sizing.
- **Accounting**: every family crosses `engine.run` — the one pipeline seam (ADR 0001). `run`
  reports Sharpe, Sortino, max drawdown, turnover and the Deflated Sharpe Ratio at 0/2/5/10 bps per
  side on traded notional.
- **Headline cost level**: 2 bps/side — the level a decision is read at.
- **Windows**: develop+validate 2010-01-01 → 2021-12-31 (the develop+validate tables); out-of-sample
  2022-01-03 → 2024-03-28, read once by `scripts/build_out_of_sample.py` (the out-of-sample tables).
- **Evidence provenance**: `results/families/tables.csv`, `results/decision/tables.csv`,
  `results/out_of_sample/tables.csv`. The tables are frozen; the numbers below are read from them, never
  recomputed here.

| family | doc | rule source |
|---|---|---|
| Baseline (null model) | [baseline.md](baseline.md) | M4 — Gorton & Rouwenhorst; Erb & Harvey |
| TSMOM / trend | [tsmom.md](tsmom.md) | M1, M2, S4 |
| Cross-sectional momentum | [xs_momentum.md](xs_momentum.md) | S3 — Asness, Moskowitz & Pedersen |
| Carry | [carry.md](carry.md) | M3 — Koijen, Moskowitz, Pedersen & Vrugt |
| Seasonality (standalone + tilt) | [seasonality.md](seasonality.md) | S4 — Baltas & Kosowski; spec amendment |
| ML variants 6a / 6b | [ml_variants.md](ml_variants.md) | M5, M6, M7 |

Interpretation — why a family behaved as it did, and what the out-of-sample reversal means — belongs
to the findings memo (ticket 38) behind the reading canon, not to these docs. Where something is
still open, the doc says so rather than reaching for an explanation.
