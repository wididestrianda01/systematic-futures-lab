# MOP (2012) replication on free data — what the paper's construction does here

Run: `uv run python scripts/fetch_replication.py && uv run python scripts/build_mop_replication.py`
Data: the pinned `pysystemtrade` snapshot (`b4a25e6`), 52 of MOP Table 1's 55 instruments
mapped, 46 with a usable continuous series. Artifacts: `table1_vol.csv`, `tsmom.csv` (this file
quotes them; nothing here is written by hand that the CSVs do not carry).

## Why this exists

The lab's numbers had no external level check: the tables could be reproduced and the DSR
re-derived while the *size* of every Sharpe remained unverified (issue 51). MOP's own universe
cannot be rebuilt on free data — the snapshot has no LME metals before 2023, no Bund before 2006,
no DAX before 2000, no CAC/AEX before 2009, no euro before 1999 — so this replicates as much of
the paper as the data allows and states the breadth it is missing.

## Check 1 — the data layer against published per-instrument statistics

`table1_vol.csv`: annualised volatility per instrument against MOP Table 1, both constructions.
The paper's table spans 1965/start → 2009; the free series start later, and the ratio drifts with
that gap — mean `|ratio − 1|` runs 0.11 for series that overlap the paper's sample, 0.16 for those
starting 16–25 years late and 0.20 beyond 25 years (rank correlation 0.35, so the level is noisy
instrument by instrument while the widening is plain in the groups below).

| group | instruments | vol ratio ours/published |
|---|---|---|
| free history overlapping the paper's sample | CORN, WHEAT, SUGAR, CATTLE, GOLD, SILVER, SP500, US_10Y, GBP_USD, JPY_USD, CHF_USD, CAD_USD | 0.71 – 1.06 (median 0.94) |
| free history starting after 1990 | CRUDE, NATGAS, COPPER, JP_10Y, NOK_USD, NZD_USD, SEK_USD, IBEX35, ASX_SPI200, FTSE_MIB … | 0.46 – 1.12 |
| free history starting after 2015 | EURO_2Y, EURO_10Y, EURO_30Y, SOYOIL, SOYMEAL, FTSE100 | 0.62 – 1.33 |

Median across all 46: **0.94** (adjusted) and **0.94** (naive splice). Twelve instruments with
overlapping history land within ±30% of published — for a different vendor, different contract
selection and a five-decade window mismatch, that is the data layer agreeing with the paper.

The construction shows up where it should: the paper chains the most liquid contract's daily
return, so the price gap at each roll enters its series. Our ratio back-adjustment excludes it.
For high-basis commodities the naive splice moves the estimate toward the published number
(CORN 0.98 → 1.07, GOLD 0.87 → 0.87, SILVER 0.89 → 1.04, CATTLE 0.80 → 0.90), and for
FX/bond/equity futures, whose bases are near zero, it changes nothing (SP500 0.99 → 0.99,
GBP 0.98 → 0.98).

## Check 2 — the factor, 1985–2009 (the paper's window)

`tsmom.csv`, `construction = adjusted` (roll-adjusted, the lab's construction) and `naive`
(the paper's splice). Published column is MOP Table 5 Panel B, monthly mean of the class portfolio.

| portfolio | n (free) | mean %/mo adjusted | mean %/mo naive | published %/mo | SR adjusted | SR naive |
|---|---|---|---|---|---|---|
| ALL | 31 | 1.454 | 0.812 | 0.83 | **1.20** | 0.74 |
| COM | 12 | 1.317 | 0.500 | 0.59 | 0.84 | 0.37 |
| FX | 9 | 1.458 | 1.269 | 0.96 | 0.66 | 0.57 |
| FI | 7 | 1.662 | 0.784 | 1.05 | 0.62 | 0.29 |
| EQ | 3 | 1.875 | 2.028 | 1.00 | 0.59 | 0.67 |

What lands:

- **The paper's risk and Sharpe claims reproduce.** The diversified factor comes out at 13.7%
  annualised volatility (the paper states 12%) with a gross Sharpe of 1.20 (the paper claims
  "greater than one"). Under the paper's own splice the same factor is 13.1% vol and SR 0.74.
- **The commodity class level is a construction artefact, not a data disagreement.** With the
  paper's splice the commodity portfolio returns 0.500%/mo against the published 0.59 — 15%
  apart. With the roll-adjusted series it is 1.317%/mo, 2.2× the published level. The residual
  disagreement in the one class where the free data has real breadth is therefore mostly the
  choice of splice, and the paper's own number embeds it.
- **Equity and bond classes are not comparable here**: 3 and 7 instruments against the paper's 9
  and 13, so their class portfolios carry 33–36% volatility and their means come out on the high
  side. They are reported, not read.

Costs barely bite: at 2 bps the diversified Sharpe moves 1.20 → 1.19 over the paper's window.
The construction rebalances monthly and turns over slowly, which is why the lab's 2 bps matters
much more on its own daily-updated overlay than it does here.

## Check 3 — the post-2009 regime, 2010–2016

Hurst, Ooi & Pedersen (2017) report a *net* Sharpe of 0.41 for the trend factor over 2010–2016,
the drawdown the lab's out-of-sample window sits inside. Same construction, free data:

| window | construction | n | SR gross | SR at 2 bps |
|---|---|---|---|---|
| 2010–2016 | adjusted | 34 | **0.58** | 0.56 |
| 2010–2016 | naive | 46 | 0.14 | 0.11 |
| 2010–2024Q1 | adjusted | 46 | 0.46 | 0.43 |

Both constructions bracket the published 0.41, and both are negative-to-flat in equity and FX
over that window — the same shape the lab's own out-of-sample read found. The lab's weak OOT
trend numbers are the published regime, not an implementation failure.

## What this cannot say

- The paper's *exact* class means, its 58-instrument diversified factor, and its per-instrument
  Sharpe figure are not reproducible free: the missing breadth is LME metals, pre-1999 euro,
  pre-2006 Bund and pre-2000 European equity futures, and a smaller universe is a different
  factor, not a measurement error. Those stay declared-unverifiable; no band was invented for them.
- The naive splice approximates the paper's contract choice (the repo's roll calendar defines the
  front leg; the paper takes the most liquid contract) — it is a direction, not a bit-exact
  reproduction of their series.
- Both constructions share the free vendor's data, so a vendor-wide bias would move both.
