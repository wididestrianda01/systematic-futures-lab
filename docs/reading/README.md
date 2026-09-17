# Reading canon — gated before interpreted results

MUST canon (8 papers, ~15h) gates interpretation of any backtest result (Phases 3+).
Read in the order below; commit a one-paragraph personal note per paper to
[notes.md](notes.md) as you finish it. 

| Order | ID | Paper | Ref / link | ~h |
|---|---|---|---|---|
| 1 | M8 | The Sharpe Ratio (canonical definition) | Sharpe 1994, JPM 21(1) — [HTML](https://web.stanford.edu/~wfsharpe/art/sr/sr.htm) | 1 |
| 2 | M4 | Facts & Fantasies about Commodity Futures + Strategic/Tactical Value (pair) | Gorton & Rouwenhorst 2006, JF 61(1) — [PDF](https://www.nber.org/system/files/working_papers/w10595/w10595.pdf) · Erb & Harvey 2006, FAJ 62(2) — [PDF](https://people.duke.edu/~charvey/Research/Published_Papers/P91_The_strategic_and.pdf) | 2 |
| 3 | M1 | Time Series Momentum | Moskowitz, Ooi, Pedersen 2012, JFE 104(2) — [PDF](https://pages.stern.nyu.edu/~lpederse/papers/TimeSeriesMomentum.pdf) | 2 |
| 4 | M2 | A Century of Evidence on Trend-Following Investing | Hurst, Ooi, Pedersen 2017, JPM 44(1) — [PDF](https://www.aqr.com/-/media/AQR/Documents/Insights/Journal-Article/AQR-JPM-Fall-2017.pdf) | 1.5 |
| 5 | M3 | Carry | Koijen, Moskowitz, Pedersen, Vrugt 2018, JFE 127(2) — [PDF](https://pages.stern.nyu.edu/~lpederse/papers/Carry.pdf) | 2 |
| 6 | M5 | The Deflated Sharpe Ratio | Bailey & López de Prado 2014, JPM 40(5) — [PDF](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf) | 1.5 |
| 7 | M6 | The Probability of Backtest Overfitting (CSCV) | Bailey, Borwein, López de Prado, Zhu 2017, JCF 20(4) — [PDF](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf) | 2 |
| 8 | M7 | AFML ch. 7, 11–12 (purged/embargoed CV) | López de Prado 2018, Wiley, ISBN 978-1119482086 — [Wiley](https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086) | 3 |

## SHOULD schedule (read while building; each pinned to the part that uses it)

| ID | Paper | Used by | Why there |
|---|---|---|---|
| S8 | Novy-Marx & Velikov 2016 — Taxonomy of Anomalies and Their Trading Costs | the engine | legitimizes the linear cost model before wiring 0/5/10 bps sensitivity |
| S1 | Harvey, Liu, Zhu 2018 — Impact of Volatility Targeting | the baseline | target-level choice for the 10% vol overlay |
| S2 | Moreira & Muir 2017 — Volatility-Managed Portfolios | the baseline | counterparty evidence; prevents parroting one side in the memo |
| S6 | Asness, Frazzini, Pedersen 2012 — Leverage Aversion and Risk Parity | the baseline | portfolio-construction frame for the baseline |
| S7 | Maillard, Roncalli, Teïletche 2010 — ERC portfolios | the baseline | primary source if baseline uses ERC rather than naive vol scaling |
| S3 | Asness, Moskowitz, Pedersen 2013 — Value and Momentum Everywhere | XS momentum | defines the 12m futures XS momentum half of family #3 |
| S4 | Baltas & Kosowski 2013 — Momentum Strategies in Futures | the families | closest prior work; implementation details (signals, turnover) |
| S5 | Daniel & Moskowitz 2016 — Momentum Crashes | the findings | risk framing for the OOT window (2022 → 2024Q1) |
| S9 | Fung & Hsieh 2001 — Trend Followers | the findings | CTA lineage for the memo's industry framing |

Note: no canonical "roll mechanics" paper exists: the canon carries it via M3 + M4.
