"""MOP (2012) Table 1: published per-instrument statistics, and the free stem each maps to.

Table 1 lists 55 instruments — 24 commodity, 9 equity index, 13 bond, 9 currency.
The paper's text counts 58 ("12 cross-currency pairs (from nine underlying
currencies)"); the table and the text disagree by three currency pairs, and this
module records the table, which is the one that carries numbers.

`stem` is the filename in the pinned pysystemtrade snapshot that supplies the
series, or None where the free store has nothing. Availability *in a window* is
measured from the snapshot at build time, never assumed from `start` — the
published starts are 1965-1998 dates (Datastream/Bloomberg/exchange feeds) and
the free snapshot begins later for almost all of them.
"""

from __future__ import annotations

from dataclasses import dataclass

COM, EQ, FI, FX = "COM", "EQ", "FI", "FX"

#: MOP Table 5, Panel B, "TSMOM Total": mean monthly return of the class TSMOM
#: portfolio, 1985-2009, quarterly-scale factor. The paper's own annualisation is
#: x261 in Eq. (1) and its Sharpe claims are annualised; these monthly means are
#: the level check the replication reads against.
PUBLISHED_MONTHLY = {"ALL": 0.83, "COM": 0.59, "EQ": 1.00, "FI": 1.05, "FX": 0.96}

#: MOP abstract/Table 3: the diversified factor's Sharpe is "greater than one"
#: gross, and the paper states its annualised volatility is 12%/yr over
#: 1985-2009. Both need the paper's 58-instrument breadth; see FINDINGS.
PUBLISHED_DIVERSIFIED_VOL = 12.0


@dataclass(frozen=True)
class Instrument:
    """One MOP Table 1 row: published statistics plus the free-data mapping."""

    name: str
    asset: str
    start: str
    mean: float
    vol: float
    stem: str | None
    why: str = ""


TABLE1: tuple[Instrument, ...] = (
    # Commodity futures (24)
    Instrument("ALUMINUM", COM, "Jan-79", 0.97, 23.50, "ALUMINIUM_LME"),
    Instrument("BRENTOIL", COM, "Apr-89", 13.87, 32.51, "BRENT_W"),
    Instrument("CATTLE", COM, "Jan-65", 4.52, 17.14, "LIVECOW"),
    Instrument("COCOA", COM, "Jan-65", 5.61, 32.38, "COCOA"),
    Instrument("COFFEE", COM, "Mar-74", 5.72, 38.62, "COFFEE"),
    Instrument("COPPER", COM, "Jan-77", 8.90, 27.39, "COPPER"),
    Instrument("CORN", COM, "Jan-65", 3.19, 24.37, "CORN"),
    Instrument("COTTON", COM, "Aug-67", 1.41, 24.35, "COTTON2"),
    Instrument("CRUDE", COM, "Mar-83", 11.61, 34.72, "CRUDE_W"),
    Instrument("GASOIL", COM, "Oct-84", 11.95, 33.18, "GASOIL"),
    Instrument("GOLD", COM, "Dec-69", 5.36, 21.37, "GOLD"),
    Instrument("HEATOIL", COM, "Dec-78", 9.79, 33.78, "HEATOIL"),
    Instrument("HOGS", COM, "Feb-66", 3.39, 26.01, "LEANHOG"),
    Instrument("NATGAS", COM, "Apr-90", 9.74, 53.30, "GAS_US"),
    Instrument("NICKEL", COM, "Jan-93", 12.69, 35.76, "NICKEL_LME"),
    Instrument("PLATINUM", COM, "Jan-92", 13.15, 20.95, None, "no upstream instrument"),
    Instrument("SILVER", COM, "Jan-65", 3.17, 31.11, "SILVER"),
    Instrument("SOYBEANS", COM, "Jan-65", 5.57, 27.26, "SOYBEAN"),
    Instrument("SOYMEAL", COM, "Sep-83", 6.14, 24.59, "SOYMEAL"),
    Instrument("SOYOIL", COM, "Oct-90", 1.07, 25.39, "SOYOIL"),
    Instrument("SUGAR", COM, "Jan-65", 4.44, 42.87, "SUGAR11"),
    Instrument("UNLEADED", COM, "Dec-84", 15.92, 37.36, "GASOILINE"),
    Instrument("WHEAT", COM, "Jan-65", 1.84, 25.11, "WHEAT"),
    Instrument("ZINC", COM, "Jan-91", 1.98, 24.76, "ZINC_LME"),
    # Equity index futures (9)
    Instrument("ASX_SPI200", EQ, "Jan-77", 7.25, 18.33, "SPI200"),
    Instrument("DAX", EQ, "Jan-75", 6.33, 20.41, "DAX"),
    Instrument("IBEX35", EQ, "Jan-80", 9.37, 21.84, "IBEX"),
    Instrument("CAC40", EQ, "Jan-75", 6.73, 20.87, "CAC"),
    Instrument("FTSE_MIB", EQ, "Jun-78", 6.13, 24.59, "MIB"),
    Instrument("TOPIX", EQ, "Jul-76", 2.29, 18.66, "TOPIX"),
    Instrument("AEX", EQ, "Jan-75", 7.72, 19.18, "AEX"),
    Instrument("FTSE100", EQ, "Jan-75", 6.97, 17.77, "FTSE100"),
    Instrument("SP500", EQ, "Jan-65", 3.47, 15.45, "SP500"),
    # Bond futures (13)
    Instrument("AUS_3Y", FI, "Jan-92", 1.34, 2.57, None, "no upstream instrument"),
    Instrument("AUS_10Y", FI, "Dec-85", 3.83, 8.53, None, "no upstream instrument"),
    Instrument("EURO_2Y", FI, "Mar-97", 1.02, 1.53, "SHATZ"),
    Instrument("EURO_5Y", FI, "Jan-93", 2.56, 3.22, "BOBL"),
    Instrument("EURO_10Y", FI, "Dec-79", 2.40, 5.74, "BUND"),
    Instrument("EURO_30Y", FI, "Dec-98", 4.71, 11.70, "BUXL"),
    Instrument("CAN_10Y", FI, "Dec-84", 4.04, 7.36, "CAD10"),
    Instrument("JP_10Y", FI, "Dec-81", 3.66, 5.40, "JGB"),
    Instrument("UK_10Y", FI, "Dec-79", 3.00, 9.12, "GILT"),
    Instrument("US_2Y", FI, "Apr-96", 1.65, 1.86, "US2"),
    Instrument("US_5Y", FI, "Jan-90", 3.17, 4.25, "US5"),
    Instrument("US_10Y", FI, "Dec-79", 3.80, 9.30, "US10"),
    Instrument("US_30Y", FI, "Jan-90", 9.50, 18.56, "US30"),
    # Currency forwards (Table 1 lists 9; the text counts 12 pairs from 9 currencies)
    Instrument("AUD_USD", FX, "Mar-72", 1.85, 10.86, "AUD"),
    Instrument("EUR_USD", FX, "Sep-71", 1.57, 11.21, "EUR"),
    Instrument("CAD_USD", FX, "Mar-72", 0.60, 6.29, "CAD"),
    Instrument("JPY_USD", FX, "Sep-71", 1.35, 11.66, "JPY"),
    Instrument("NOK_USD", FX, "Feb-78", 1.37, 10.56, "NOK"),
    Instrument("NZD_USD", FX, "Feb-78", 2.31, 12.01, "NZD"),
    Instrument("SEK_USD", FX, "Feb-78", 0.05, 11.06, "SEK"),
    Instrument("CHF_USD", FX, "Sep-71", 1.34, 12.33, "CHF"),
    Instrument("GBP_USD", FX, "Sep-71", 1.39, 10.32, "GBP"),
)

BY_NAME: dict[str, Instrument] = {i.name: i for i in TABLE1}
AVAILABLE: dict[str, Instrument] = {i.stem: i for i in TABLE1 if i.stem is not None}
UNAVAILABLE: dict[str, Instrument] = {i.name: i for i in TABLE1 if i.stem is None}

#: MOP's own reporting window (§3.2: "we rely on the sample starting in 1985").
MOP_WINDOW = ("1985-01-01", "2009-12-31")

#: Hurst, Ooi & Pedersen (2017) report a net Sharpe of 0.41 for 2010-2016 — the
#: post-2009 trend drawdown the lab's own out-of-sample window sits inside.
#: Source: docs/reading/notes.md (M2). Different universe (67 markets) and a
#: 10% portfolio vol target, so it is a band, not a value.
HOP_2010_2016 = ("2010-01-01", "2016-12-31", 0.41)
