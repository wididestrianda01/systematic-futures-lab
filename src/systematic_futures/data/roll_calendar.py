"""Roll calendar from raw leg data — the schedule is OBSERVED, not estimated.

In pysystemtrade's multiple_prices format each day carries the three active legs
with their contract IDs: PRICE (front, what the position holds), FORWARD (the
front's expiry successor), CARRY (the prior contract). The calendar is a
projection of those legs: front flips exactly on observed transitions.
Contract IDs are structured YYYYMM00 and are decoded only through
`contract_month` / `contract_id`, which feed the successor checks — never a
price or a return.
"""

from __future__ import annotations

import pandas as pd

CALENDAR_COLUMNS = ["date", "symbol", "front", "next"]
MP_COLUMNS = [
    "DATETIME",
    "CARRY",
    "CARRY_CONTRACT",
    "PRICE",
    "PRICE_CONTRACT",
    "FORWARD",
    "FORWARD_CONTRACT",
]


def contract_month(contract_id: int | pd.Series) -> int | pd.Series:
    """YYYYMM00 → absolute month index (year-boundary-safe for successor checks).

    One of the two places contract IDs are decoded/encoded; scalar and Series in,
    same out.
    """
    return (contract_id // 10000) * 12 + (contract_id // 100) % 100


def contract_id(period: pd.Period) -> int:
    """Calendar month → YYYYMM00 contract ID (the encoder `contract_month` reads)."""
    return period.year * 10000 + period.month * 100


def build_roll_calendar(mp: pd.DataFrame) -> pd.DataFrame:
    """mp: multiple_prices legs frame (DATETIME + the leg columns), one symbol per frame.

    The schedule is OBSERVED data: each day the position holds PRICE_CONTRACT; FORWARD_CONTRACT
    is its expiry successor. Intraday snapshots are collapsed to the last priced row per day;
    holiday rows (contract recorded, price null) and days with no successor are dropped.
    Returns [date, symbol, front, next] — both legs alive on every calendar day.
    """
    parts = []
    for sym, grp in mp.groupby("symbol"):
        g = (
            grp.sort_values("DATETIME")
            .dropna(subset=["PRICE_CONTRACT", "PRICE"])
            .assign(day=lambda d: d["DATETIME"].dt.normalize())
            .groupby("day", sort=False)
            .tail(1)
        )
        front = g["PRICE_CONTRACT"]
        if not front.is_monotonic_increasing:
            raise ValueError(f"{sym}: front contracts not non-decreasing over time")
        keep = g["FORWARD_CONTRACT"].notna()
        parts.append(
            pd.DataFrame(
                {
                    "date": g.loc[keep, "day"].to_numpy(),
                    "symbol": sym,
                    "front": front[keep].astype("int64").to_numpy(),
                    "next": g.loc[keep, "FORWARD_CONTRACT"].astype("int64").to_numpy(),
                }
            )
        )
    return (
        pd.concat(parts, ignore_index=True)[CALENDAR_COLUMNS]
        .sort_values(["symbol", "date"])
        .reset_index(drop=True)
    )


def extract_contract_prices(mp: pd.DataFrame) -> pd.DataFrame:
    """Melt the three legs into a long per-contract close panel
    [symbol, date, raw_symbol, close] (one row per symbol-contract-day,
    latest snapshot wins). Contract IDs are bare YYYYMM00 month numbers —
    they collide across instruments, so the symbol must ride along."""
    out = []
    for price_col, contract_col in (
        ("PRICE", "PRICE_CONTRACT"),
        ("FORWARD", "FORWARD_CONTRACT"),
        ("CARRY", "CARRY_CONTRACT"),
    ):
        leg = mp[["symbol", "DATETIME", price_col, contract_col]].dropna()
        out.append(
            pd.DataFrame(
                {
                    "symbol": leg["symbol"],
                    "date": leg["DATETIME"].dt.normalize(),
                    "raw_symbol": leg[contract_col].astype("int64"),
                    "close": leg[price_col].astype(float),
                }
            )
        )
    return (
        pd.concat(out)
        .sort_values("date")
        .groupby(["symbol", "date", "raw_symbol"], as_index=False)
        .last()
        .reset_index(drop=True)
    )
