"""Roll calendar: front/next contract schedule from definition expiries.

Rule (single knob): roll_date(c) = last_trade_date(c) − n_bd business days; on that day
the front becomes c's expiry successor. Symbol strings are never parsed for logic —
expiry knowledge comes only from definition data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CALENDAR_COLUMNS = ["date", "symbol", "front", "next"]


def build_roll_calendar(
    definitions: pd.DataFrame, n_bd: int = 5, start: str | None = None
) -> pd.DataFrame:
    """definitions: [symbol, raw_symbol, expiration, last_trade_date], one row per contract.

    Returns [date, symbol, front, next] for business days from `start` (default: the
    first roll date) through the last roll date, keeping only days where a next
    contract exists — both legs are alive on every calendar day by construction
    (roll < last trade date < expiration).
    """
    parts = []
    for sym, grp in definitions.groupby("symbol"):
        g = grp.sort_values("expiration").reset_index(drop=True)
        rolls = (g["last_trade_date"] - pd.tseries.offsets.BDay(n_bd)).dt.normalize()
        if not rolls.is_monotonic_increasing:
            raise ValueError(f"{sym}: roll dates not strictly increasing")
        grid = pd.bdate_range(start or rolls.iloc[0], rolls.iloc[-1])
        idx = np.searchsorted(rolls.to_numpy(), grid.to_numpy(), side="right")
        keep = idx + 1 < len(g)  # next = front's expiry successor must exist
        idx, days = idx[keep], grid[keep]
        symbols = g["raw_symbol"].to_numpy()
        parts.append(
            pd.DataFrame({"date": days, "symbol": sym, "front": symbols[idx], "next": symbols[idx + 1]})
        )
    return pd.concat(parts, ignore_index=True)[CALENDAR_COLUMNS].sort_values(
        ["symbol", "date"]
    ).reset_index(drop=True)
