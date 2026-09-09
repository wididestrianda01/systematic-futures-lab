"""Back-adjusted continuous contracts — ratio method, backward-cumulative.

At each boundary day r (front flips to its successor), the stitch factor uses the
LAST day the old front trades: f = close(next, r−1) / close(front, r−1). All prices
before r carry the cumulative product of later f's (rolls processed newest→oldest),
so the continuous series' daily return equals the held contract's return through
every roll — no phantom roll gaps, no look-ahead (factors use data up to r−1 only).
"""
from __future__ import annotations

import pandas as pd

CONTINUOUS_COLUMNS = ["date", "symbol", "contract", "open", "high", "low", "close", "volume"]
OHLC = ["open", "high", "low", "close"]


def back_adjust(raw: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """raw: [date, raw_symbol, open, high, low, close, volume]; calendar: [date, symbol, front, next].

    Returns [date, symbol, contract, open, high, low, close, volume] with the held
    contract's OHLC ratio-adjusted; volume untouched.
    """
    parts = []
    for sym, cal in calendar.groupby("symbol"):
        cal = cal.sort_values("date").reset_index(drop=True)
        legs = set(cal["front"]) | set(cal["next"])
        wide = (
            raw.loc[raw["raw_symbol"].isin(legs)]
            .pivot(index="date", columns="raw_symbol", values="close")
            .sort_index()
        )
        factor = pd.Series(1.0, index=pd.DatetimeIndex(cal["date"]))
        cum = 1.0
        boundaries = cal.index[cal["front"].ne(cal["front"].shift()) & (cal.index > 0)]
        for i in reversed(boundaries):
            prev_day = cal.at[i - 1, "date"]
            cum *= wide.at[prev_day, cal.at[i, "front"]] / wide.at[prev_day, cal.at[i - 1, "front"]]
            factor[factor.index < cal.at[i, "date"]] = cum
        held = cal.merge(
            raw, left_on=["date", "front"], right_on=["date", "raw_symbol"], how="inner"
        )
        assert held["date"].tolist() == cal["date"].tolist(), f"{sym}: raw missing held closes"
        out = held[["date", "symbol"] + OHLC + ["volume"]].copy()
        out.insert(2, "contract", held["raw_symbol"].to_numpy())
        out[OHLC] = out[OHLC].mul(factor.to_numpy(), axis=0)
        parts.append(out)
    return pd.concat(parts, ignore_index=True)[CONTINUOUS_COLUMNS].sort_values(
        ["symbol", "date"]
    ).reset_index(drop=True)
