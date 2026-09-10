"""Front/deferred basis series — the term-structure slope feeding the carry method."""

from __future__ import annotations

import pandas as pd

BASIS_COLUMNS = ["date", "symbol", "front_close", "next_close", "basis"]


def compute_basis(raw: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """basis = close(next) / close(front) − 1, computed on RAW closes (never the
    adjusted series). Days missing either leg are excluded — never zero-filled."""
    closes = raw[["date", "raw_symbol", "close"]]
    out = calendar.merge(
        closes.rename(columns={"raw_symbol": "front", "close": "front_close"}),
        on=["date", "front"],
    ).merge(
        closes.rename(columns={"raw_symbol": "next", "close": "next_close"}),
        on=["date", "next"],
    )
    out["basis"] = out["next_close"] / out["front_close"] - 1
    return out[BASIS_COLUMNS].sort_values(["symbol", "date"]).reset_index(drop=True)
