"""Synthetic OHLCV fixtures — deterministic, seeded; never real market data."""
from __future__ import annotations

import numpy as np
import pandas as pd

OHLCV_COLUMNS = ["date", "symbol", "open", "high", "low", "close", "volume"]


def make_synthetic_ohlcv(
    symbols: tuple[str, ...] = ("AAA", "BBB", "CCC"),
    start: str = "2020-01-01",
    end: str = "2020-12-31",
    seed: int = 42,
    start_price: float = 100.0,
) -> pd.DataFrame:
    """Seeded daily OHLCV panel: one geometric random walk per symbol.

    Determinism contract: same arguments → byte-identical DataFrame.
    """
    dates = pd.bdate_range(start, end)
    rng = np.random.default_rng(seed)
    frames = []
    for i, symbol in enumerate(symbols):
        rets = rng.normal(0.0003, 0.01, size=len(dates))
        close = start_price * (1 + i * 0.5) * np.exp(np.cumsum(rets))
        spread = np.abs(rng.normal(0.004, 0.002, size=len(dates))) + 1e-6
        high = close * (1 + spread)
        low = close * (1 - spread)
        open_ = low + (high - low) * rng.random(len(dates))
        volume = rng.lognormal(10.0, 0.5, size=len(dates)).astype("int64")
        frames.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "symbol": symbol,
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)[OHLCV_COLUMNS]
