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


MONTH_CODES = "FGHJKMNQUVXZ"
CONTRACT_COLUMNS = ["date", "raw_symbol", "open", "high", "low", "close", "volume"]


def make_synthetic_definitions(
    roots: tuple[str, ...] = ("ES",),
    start: str = "2020-01",
    n_months: int = 24,
    roll_lag_bd: int = 10,
) -> pd.DataFrame:
    """Monthly contracts per root: raw symbol {root}{monthcode}{yy},
    expiration = last business day of the month, last_trade_date = expiration − roll_lag_bd."""
    start_period = pd.Period(start, freq="M")
    rows = []
    for root in roots:
        for i in range(-1, n_months):
            p = start_period + i
            month_end = p.end_time.normalize()
            expiration = pd.bdate_range(month_end - pd.Timedelta(days=5), month_end)[-1]
            rows.append(
                {
                    "symbol": root,
                    "raw_symbol": f"{root}{MONTH_CODES[p.month - 1]}{p.year % 100:02d}",
                    "expiration": expiration,
                    "last_trade_date": expiration - pd.tseries.offsets.BDay(roll_lag_bd),
                }
            )
    return pd.DataFrame(rows)


def make_synthetic_contract_prices(
    definitions: pd.DataFrame,
    start: str = "2020-01-01",
    end: str = "2021-12-31",
    seed: int = 42,
) -> pd.DataFrame:
    """Contract-level daily OHLCV: one seeded underlying path per root;
    each contract = underlying × (1 + 2% × years to expiry), so deferreds trade at a premium."""
    days = pd.bdate_range(start, end)
    rows = []
    for root, grp in definitions.groupby("symbol"):
        rng = np.random.default_rng(seed + abs(hash(root)) % 1000)
        level = 100.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, size=len(days))))
        for c in grp.itertuples():
            tte = np.clip((c.expiration - days).days / 365.25, 0, None)
            close = level * (1 + 0.02 * tte)
            spread = np.abs(rng.normal(0.004, 0.002, size=len(days))) + 1e-6
            high = close * (1 + spread)
            low = close * (1 - spread)
            open_ = low + (high - low) * rng.random(len(days))
            volume = rng.lognormal(10.0, 0.5, size=len(days)).astype("int64")
            rows.append(
                pd.DataFrame(
                    {
                        "date": days,
                        "raw_symbol": c.raw_symbol,
                        "open": open_,
                        "high": high,
                        "low": low,
                        "close": close,
                        "volume": volume,
                    }
                )
            )
    return pd.concat(rows, ignore_index=True)[CONTRACT_COLUMNS]
