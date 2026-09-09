"""Synthetic fixtures — deterministic, seeded; never real market data."""
from __future__ import annotations

import numpy as np
import pandas as pd

OHLCV_COLUMNS = ["date", "symbol", "open", "high", "low", "close", "volume"]
MP_COLUMNS = [
    "DATETIME", "CARRY", "CARRY_CONTRACT", "PRICE", "PRICE_CONTRACT",
    "FORWARD", "FORWARD_CONTRACT",
]


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


def make_synthetic_multiple_prices(
    symbols: tuple[str, ...] = ("ES",),
    start: str = "2020-01-01",
    end: str = "2021-06-30",
    premium: float = 0.02,
    seed: int = 42,
    intraday: bool = True,
) -> pd.DataFrame:
    """multiple_prices-format fixture: monthly contract IDs (YYYYMM00), three legs.

    Front rolls on the first business day on/after the 18th (an observed
    transition); carry = the prior month's contract, forward = the next month's.
    Deferred legs trade at a decaying premium vs the underlying path. An
    intraday duplicate (23:00) is appended per day when intraday=True, plus one
    leading legacy row with a null front — both exercise the calendar builder.
    """
    days = pd.bdate_range(start, end)
    first_month = pd.Period(start, freq="M") - 1
    rows = []
    for sym in symbols:
        rng = np.random.default_rng(seed + abs(hash(sym)) % 1000)
        level = 100.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, size=len(days))))
        # leading legacy row: carry contract only, no front (like upstream's 1982 rows)
        rows.append(
            {
                "DATETIME": days[0] - pd.Timedelta(days=1) + pd.Timedelta(hours=23),
                "CARRY": None, "CARRY_CONTRACT": first_month.year * 10000 + first_month.month * 100,
                "PRICE": None, "PRICE_CONTRACT": None,
                "FORWARD": None, "FORWARD_CONTRACT": None,
                "symbol": sym,
            }
        )
        for d, lv in zip(days, level):
            m = d.to_period("M")
            cutoff = pd.bdate_range(d.replace(day=18), d.replace(day=22))[0]
            front = m if d < cutoff else m + 1
            ids = {
                "CARRY": (front - 1).year * 10000 + (front - 1).month * 100,
                "PRICE": front.year * 10000 + front.month * 100,
                "FORWARD": (front + 1).year * 10000 + (front + 1).month * 100,
            }
            prices = {}
            for leg_name, cid in ids.items():
                c_period = pd.Period(year=cid // 10000, month=(cid // 100) % 100, freq="M")
                mte = max((c_period.end_time - d).days / 30.44, 0.0)
                prices[leg_name] = round(float(lv) * (1 + premium * mte), 4)
            for hours in ((20,) if not intraday else (20, 23)):
                rows.append(
                    {
                        "DATETIME": d + pd.Timedelta(hours=int(hours)),
                        "CARRY": prices["CARRY"], "CARRY_CONTRACT": ids["CARRY"],
                        "PRICE": prices["PRICE"], "PRICE_CONTRACT": ids["PRICE"],
                        "FORWARD": prices["FORWARD"], "FORWARD_CONTRACT": ids["FORWARD"],
                        "symbol": sym,
                    }
                )
    return pd.DataFrame(rows)[["symbol"] + MP_COLUMNS]
