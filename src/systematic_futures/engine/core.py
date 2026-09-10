"""Shared vectorized backtester — the pipeline seam every method runs through.

One accounting path for all methods: a method produces a daily signal table
(date x symbol, values in [-1, 1]) from the frozen continuous panel; the sizing
overlay turns signals into exposure fractions; `account` turns exposures into
daily returns; `run` assembles the metrics table.

As-of convention (no look-ahead): a signal formed with data through day t
drives the exposure held from t to t+1 — day-t PnL uses exposure(t-1) against
the continuous return R(t), which is the held contract's return including roll
dates (the continuous builder guarantees that; the engine never re-stitches).

Exposure is a fraction of notional capital; daily returns compound into the
equity curve. Costs charge bps per side on traded notional: one unit of
|dE| is one side of full notional, so cost(t) = bps * |dE(t)| * 1e-4, charged
on the trade day. Portfolio return is the equal-capital mean across symbols.
"""

from __future__ import annotations

import pandas as pd

from systematic_futures.engine.metrics import (
    deflated_sharpe,
    max_drawdown,
    sharpe,
    sortino,
    turnover,
)
from systematic_futures.engine.sizing import vol_target_positions


def to_wide(continuous: pd.DataFrame) -> pd.DataFrame:
    """[date, symbol, contract, close] -> wide close panel (date x symbol)."""
    return continuous.pivot(index="date", columns="symbol", values="close").sort_index()


def account(exposure: pd.DataFrame, closes: pd.DataFrame, bps: float = 0.0) -> pd.DataFrame:
    """Daily per-symbol strategy returns from an exposure path.

    r(t) = E(t-1) * R(t) - (bps/1e4) * |E(t) - E(t-1)|. The first day enters
    from flat (|E(t0)| is charged as the entry trade) and earns nothing (no
    prior close to measure a return against).
    """
    rets = closes.pct_change()
    pnl = (exposure.shift(1) * rets).fillna(0.0)
    traded = exposure.diff().abs().fillna(exposure.abs())
    return pnl - bps / 1e4 * traded


def _signals(method, closes: pd.DataFrame) -> pd.DataFrame:
    sig = method(closes) if callable(method) else method
    if not sig.index.equals(closes.index) or not sig.columns.equals(closes.columns):
        raise ValueError("signals must align with the close panel (index and columns)")
    return sig


def run(
    method,
    closes: pd.DataFrame,
    *,
    vol_target: float | None = None,
    cap: float = 1.0,
    vol_lookback: int = 30,
    bps_grid: tuple[float, ...] = (0.0, 2.0, 5.0, 10.0),
    trials: int = 1,
) -> pd.DataFrame:
    """Pipeline seam: run(method, data) -> metrics table.

    method: callable(closes) -> signal table, or a signal table itself.
    vol_target None = raw signals (clipped to caps); otherwise the shared
    annualized-vol overlay sizes every method identically. One row per bps
    level in bps_grid (cost sensitivity, headline default 2 bps). `trials`
    feeds the Deflated Sharpe Ratio column.
    """
    sig = _signals(method, closes)
    if vol_target is None:
        exposure = sig.clip(-cap, cap)
    else:
        exposure = vol_target_positions(sig, closes, vol_target, cap, vol_lookback)
    rows = {}
    for bps in bps_grid:
        r = account(exposure, closes, bps).mean(axis=1)
        rows[bps] = {
            "sharpe": sharpe(r),
            "sortino": sortino(r),
            "max_dd": max_drawdown(r),
            "turnover": turnover(exposure),
            "dsr": deflated_sharpe(r, trials),
        }
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("bps")
