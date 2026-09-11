"""Shared vectorized backtester — the pipeline seam every method runs through.

One accounting path for all methods: a method produces a daily signal table
(date x symbol, values in [-1, 1]) from the frozen continuous panel; the sizing
overlay turns signals into exposure fractions; `account` turns exposures into
daily returns; `run` assembles the metrics table.

As-of convention (no look-ahead): a signal formed with data through day t
drives the exposure held from t to t+1 — day-t PnL uses exposure(t-1) against
the continuous return R(t), which is the held contract's return including roll
dates (the continuous builder guarantees that; the engine never re-stitches).
Returns are measured per symbol over that symbol's own observations
(`consecutive_returns`), so a root that does not quote a session does not lose
the return that spans it.

Exposure is a fraction of notional capital; daily returns compound into the
equity curve. Costs charge bps per side on traded notional: one unit of
|dE| is one side of full notional, so cost(t) = bps * |dE(t)| * 1e-4, charged
on the trade day. Portfolio return is the equal-capital mean across symbols.
"""

from __future__ import annotations

import pandas as pd

from systematic_futures.data.panels import consecutive_returns, held_positions
from systematic_futures.engine.metrics import (
    deflated_sharpe,
    max_drawdown,
    sharpe,
    sortino,
    traded_notional,
    turnover,
)
from systematic_futures.engine.sizing import vol_target_positions


def account(
    exposure: pd.DataFrame,
    closes: pd.DataFrame,
    bps: float = 0.0,
) -> pd.DataFrame:
    """Daily per-symbol strategy returns from an exposure path.

    r(t) = E(t-1) * R(t) - (bps/1e4) * |E(t) - E(t-1)|, where E is the position
    actually held per symbol (carried across sessions the symbol does not quote —
    `held_positions`) and R(t) is measured over that symbol's own observations
    (`consecutive_returns`). The first day enters from flat (|E(t0)| is charged as
    the entry trade) and earns nothing (no prior close to measure a return against).

    The held path is derived here, never supplied: the trade charged and the trade
    `run` reports turnover on are both `traded_notional(held_positions(...))` of the
    same exposure, so they cannot disagree.
    """
    held = held_positions(exposure, closes)
    pnl = (held.shift(1) * consecutive_returns(closes)).fillna(0.0)
    return pnl - bps / 1e4 * traded_notional(held)


def _signals(method, closes: pd.DataFrame) -> pd.DataFrame:
    sig = method(closes) if callable(method) else method
    if not sig.index.equals(closes.index) or not sig.columns.equals(closes.columns):
        raise ValueError("signals must align with the close panel (index and columns)")
    return sig


def _exposure(
    method,
    closes: pd.DataFrame,
    *,
    vol_target: float | None,
    cap: float,
    vol_lookback: int,
) -> pd.DataFrame:
    """Signals in, exposure fractions out — the one place the overlay is applied.

    `vol_target=None` is the raw-signal path (clipped to caps); otherwise the
    shared annualized-vol overlay sizes every method identically. Private and
    shared by `run` and `curve`, so a plotted curve and a reported metric cannot
    come from different exposure arithmetic.
    """
    sig = _signals(method, closes)
    if vol_target is None:
        return sig.clip(-cap, cap)
    return vol_target_positions(sig, closes, vol_target, cap, vol_lookback)


def curve(
    method,
    closes: pd.DataFrame,
    *,
    vol_target: float | None = None,
    cap: float = 1.0,
    vol_lookback: int = 30,
    bps: float = 0.0,
    dates: pd.DatetimeIndex | None = None,
) -> pd.Series:
    """Daily portfolio return series at one cost level — the same accounting path as `run`.

    The engine owns the overlay, the held path and the cost charge once, so an
    equity curve drawn from this cannot disagree with the Sharpe in the table it
    sits beside. `dates` masks the reported days exactly as `run` does; the
    exposure behind them is still computed on the full panel.
    """
    exposure = _exposure(method, closes, vol_target=vol_target, cap=cap, vol_lookback=vol_lookback)
    returns = account(exposure, closes, bps).mean(axis=1)
    return returns.reindex(dates) if dates is not None else returns


def run(
    method,
    closes: pd.DataFrame,
    *,
    vol_target: float | None = None,
    cap: float = 1.0,
    vol_lookback: int = 30,
    bps_grid: tuple[float, ...] = (0.0, 2.0, 5.0, 10.0),
    trials: int = 1,
    dates: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """Pipeline seam: run(method, data) -> metrics table.

    method: callable(closes) -> signal table, or a signal table itself.
    vol_target None = raw signals (clipped to caps); otherwise the shared
    annualized-vol overlay sizes every method identically. One row per bps
    level in bps_grid (cost sensitivity, headline default 2 bps). `trials`
    feeds the Deflated Sharpe Ratio column.

    `dates` evaluates every metric on those days only, keeping the same
    accounting path — the like-for-like read when one method is flat by
    construction on days another is not (e.g. ML variants outside their folds).
    The exposure and its trailing windows are still computed on the full panel,
    so the window semantics never change with the mask.
    """
    exposure = _exposure(method, closes, vol_target=vol_target, cap=cap, vol_lookback=vol_lookback)
    rows = {}
    held = held_positions(exposure, closes)
    traded = traded_notional(held)
    for bps in bps_grid:
        daily = account(exposure, closes, bps)
        if dates is None:
            charged = traded
        else:
            daily, charged = daily.reindex(dates), traded.reindex(dates)
        r = daily.mean(axis=1)
        rows[bps] = {
            "sharpe": sharpe(r),
            "sortino": sortino(r),
            "max_dd": max_drawdown(r),
            "turnover": turnover(charged),
            "dsr": deflated_sharpe(r, trials),
        }
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("bps")
