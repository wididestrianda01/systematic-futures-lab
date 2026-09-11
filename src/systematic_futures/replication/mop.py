"""MOP (2012) Section 2.4 and Eq. (5): ex ante volatility, 12-month sign, 40% sizing.

At time t: s_t^2 = 261 * sum_i (1 - d) d^i (r_{t-1-i} - rbar)^2 with d/(1-d) = 60
(the centre of mass of the weights), rbar the same EWMA of returns, 261
annualising the variance. Position size is sign(r_{t-12,t}) * 40% / s_{t-1},
rebalanced at each month end and held for one month (k = 12, h = 1).

Two deliberate readings, both stated rather than silent:

* The volatility estimate is only used once 60 observations exist. MOP's formula
  is defined from the first observation, but an EWMA with a 60-day centre of mass
  started on three weeks of data is a different object; every instrument here has
  years of history before either reported window, so the choice moves no reported
  number.
* Month-end signal, next-month holding: the weight a month contributes is formed
  from data through that month's last session, which is what the paper's monthly
  rebalance means and what the engine's as-of convention expects.

Rolls: `naive_splice` reproduces the paper's own series — the most liquid
contract's daily price chained by percentage change, so the price gap at a roll
enters the return. `data.continuous.back_adjust` deliberately does not do that
(it measures the held contract's own return, roll days included). The
replication reports both, because the difference is visible in the commodity
volatilities the paper publishes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from systematic_futures.data.panels import consecutive_returns, monthly_to_daily
from systematic_futures.data.roll_calendar import build_roll_calendar, extract_contract_prices
from systematic_futures.engine import account

LOOKBACK_MONTHS = 12
VOL_COM = 60  # centre of mass of the EWMA weights, in days (MOP §2.4)
VOL_SCALE = 261  # MOP's variance annualisation
VOL_TARGET = 0.40  # ex ante annualised volatility per instrument (MOP §4.1)
MIN_VOL_OBS = 60  # observations before the estimate is used (see module docstring)


def ex_ante_vol(
    returns: pd.DataFrame,
    com: int = VOL_COM,
    scale: int = VOL_SCALE,
    min_obs: int = MIN_VOL_OBS,
) -> pd.DataFrame:
    """MOP Eq. (1): annualised EWMA volatility known at each date (returns through t).

    `ewm(com=com, adjust=True)` is the paper's weighting — weights proportional to
    d^i on lag i with d/(1-d) = com — renormalised over the observations available
    at each date, which is what the sum means before its history is long. Callers
    pass each instrument's series from its own start: an EWMA over a slice is
    renormalised over that slice, not over the history before it.
    """
    mean = returns.ewm(com=com, adjust=True).mean()
    variance = (returns - mean) ** 2
    sigma = np.sqrt(scale * variance.ewm(com=com, adjust=True).mean())
    return sigma.where(returns.notna().cumsum() >= min_obs)


def month_end_vol(sigma: pd.DataFrame) -> pd.DataFrame:
    """The volatility estimate at each calendar month's last session, on a month grid."""
    last = sigma.groupby(sigma.index.to_period("M")).tail(1)
    return last.set_index(last.index.to_period("M"))


def mop_signal(closes: pd.DataFrame, lookback: int = LOOKBACK_MONTHS) -> pd.DataFrame:
    """Sign of the trailing `lookback`-month excess return, on a month grid (MOP §3.2)."""
    monthly = closes.groupby(closes.index.to_period("M")).tail(1)
    monthly.index = monthly.index.to_period("M")
    return np.sign(monthly / monthly.shift(lookback) - 1.0)


def mop_weights(
    closes: pd.DataFrame,
    vol_target: float = VOL_TARGET,
    lookback: int = LOOKBACK_MONTHS,
) -> pd.DataFrame:
    """Daily exposure held from t to t+1 — MOP Eq. (5) at k = 12, h = 1.

    The weight is 40% / s_{t-1} with the sign of the past 12 months, formed at each
    month end and held through the next month; NaN where the signal or the
    volatility is not yet available (flat, and excluded from the cross-section).
    """
    sigma = month_end_vol(ex_ante_vol(consecutive_returns(closes)))
    monthly = vol_target / sigma * mop_signal(closes, lookback)
    formed = monthly.copy()
    formed.index = formed.index + 1  # formed at month end m, held through m+1
    return monthly_to_daily(formed, closes)


def naive_splice(mp: pd.DataFrame) -> pd.Series:
    """The paper's own series: the front contract's close each day, chained by pct_change.

    A roll therefore contributes the price difference between the two contracts.
    `data.continuous.back_adjust` is the alternative — the return of the contract
    actually held — and is what every lab result uses.
    """
    mp = mp.copy()
    mp["DATETIME"] = pd.to_datetime(mp["DATETIME"])
    if "symbol" not in mp:
        mp["symbol"] = "X"
    raw = extract_contract_prices(mp)
    calendar = build_roll_calendar(mp)
    held = raw.merge(calendar[["symbol", "date", "front"]], on=["symbol", "date"])
    held = held[held["raw_symbol"] == held["front"]].sort_values(["symbol", "date"])
    return held.set_index("date")["close"].astype(float)


def _exposure(closes: pd.DataFrame, columns: list[str] | None) -> pd.DataFrame:
    return mop_weights(closes if columns is None else closes[list(columns)])


def cross_section(weights: pd.DataFrame) -> pd.DataFrame:
    """Which instruments are in the factor's average each day (MOP's S_t).

    An instrument joins when it first has a position — a signal and a volatility
    estimate, so twelve months of its own history — and stays. A contract that
    has not been listed yet, or has not yet filled its lookback, is absent from
    the average rather than averaged in as a flat zero, which would scale the
    factor by the fraction of the universe that exists.
    """
    return weights.notna().cummax().shift(1, fill_value=False)


def factor_returns(
    closes: pd.DataFrame,
    columns: list[str] | None = None,
    bps: float = 0.0,
) -> pd.Series:
    """MOP's factor: the equal-weighted mean of instrument returns, over the instruments in play.

    The cross-section is MOP's S_t (see `cross_section`); accounting stays in
    `engine.account`, and only the membership rule is decided here.
    """
    weights = _exposure(closes, columns)
    return account(weights, closes, bps).where(cross_section(weights)).mean(axis=1)


def live_instruments(closes: pd.DataFrame, columns: list[str] | None = None) -> pd.Series:
    """How many instruments are in the cross-section each day (`factor_returns`' S_t)."""
    return cross_section(_exposure(closes, columns)).sum(axis=1)
