"""Portfolio metrics computed on daily strategy returns (fraction of capital).

Every function takes a daily return series (or a traded-notional panel for
turnover) and returns a float. Conventions are fixed here so all method families
are compared on identical definitions.
"""

from __future__ import annotations

from math import e, sqrt
from statistics import NormalDist

import numpy as np
import pandas as pd

ANN = 252  # trading days per year

_NORM = NormalDist()


def sharpe(returns, periods: int = ANN) -> float:
    """Annualized Sharpe: mean / std(ddof=1) * sqrt(periods)."""
    r = pd.Series(returns).dropna()
    sd = r.std(ddof=1)
    if len(r) < 2 or not np.isfinite(sd) or sd == 0:
        return float("nan")
    return float(r.mean() / sd * np.sqrt(periods))


def sortino(returns, periods: int = ANN) -> float:
    """Annualized mean return over annualized downside deviation (target 0)."""
    r = pd.Series(returns).dropna()
    downside = np.minimum(r.to_numpy(), 0.0)
    dd = float(np.sqrt(np.mean(downside**2)) * np.sqrt(periods))
    if dd == 0:
        return float("nan")
    return float(r.mean() * periods / dd)


def max_drawdown(returns) -> float:
    """Deepest peak-to-trough fall of the compounded equity curve (<= 0)."""
    equity = (1.0 + pd.Series(returns).fillna(0.0)).cumprod()
    return float((equity / equity.cummax() - 1.0).min())


def traded_notional(exposure: pd.DataFrame) -> pd.DataFrame:
    """Per-day traded notional |E(t) - E(t-1)|; first day enters from flat, so |E(t0)| counts as the entry trade."""
    return exposure.diff().abs().fillna(exposure.abs())


def turnover(traded: pd.DataFrame) -> float:
    """Mean daily traded notional from a `traded_notional` panel.

    The same base the cost model charges, so reporting and charging can never
    drift apart (both take the frame `traded_notional` returns).
    """
    return float(traded.sum(axis=1).mean())


def deflated_sharpe(returns, trials: int = 1) -> float:
    """Deflated Sharpe Ratio (Bailey & López de Prado 2014) on daily returns.

    Probability that the observed daily Sharpe beats the expected maximum
    Sharpe of `trials` independent strategies under the same estimator
    variance (kurtosis-adjusted). `trials <= 1` applies no multiple-testing
    deflation (SR0 = 0), so the result is the probabilistic Sharpe ratio.
    """
    r = pd.Series(returns).dropna()
    n = len(r)
    sd = r.std(ddof=1)
    if n < 2 or not np.isfinite(sd) or sd == 0:
        return float("nan")
    sr = r.mean() / sd
    g3 = float(r.skew())
    g4 = float(r.kurt()) + 3.0  # pandas kurt is excess; the formula wants Pearson
    var_unit = 1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr * sr
    if var_unit <= 0:
        return float("nan")
    gamma = 0.5772156649015329  # Euler–Mascheroni constant
    if trials <= 1:
        sr0 = 0.0
    else:
        sr0 = sqrt(var_unit / (n - 1)) * (
            (1.0 - gamma) * _NORM.inv_cdf(1.0 - 1.0 / trials)
            + gamma * _NORM.inv_cdf(1.0 - 1.0 / (trials * e))
        )
    return float(_NORM.cdf((sr - sr0) * sqrt(n - 1) / sqrt(var_unit)))
