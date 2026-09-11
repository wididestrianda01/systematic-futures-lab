"""Seasonality — same-calendar-month trailing mean as a signal.

`month_score`: per instrument, the mean daily return earned on the same
calendar month over the trailing `window` months, using only PRIOR years'
same-month returns (the current, incomplete month never enters the score —
as-of safe). NaN until `min_years` prior same-months exist inside the window.

The standalone method signs the score: long a historically strong month,
short a weak one, flat elsewhere. The shared overlay owns sizing; the
tilt-on-trend variant (ticket 25) reuses the raw score.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from systematic_futures.data.panels import consecutive_returns, monthly_to_daily
from systematic_futures.methods.tsmom import tsmom

WINDOW = 60  # trailing months of same-calendar-month history
MIN_YEARS = 3  # priors required before a score exists
TILT = 0.5  # favorable month amplifies the trend position by 1 + TILT


def month_score(
    closes: pd.DataFrame, window: int = WINDOW, min_years: int = MIN_YEARS
) -> pd.DataFrame:
    """Monthly-grid score: same-calendar-month mean daily return from prior
    years only, NaN until `min_years` priors fall inside the window.

    Daily returns are per symbol over that symbol's own observations
    (`data.panels`), and the monthly mean ignores the days a root did not
    quote rather than treating them as zero-return days.
    """
    rets = consecutive_returns(closes)
    monthly = rets.groupby(rets.index.to_period("M")).mean()
    lags = range(12, window + 1, 12)
    arr = np.stack([monthly.shift(k).to_numpy() for k in lags])  # (lag, month, symbol)
    priors = (~np.isnan(arr)).sum(axis=0)
    mean = np.divide(
        np.nansum(arr, axis=0), priors, out=np.full(arr.shape[1:], np.nan), where=priors > 0
    )
    mean[priors < min_years] = np.nan
    return pd.DataFrame(mean, index=monthly.index, columns=monthly.columns)


def seasonality(closes: pd.DataFrame) -> pd.DataFrame:
    """Standalone long-short seasonality: sign of the month score.

    As-of convention: a date's score uses only strictly prior same-months,
    so no current-month data ever informs the position.
    """
    score = month_score(closes)
    return np.sign(monthly_to_daily(score, closes))


def seasonal_tilt(closes: pd.DataFrame) -> pd.DataFrame:
    """Seasonality as a filter/tilt on the TSMOM ensemble (the spec's
    intended production use): positions scale by 1 + TILT·sign(score) where
    a score exists — favorable month amplifies, unfavorable dampens — and
    stay untouched where seasonality data is still warming up. Clipped back
    into [-1, 1]; the shared overlay owns sizing.
    """
    trend = tsmom(closes)
    score = month_score(closes)
    per_day = monthly_to_daily(score, closes)
    mult = (1.0 + TILT * np.sign(per_day)).where(per_day.notna(), 1.0)
    return (trend * mult).clip(-1.0, 1.0)
