"""TSMOM / trend — per-horizon vol-scaled signals, Hurst-Ooi-Pedersen style.

One sleeve per lookback: the trailing-lookback return's sign scaled by
0.40 / trailing annualized vol (the literature's vol-scaled sleeve), clipped
to [-1, 1]. Equal-weight averaging of sleeves is the ensemble (see the
`tsmom` method, ticket 21); the shared engine overlay owns final sizing.
Lookbacks are in trading days (21 per calendar month): 1m = 21, 3m = 63,
12m = 252, sleeves 5/10/20.
"""

from __future__ import annotations

from math import sqrt

import numpy as np
import pandas as pd

from systematic_futures.data.panels import consecutive_returns, trailing_return, trailing_std
from systematic_futures.engine.metrics import ANN

VOL_SCALAR = 0.40  # HOP-style sleeve vol scalar; the ensemble averages sleeves

MONTH = 21  # trading days per calendar month

SLEEVES = (5, 10, 20)  # optional short-term sleeves (ticket 26, ledger +1–2h)

TSMOM_HORIZONS = SLEEVES + (21, 63, 252)  # 5/10/20d sleeves + 1m/3m/12m


def horizon_signal(closes: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """Sleeve signal: sign(trailing `lookback`-day return) * 0.40 / sigma_ann.

    As-of convention: uses closes through t only. Both the trailing return and
    the trailing vol are measured over the symbol's own `lookback` observations
    (`data.panels`), so a quote hole neither shifts the horizon nor breaks the
    window. NaN until the lookback window fills — the engine treats NaN as flat
    (no warmup fabrication).
    """
    past = trailing_return(closes, lookback)
    vol = trailing_std(consecutive_returns(closes), lookback) * sqrt(ANN)
    sig = np.sign(past) * (VOL_SCALAR / vol)
    return sig.clip(-1.0, 1.0)


def tsmom(closes: pd.DataFrame) -> pd.DataFrame:
    """The decision-rule benchmark: equal-weight mean of the vol-scaled
    1/3/12m horizon sleeves. The ML variants (6a defaults, 6b Optuna) must
    beat this on test-window DSR after costs, or they are documented as
    failed challengers.
    """
    sig = sum(horizon_signal(closes, h) for h in TSMOM_HORIZONS)
    return sig / len(TSMOM_HORIZONS)
