"""Shared sizing overlay — annualized vol target + position caps.

Every method's signals pass through the same overlay so the comparison stays
apples-to-apples: exposure(t) = sig(t) * vol_target / sigma_ann(t), clipped to
+-cap, where sigma_ann(t) is the trailing `lookback`-observation standard
deviation (ddof=1) of continuous returns annualized — computed from data through
t only (as-of t, matching the engine's no-look-ahead convention). Both the
returns and the window are measured per symbol over that symbol's own
observations (`data.panels`), so a quote hole neither invents a flat day nor
breaks the window. Days without a full trailing window stay flat (exposure 0).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from systematic_futures.data.panels import consecutive_returns, trailing_std
from systematic_futures.engine.metrics import ANN

VOL_TARGET = 0.10  # annualized volatility per symbol
CAP = 1.0  # position cap in notional
LOOKBACK = 30  # trailing observations the realized volatility is measured over


def vol_target_positions(
    sig: pd.DataFrame,
    closes: pd.DataFrame,
    vol_target: float = VOL_TARGET,
    cap: float = CAP,
    lookback: int = LOOKBACK,
) -> pd.DataFrame:
    sd = trailing_std(consecutive_returns(closes), lookback) * np.sqrt(ANN)
    scale = vol_target / sd
    return (sig * scale).clip(-cap, cap).fillna(0.0)
