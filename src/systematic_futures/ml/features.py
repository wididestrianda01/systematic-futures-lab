"""ML feature panel — point-in-time features built from the panels the lab owns.

Every value at (t, symbol) is computed from closes through t only: trailing
returns, realized vol and a vol ratio, distance from a moving average, the
cross-sectional ranks of momentum and basis, the the data layer basis level (carry),
and the same-calendar-month seasonality score the seasonality family uses.
The table is tidy — one row per (date, symbol), one column per feature — which
is the shape a pooled model consumes directly.

`forward_label` is the one deliberately forward-looking object: it is the
training target, never a model input.

Macro-series features are a documented non-adopt: the free-only data posture
has no redistributable point-in-time macro source, so the ML variants learn
from the panel this repo already owns (the out-of-sample read carries the line into the
non-adopt boundary).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from systematic_futures.data.panels import (
    consecutive_returns,
    forward_return,
    monthly_to_daily,
    trailing_mean,
    trailing_return,
    trailing_std,
)
from systematic_futures.engine.metrics import ANN
from systematic_futures.methods.ranking import centered_rank
from systematic_futures.methods.seasonality import month_score

RETURN_WINDOWS = (5, 21, 63, 126, 252)
VOL_WINDOWS = (21, 63)
MA_WINDOWS = (21, 63)
SEASON_WINDOW = 60
LABEL_VOL_WINDOW = 63

FEATURES = [
    *(f"ret_{w}" for w in RETURN_WINDOWS),
    *(f"vol_{w}" for w in VOL_WINDOWS),
    "vol_ratio",
    *(f"ma_dist_{w}" for w in MA_WINDOWS),
    "mom_rank",
    "basis",
    "basis_rank",
    "season",
]

_INDEX_NAMES = ["date", "symbol"]


def feature_panel(
    closes: pd.DataFrame, basis: pd.DataFrame | None = None, *, season_window: int = SEASON_WINDOW
) -> pd.DataFrame:
    """Tidy (date, symbol) x FEATURES table; NaN where a feature's window is not full.

    basis: wide date x symbol basis panel (the data layer). Omitted or misaligned, the
    two basis features are NaN and those samples drop out of training rather
    than being zero-filled. Every return and trailing window is measured over
    the symbol's own observations (`data.panels`), so the panel's union calendar
    — staggered history starts and quote holes — neither shifts a horizon nor
    breaks a window.
    """
    rets = consecutive_returns(closes)
    vol = {w: trailing_std(rets, w) * np.sqrt(ANN) for w in VOL_WINDOWS}
    basis_wide = (
        basis.reindex(index=closes.index, columns=closes.columns)
        if basis is not None
        else pd.DataFrame(np.nan, index=closes.index, columns=closes.columns)
    )
    cols = {f"ret_{w}": trailing_return(closes, w) for w in RETURN_WINDOWS}
    cols.update({f"vol_{w}": v for w, v in vol.items()})
    cols["vol_ratio"] = vol[21] / vol[63]
    cols.update({f"ma_dist_{w}": closes / trailing_mean(closes, w) - 1.0 for w in MA_WINDOWS})
    cols["mom_rank"] = centered_rank(cols["ret_252"])
    cols["basis"] = basis_wide
    cols["basis_rank"] = centered_rank(basis_wide)
    cols["season"] = monthly_to_daily(month_score(closes, window=season_window), closes)

    tidy = pd.concat({name: cols[name].stack() for name in FEATURES}, axis=1)
    return tidy.rename_axis(_INDEX_NAMES)


def forward_label(
    closes: pd.DataFrame, horizon: int = 5, vol_window: int = LABEL_VOL_WINDOW
) -> pd.Series:
    """Training target: forward h-day return scaled by trailing vol as-of t.

    Long (date, symbol) series. Both sides are measured over the symbol's own
    observations — the forward leg is the symbol's next h observations, the
    scaling is its trailing vol through t — which keeps one pooled loss
    comparable across the cross-section and across ragged histories. Uses closes
    (t, t+h] and a trailing window ending at t: deliberately forward-looking, and
    only ever fed to a fit.
    """
    unit = trailing_std(consecutive_returns(closes), vol_window) * np.sqrt(horizon)
    scaled = forward_return(closes, horizon) / unit
    return scaled.stack().rename("label").rename_axis(_INDEX_NAMES)
