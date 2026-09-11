"""Cross-sectional momentum — 12m winners/losers on the futures cross-section.

Each day, rank symbols by their trailing-lookback return and center the ranks
so the strongest sits at +1, the weakest at -1, the median at 0 — a
market-neutral tilt. Ties average their ranks. A symbol with an incomplete
trailing window stays NaN (flat in the engine); it is also excluded from
that day's ranking until its own history fills.
"""

from __future__ import annotations

import pandas as pd

from systematic_futures.data.panels import trailing_return
from systematic_futures.methods.ranking import centered_rank

LOOKBACK = 252  # trading days ≈ 12 months


def xs_momentum(closes: pd.DataFrame, lookback: int = LOOKBACK) -> pd.DataFrame:
    """Centered cross-sectional rank signal in [-1, 1].

    As-of convention: uses closes through t only, with the trailing window
    measured over each symbol's own `lookback` observations.
    """
    return centered_rank(trailing_return(closes, lookback))
