"""Cross-sectional momentum — 12m winners/losers on the futures cross-section.

Each day, rank symbols by their trailing-lookback return and center the ranks
so the strongest sits at +1, the weakest at -1, the median at 0 — a
market-neutral tilt. Ties average their ranks. A symbol with an incomplete
trailing window stays NaN (flat in the engine); it is also excluded from
that day's ranking until its own history fills.
"""

from __future__ import annotations

import pandas as pd

LOOKBACK = 252  # trading days ≈ 12 months


def xs_momentum(closes: pd.DataFrame, lookback: int = LOOKBACK) -> pd.DataFrame:
    """Centered cross-sectional rank signal in [-1, 1].

    As-of convention: uses closes through t only.
    """
    past = closes.pct_change(lookback)
    rank = past.rank(axis=1)  # average ranks on ties; NaN symbols skipped
    n = past.notna().sum(axis=1)
    center = rank.sub(n.add(1.0).div(2.0), axis=0).div(n.sub(1.0).div(2.0), axis=0)
    return center
