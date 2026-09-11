"""Baseline null model — vol-targeted buy-and-hold.

Constant full-long signal per symbol; the shared 10% vol overlay supplies all
sizing and caps. This is the null model every other family must beat to
justify its complexity.
"""

from __future__ import annotations

import pandas as pd


def baseline(closes: pd.DataFrame) -> pd.DataFrame:
    """Signal table: constant +1 per symbol (the overlay does the sizing)."""
    return pd.DataFrame(1.0, index=closes.index, columns=closes.columns)
