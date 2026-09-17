"""Carry — term-structure slope from raw front/deferred prices.

The data layer's basis series is the carry input: basis = close(next)/close(front)
− 1, computed on RAW closes. Backwardation (basis < 0) pays a long a positive
roll carry, so the signal is long a backwardated front, short a contangoed
one — direction only (sign of −basis); the shared overlay owns sizing. Days
with a missing leg carry NaN basis and stay flat, never zero-filled.

The basis panel binds into the returned callable, so the seam
`run(method, closes)` needs no change: the method still maps closes -> signals.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from systematic_futures.data.panels import to_wide


def carry(basis: pd.DataFrame):
    """Bind a basis panel and return a seam-compatible method.

    basis: long [date, symbol, ..., basis] (the data layer's compute_basis output)
    or an already-wide date x symbol basis frame.
    """
    wide = to_wide(basis, "basis") if "basis" in basis.columns else basis

    def method(closes: pd.DataFrame) -> pd.DataFrame:
        aligned = wide.reindex(index=closes.index, columns=closes.columns)
        return np.sign(-aligned)

    return method
