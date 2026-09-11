"""Method families — every family a callable, every call through the same seam.

Convention for adding a family (kept in one place so the comparison stays
apples-to-apples):

- a method is a callable mapping the wide continuous-close panel (date x
  symbol) to a signal table of the same shape, values in [-1, 1];
- sizing (10% annualized vol target + position caps) lives in the shared
  engine overlay, never inside a method;
- each family ships a hand-calculated golden fixture pinning its sign and
  scale at the pipeline seam (tests/test_methods.py).
"""

from systematic_futures.methods.baseline import baseline
from systematic_futures.methods.carry import carry
from systematic_futures.methods.seasonality import month_score, seasonal_tilt, seasonality
from systematic_futures.methods.tsmom import horizon_signal, tsmom
from systematic_futures.methods.xs_momentum import xs_momentum

__all__ = [
    "baseline",
    "carry",
    "horizon_signal",
    "month_score",
    "seasonal_tilt",
    "seasonality",
    "tsmom",
    "xs_momentum",
]
