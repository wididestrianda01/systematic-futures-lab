"""MOP (2012) replication — the paper's own universe, signal and sizing.

The lab's 16-root exercise answers "does trend work on data we have". This
package answers the other half of that question: what does the *published*
strategy do on data we are allowed to use, so the lab's levels have something
outside the repo to be checked against.

The seam stays where ADR 0001 put it: returns come from `engine.account`,
metrics from `engine.metrics`, panels from `data`. Only what the paper defines
— the universe, the ex ante volatility, the 12-month sign and the 40%
volatility target — lives here.
"""

from systematic_futures.replication.mop import (
    LOOKBACK_MONTHS,
    VOL_COM,
    VOL_SCALE,
    VOL_TARGET,
    ex_ante_vol,
    factor_returns,
    live_instruments,
    mop_signal,
    mop_weights,
    naive_splice,
)
from systematic_futures.replication.universe import (
    AVAILABLE,
    PUBLISHED_MONTHLY,
    TABLE1,
    UNAVAILABLE,
)

__all__ = [
    "AVAILABLE",
    "LOOKBACK_MONTHS",
    "PUBLISHED_MONTHLY",
    "TABLE1",
    "UNAVAILABLE",
    "VOL_COM",
    "VOL_SCALE",
    "VOL_TARGET",
    "ex_ante_vol",
    "factor_returns",
    "live_instruments",
    "mop_signal",
    "mop_weights",
    "naive_splice",
]
