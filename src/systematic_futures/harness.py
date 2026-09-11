"""The comparison harness both phase scripts share — the comparison set, written once.

the families read and the decision read must not drift: the same families, the same window, the same
table shape. The window contract (`data.panels.develop_validate`) and the
accounting path (`engine.run`) live in the package; this module is the one place
the set of compared families and the table-assembly shape are stated, so a phase
script adds families to it instead of restating it.
"""

from __future__ import annotations

import pandas as pd

from systematic_futures.engine import run
from systematic_futures.methods import (
    baseline,
    carry,
    seasonal_tilt,
    seasonality,
    tsmom,
    xs_momentum,
)

BENCHMARK = "tsmom"
HEADLINE_BPS = 2.0
VOL_TARGET = 0.10  # the shared sizing overlay: every family runs at 10% annualized vol


def classic_set(basis: pd.DataFrame) -> dict[str, object]:
    """The classic families: baseline, the trend benchmark, XS momentum, carry
    (bound to the basis panel), seasonality and its tilt on trend."""
    return {
        "baseline": baseline,
        BENCHMARK: tsmom,  # six-horizon benchmark (sleeves adopted, ticket 26)
        "xs_momentum": xs_momentum,
        "carry": carry(basis),
        "seasonality": seasonality,
        "seasonal_tilt": seasonal_tilt,
    }


def table_for(
    methods: dict[str, object],
    closes: pd.DataFrame,
    *,
    trials: dict[str, int] | None = None,
    dates: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """Every method through the identical seam — one row per method per bps level.

    `trials` feeds each method's DSR deflation (absent = 1). `dates` is the
    engine's like-for-like mask, applied identically to every method.
    """
    records = []
    for name, method in methods.items():
        table = run(
            method,
            closes,
            vol_target=VOL_TARGET,
            trials=(trials or {}).get(name, 1),
            dates=dates,
        )
        table["method"] = name
        records.append(table.reset_index().rename(columns={"index": "bps"}))
    return pd.concat(records, ignore_index=True)
