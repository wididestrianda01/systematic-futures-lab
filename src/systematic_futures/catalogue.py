"""The comparison set — the families this project compares, under one protocol.

The families read, the decision read and the out-of-sample read must not drift: the same families,
the same window, the same table shape.
This is the one place the set is stated, so a phase adds a family here instead of restating it, and the
walk-forward entry point lives beside it: the schedule the decision was taken under and the schedule
the out-of-sample read extends are one call, asserted leak-free.

This module is where the ML stack enters the import graph. Nothing that only needs the comparison
table's shape imports it (`comparison`).
"""

from __future__ import annotations

import pandas as pd

from systematic_futures.methods import (
    baseline,
    carry,
    seasonal_tilt,
    seasonality,
    tsmom,
    xs_momentum,
)
from systematic_futures.ml import (
    Fold,
    leakage_violations,
    lgbm_defaults,
    lgbm_tuned,
    purged_walk_forward,
)
from systematic_futures.protocol import PROTOCOL, Method, Protocol

BENCHMARK = "tsmom"
ML_VARIANTS = ("ml_defaults", "ml_tuned")
LABELS = {"ml_defaults": "6a", "ml_tuned": "6b"}


def classic_set(basis: pd.DataFrame) -> dict[str, Method]:
    """The classic families: baseline, the trend benchmark, XS momentum, carry
    (bound to the basis panel), seasonality and its tilt on trend."""
    return {
        "baseline": Method(baseline),
        BENCHMARK: Method(tsmom),  # six-horizon benchmark (short sleeves adopted)
        "xs_momentum": Method(xs_momentum),
        "carry": Method(carry(basis)),
        "seasonality": Method(seasonality),
        "seasonal_tilt": Method(seasonal_tilt),
    }


def ml_set(basis: pd.DataFrame, *, protocol: Protocol = PROTOCOL) -> dict[str, Method]:
    """Variants 6a/6b, both bound to the one walk-forward protocol."""
    return {
        "ml_defaults": lgbm_defaults(basis, protocol=protocol),
        "ml_tuned": lgbm_tuned(basis, protocol=protocol),
    }


def full_set(basis: pd.DataFrame, *, protocol: Protocol = PROTOCOL) -> dict[str, Method]:
    """The whole comparison set: the classics plus 6a/6b, one protocol for both phases."""
    return {**classic_set(basis), **ml_set(basis, protocol=protocol)}


def walk_forward_folds(index: pd.DatetimeIndex, *, protocol: Protocol = PROTOCOL) -> list[Fold]:
    """The one walk-forward schedule, purged and embargoed, asserted leak-free.

    The decision read runs it over develop+validate, the out-of-sample read over the whole frozen
    panel so the fold covering 2022 trains entirely inside develop+validate; both use this call, so the
    schedule cannot differ between them.
    """
    folds = purged_walk_forward(
        index, n_splits=protocol.splits, horizon=protocol.horizon, embargo=protocol.embargo
    )
    problems = leakage_violations(folds, index, horizon=protocol.horizon, embargo=protocol.embargo)
    assert not problems, f"splitter leaks: {problems[:3]}"
    return folds


__all__ = [
    "BENCHMARK",
    "LABELS",
    "ML_VARIANTS",
    "classic_set",
    "full_set",
    "ml_set",
    "walk_forward_folds",
]
