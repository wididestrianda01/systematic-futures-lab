"""The comparison set, and the protocol it is measured under."""

from __future__ import annotations

import pandas as pd
from conftest import continuous_wide

from systematic_futures.catalogue import (
    BENCHMARK,
    LABELS,
    ML_VARIANTS,
    classic_set,
    full_set,
    walk_forward_folds,
)
from systematic_futures.ml import leakage_violations
from systematic_futures.protocol import PROTOCOL, Protocol


def basis_for(closes: pd.DataFrame) -> pd.DataFrame:
    """Stand-in carry panel: a trailing return, never forward-looking."""
    return closes.pct_change(21)


def test_classic_set_binds_the_basis_panel_it_is_given():
    closes = continuous_wide(("ES", "GC"))
    up = classic_set(basis_for(closes))["carry"].signal(closes)
    down = classic_set(-basis_for(closes))["carry"].signal(closes)
    assert up.notna().any().any()  # the bound family does read the panel it was handed
    pd.testing.assert_frame_equal(down, -up)


def test_the_full_set_is_the_classics_plus_the_named_variants():
    closes = continuous_wide(("ES", "GC"))
    every = full_set(basis_for(closes))
    comparison_set = classic_set(basis_for(closes))
    assert set(every) == set(comparison_set) | set(ML_VARIANTS)
    assert BENCHMARK in comparison_set
    assert set(LABELS) == set(ML_VARIANTS)


def test_every_family_declares_the_trials_its_selection_evaluated():
    """The DSR's deflation input is part of the contract, not an attribute a family may forget."""
    closes = continuous_wide(("ES", "GC"))
    tiny = Protocol(n_trials=2, splits=3)
    every = full_set(basis_for(closes), protocol=tiny)
    assert every["ml_defaults"].trials == 1  # searched nothing
    assert every["ml_tuned"].trials == tiny.splits * tiny.n_trials
    assert all(method.trials >= 1 for method in every.values())
    assert full_set(basis_for(closes))["ml_tuned"].trials == (PROTOCOL.splits * PROTOCOL.n_trials)


def test_the_declared_protocol_is_the_schedule_the_folds_come_from():
    """The folds are the protocol's geometry, purged and embargoed — not a splitter default."""
    index = pd.bdate_range("2010-01-01", "2021-12-31")
    protocol = Protocol(horizon=5, splits=3, embargo=10)
    folds = walk_forward_folds(index, protocol=protocol)
    assert len(folds) == protocol.splits
    assert (
        leakage_violations(folds, index, horizon=protocol.horizon, embargo=protocol.embargo) == []
    )

    start = index.get_loc(folds[1].test[0])
    assert not folds[1].train.isin(index[start - protocol.horizon : start]).any()  # purged
    shorter = walk_forward_folds(index, protocol=Protocol(horizon=1, splits=3, embargo=10))
    assert shorter[1].train.size > folds[1].train.size  # a shorter label window purges fewer dates
