"""Ticket 29 — the purged, embargoed walk-forward splitter.

The contract: expanding contiguous folds, zero train/test label overlap, an
embargo after every earlier test block, and a naive split that demonstrably
leaks where this one does not.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pandas as pd
import pytest

from systematic_futures.ml.cv import Fold, leakage_violations, purged_walk_forward

INDEX = pd.bdate_range("2015-01-01", "2021-12-31")
HORIZON, EMBARGO, SPLITS = 5, 10, 5


def folds() -> list[Fold]:
    return purged_walk_forward(INDEX, n_splits=SPLITS, horizon=HORIZON, embargo=EMBARGO)


def test_folds_expand_forward_with_ordered_disjoint_test_blocks():
    built = folds()
    assert len(built) == SPLITS
    for previous, fold in pairwise(built):
        assert fold.test[0] > previous.test[-1]  # test blocks ordered and disjoint
        assert previous.train.isin(fold.train).all()  # the training set only grows
        assert fold.train.max() < fold.test[0]  # never trained on its own test period
    assert built[-1].test[-1] == INDEX[-1]
    assert not any(fold.test.max() >= pd.Timestamp("2022-01-01") for fold in built)


def test_purge_and_embargo_remove_exactly_their_windows():
    built = folds()
    assert leakage_violations(built, INDEX, horizon=HORIZON, embargo=EMBARGO) == []

    fold = built[2]
    start = INDEX.get_loc(fold.test[0])
    assert not fold.train.isin(INDEX[start - HORIZON : start]).any()  # purged
    assert INDEX[start - HORIZON - 1] in fold.train  # ... and only the window: not over-purged

    earlier = built[0]  # block 1: the only earlier embargo that lands inside this fold's train
    end = INDEX.get_loc(earlier.test[-1])
    assert not fold.train.isin(INDEX[end + 1 : end + 1 + EMBARGO]).any()  # embargoed
    assert INDEX[end + 1 + EMBARGO] in fold.train  # ... and only the window

    for j in range(2):  # every earlier test block is embargoed out of the train set
        end_j = INDEX.get_loc(built[j].test[-1])
        assert not fold.train.isin(INDEX[end_j + 1 : end_j + 1 + EMBARGO]).any()


def test_naive_split_leaks_where_the_purged_one_does_not():
    """The red case: the same blocks without purge/embargo train on their own test labels."""
    edges = np.linspace(0, len(INDEX), SPLITS + 2).astype(int)
    naive = [Fold(INDEX[: edges[k]], INDEX[edges[k] : edges[k + 1]]) for k in range(1, SPLITS + 1)]
    assert leakage_violations(naive, INDEX, horizon=HORIZON, embargo=EMBARGO)
    assert leakage_violations(folds(), INDEX, horizon=HORIZON, embargo=EMBARGO) == []


def test_deterministic_and_input_validated():
    first, second = folds(), folds()
    for a, b in zip(first, second, strict=True):
        assert a.train.equals(b.train) and a.test.equals(b.test)
    with pytest.raises(ValueError):
        purged_walk_forward(INDEX, n_splits=0)
    with pytest.raises(ValueError):
        purged_walk_forward(INDEX[:3], n_splits=5)
