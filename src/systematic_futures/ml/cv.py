"""Purged, embargoed walk-forward cross-validation — hand-rolled, by design.

Why, from first principles (AFML ch. 7): with overlapping h-day forward labels
a training sample at date d carries information about returns through d+h. If d
sits within h days before a test block, its label overlaps the test period — the
model has effectively trained on the test set, and a plain (or shuffled) k-fold
split cannot see it because it splits samples, not time. Two mechanisms remove
it:

- **purge**: drop every training date whose h-day label window reaches into the
  test block;
- **embargo**: drop the `embargo` dates immediately after each test block from
  every later training set — serial correlation makes those days' features
  dependent on the test period even when their labels do not overlap.

Folds are walk-forward (expanding train, contiguous test blocks, no shuffling),
which is also the honest order a live model would have been refit in.
`leakage_violations` re-checks both mechanisms from the fold semantics, so the
harness can assert cleanliness rather than trust the constructor.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Fold:
    """One walk-forward fold: a contiguous test block plus everything usable before it."""

    train: pd.DatetimeIndex
    test: pd.DatetimeIndex


def purged_walk_forward(
    index: pd.DatetimeIndex, *, n_splits: int = 5, horizon: int = 1, embargo: int = 5
) -> list[Fold]:
    """Expanding-window folds over `index`, purged and embargoed.

    index: the sample dates (unique). horizon: label length in trading days.
    embargo: dates excluded after each test block. Deterministic by construction.
    """
    dates = pd.DatetimeIndex(sorted(pd.Index(index).unique()))
    n = len(dates)
    if n_splits < 1:
        raise ValueError("n_splits must be >= 1")
    if n < n_splits + 1:
        raise ValueError(f"need at least {n_splits + 1} dates, got {n}")
    edges = np.linspace(0, n, n_splits + 2).astype(int)
    blocks = [dates[edges[i] : edges[i + 1]] for i in range(n_splits + 1)]

    folds = []
    for k in range(1, n_splits + 1):
        start = int(dates.get_loc(blocks[k][0]))
        keep = np.zeros(n, dtype=bool)
        keep[: max(start - horizon, 0)] = True  # purge: labels reaching into the test block
        for j in range(1, k):  # embargo after every earlier test block
            end = int(dates.get_loc(blocks[j][-1]))
            keep[end + 1 : end + 1 + embargo] = False
        folds.append(Fold(dates[keep], blocks[k]))
    return folds


def leakage_violations(
    folds: list[Fold], index: pd.DatetimeIndex, *, horizon: int = 1, embargo: int = 5
) -> list[str]:
    """Every train/test contamination in `folds` — empty means clean.

    Checks the two mechanisms independently: a training date whose label window
    reaches into a fold's test block, and a training date inside the embargo
    window after an earlier test block.
    """
    dates = pd.DatetimeIndex(sorted(pd.Index(index).unique()))
    pos = {date: i for i, date in enumerate(dates)}
    problems: list[str] = []
    for k, fold in enumerate(folds):
        train = set(fold.train)
        start = pos[fold.test[0]]
        for i in range(max(start - horizon, 0), start):
            if dates[i] in train:
                problems.append(
                    f"fold {k}: train date {dates[i].date()} labels into the test block "
                    f"starting {fold.test[0].date()}"
                )
        for j in range(k):
            end = pos[folds[j].test[-1]]
            for date in dates[end + 1 : end + 1 + embargo]:
                if date in train:
                    problems.append(
                        f"fold {k}: train date {date.date()} inside the embargo "
                        f"after fold {j}'s test block"
                    )
    return problems
