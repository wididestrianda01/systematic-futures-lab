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
from systematic_futures.ml import (
    Fold,
    leakage_violations,
    lgbm_defaults,
    lgbm_tuned,
    purged_walk_forward,
)

BENCHMARK = "tsmom"
HEADLINE_BPS = 2.0
VOL_TARGET = 0.10  # the shared sizing overlay: every family runs at 10% annualized vol

# The ML walk-forward protocol (tickets 29/31), stated once for the decision read and the out-of-sample read: a change to
# the fold geometry, the label horizon or the search budget is a protocol change, and both phases
# must read the same one or the OOT read no longer extends the schedule the rule was decided under.
HORIZON, SPLITS, EMBARGO, N_TRIALS = 5, 5, 10, 20
ML_VARIANTS = ("ml_defaults", "ml_tuned")
LABELS = {"ml_defaults": "6a", "ml_tuned": "6b"}


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


def ml_set(basis: pd.DataFrame) -> dict[str, object]:
    """Variants 6a/6b, both bound to the one walk-forward protocol above."""
    return {
        "ml_defaults": lgbm_defaults(basis, horizon=HORIZON, n_splits=SPLITS, embargo=EMBARGO),
        "ml_tuned": lgbm_tuned(
            basis, horizon=HORIZON, n_splits=SPLITS, embargo=EMBARGO, n_trials=N_TRIALS
        ),
    }


def full_set(basis: pd.DataFrame) -> dict[str, object]:
    """The whole comparison set: the classics plus 6a/6b, one protocol for both phases."""
    return {**classic_set(basis), **ml_set(basis)}


def walk_forward_folds(index: pd.DatetimeIndex) -> list[Fold]:
    """The one walk-forward schedule, purged and embargoed, asserted leak-free.

    the decision read runs it over develop+validate, the out-of-sample read over the whole frozen panel so the fold covering
    2022 trains entirely inside develop+validate; both use this call, so the schedule cannot differ
    between the decision and the out-of-sample read.
    """
    folds = purged_walk_forward(index, n_splits=SPLITS, horizon=HORIZON, embargo=EMBARGO)
    problems = leakage_violations(folds, index, horizon=HORIZON, embargo=EMBARGO)
    assert not problems, f"splitter leaks: {problems[:3]}"
    return folds


def headline_rows(tables: pd.DataFrame, bps: float = HEADLINE_BPS) -> pd.DataFrame:
    """The comparison table's one read: one row per method at one cost level, indexed by method.

    Every consumer of the table — the phase scripts, the notebooks, the decision
    rule — reads it this way, so the shape (a `bps` level, a `method` column) is
    known here and nowhere else.
    """
    return tables[tables["bps"] == bps].set_index("method")


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


def protocol_meta(
    signals: dict[str, pd.DataFrame],
    trials: dict[str, int],
    dates: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """Per-method multiple-testing and coverage inputs, indexed by method.

    `trials` is what each family's selection actually evaluated (6a searched nothing, 6b searched
    folds x trials), and `signal_coverage` is the share of `dates` — the family's own index when
    absent — on which it holds a non-constructed position.
    """
    index = next(iter(signals.values())).index if dates is None else dates
    return pd.DataFrame(
        {
            name: {
                "trials": trials[name],
                "signal_coverage": float((signal.loc[index] != 0).any(axis=1).mean()),
            }
            for name, signal in signals.items()
        }
    ).T


def like_for_like(
    signals: dict[str, pd.DataFrame],
    closes: pd.DataFrame,
    trials: dict[str, int],
    dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """The robustness read: every family re-measured on each ML variant's own covered dates.

    A robustness read, never a second decision surface — the phase scripts assert both reads return
    the same verdict before either is written. `dates` is the window being evaluated, so the same
    call serves the decision read (develop+validate) and the out-of-sample read (the OOT window).
    """
    parts = []
    for variant in ML_VARIANTS:
        covered = signals[variant].index[(signals[variant] != 0).any(axis=1)].intersection(dates)
        table = table_for(signals, closes, trials=trials, dates=covered)
        table["window"] = variant
        table["window_dates"] = len(covered)
        parts.append(table)
        print(f"like-for-like window ({variant}): {len(covered)} of {len(dates)} dates")
    return pd.concat(parts, ignore_index=True)
