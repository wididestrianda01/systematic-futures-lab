"""The comparison table's shape, and every read taken of it.

`table_for` assembles the table through the pipeline seam — one row per method per cost level, the
method named in a column. `headline_rows` is the one read every consumer takes of it.
`protocol_meta` reports the inputs each family's numbers need to be read honestly: the trial count its
selection evaluated, and how much of the window it held a constructed position on. `like_for_like`
re-measures every family on one ML variant's own covered dates, which is a robustness read, never a
second decision surface.

Nothing here knows which families are compared, or what a protocol is beyond the values it is handed:
the comparison set lives in `catalogue`. This module imports no family, so a consumer that only needs
the table's shape — the decision rule, the carry-frequency diagnostic — pays for neither the method
packages nor the ML stack.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import pandas as pd

from systematic_futures.engine import run
from systematic_futures.protocol import PROTOCOL, Method, Protocol


def resolved(methods: Mapping[str, Method], closes: pd.DataFrame) -> dict[str, Method]:
    """The method set with each family's signal computed on `closes` — the currency every read takes.

    A callable signal is called once here, so a read that needs both the numbers and the contract they
    were produced under (a table, a coverage column, a like-for-like re-measure) does not run a tuned
    family's search twice.
    """
    return {
        name: Method(_computed(method.signal, closes), method.trials)
        for name, method in methods.items()
    }


def _computed(signal, closes: pd.DataFrame):
    """A signal source in its computed form: callables are called, tables pass through."""
    return signal(closes) if callable(signal) else signal


def headline_rows(tables: pd.DataFrame, bps: float) -> pd.DataFrame:
    """The comparison table's one read: one row per method at one cost level, indexed by method.

    `bps` is the headline cost level, passed by the caller rather than defaulted here: which level a
    decision is read at is a property of the protocol, not of the table's shape.
    """
    return tables[tables["bps"] == bps].set_index("method")


def table_for(
    methods: Mapping[str, Method],
    closes: pd.DataFrame,
    *,
    protocol: Protocol = PROTOCOL,
    dates: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """Every method through the identical seam — one row per method per bps level.

    `dates` is the engine's like-for-like mask, applied identically to every method; the method's own
    trial count rides into the table as the DSR's deflation input.
    """
    records = []
    for name, method in methods.items():
        table = run(
            method.signal,
            closes,
            vol_target=protocol.vol_target,
            cap=protocol.cap,
            trials=method.trials,
            dates=dates,
        )
        table["method"] = name
        records.append(table.reset_index().rename(columns={"index": "bps"}))
    return pd.concat(records, ignore_index=True)


def covered_dates(signal: pd.DataFrame, dates: pd.DatetimeIndex | None = None) -> pd.DatetimeIndex:
    """The dates a family holds a constructed position on, inside `dates` when given."""
    index = signal.index if dates is None else dates
    return index[(signal.reindex(index) != 0).any(axis=1)]


def protocol_meta(
    methods: Mapping[str, Method], dates: pd.DatetimeIndex | None = None
) -> pd.DataFrame:
    """Per-method multiple-testing and coverage inputs, indexed by method.

    `trials` is what each family's selection actually evaluated (6a searched nothing, 6b searched
    folds x trials), and `signal_coverage` is the share of `dates` — the family's own index when
    absent — on which it holds a constructed position. Takes the resolved methods: coverage is a
    property of the signals computed for this run, the trial count a property of the contract.
    """
    index = next(iter(methods.values())).signal.index if dates is None else dates
    return pd.DataFrame(
        {
            name: {
                "trials": method.trials,
                "signal_coverage": float(len(covered_dates(method.signal, index)) / len(index)),
            }
            for name, method in methods.items()
        }
    ).T


def like_for_like(
    methods: Mapping[str, Method],
    closes: pd.DataFrame,
    *,
    variants: Iterable[str],
    protocol: Protocol = PROTOCOL,
    dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """The robustness read: every family re-measured on each ML variant's own covered dates.

    A robustness read, never a second decision surface — the phase scripts assert both reads return the
    same verdict before either is written. `dates` is the window being evaluated, so the same call
    serves the decision read (develop+validate) and the out-of-sample read (the OOT window).
    """
    parts = []
    for variant in variants:
        covered = covered_dates(methods[variant].signal, dates)
        table = table_for(methods, closes, protocol=protocol, dates=covered)
        table["window"] = variant
        table["window_dates"] = len(covered)
        parts.append(table)
        print(f"like-for-like window ({variant}): {len(covered)} of {len(dates)} dates")
    return pd.concat(parts, ignore_index=True)
