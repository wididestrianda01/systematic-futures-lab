"""The pre-declared decision rule, applied to the comparison table — the rule, not the prose.

the decision read pre-declared, before its numbers existed, that each ML variant must beat
the `tsmom` benchmark on headline Deflated Sharpe after costs, and that a variant
which fails is documented as a failed challenger rather than retuned until it
passes. That comparison lives here rather than in the phase script, so it can be
applied to a synthetic table; the script keeps the pre-declared text and the
markdown it writes. This module owns the comparison, and the guard that every
read of the same numbers must return the same verdict.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd

from systematic_futures.harness import BENCHMARK


def verdict(rows: pd.DataFrame, variants: Sequence[str], *, benchmark: str = BENCHMARK) -> dict:
    """Apply the rule to one read: `rows` = the headline rows, indexed by method.

    A variant beats the benchmark only on a strictly greater DSR after costs — a
    tie is not a win.
    """
    base = rows.loc[benchmark]
    return {
        "benchmark_sharpe": float(base["sharpe"]),
        "benchmark_dsr": float(base["dsr"]),
        "beats": {v: bool(rows.loc[v, "dsr"] > base["dsr"]) for v in variants},
        "variant_sharpe": {v: float(rows.loc[v, "sharpe"]) for v in variants},
        "variant_dsr": {v: float(rows.loc[v, "dsr"]) for v in variants},
    }


def require_same_verdict(primary: Mapping, reads: Mapping[str, Mapping]) -> None:
    """Stop the run when a robustness read disagrees with the primary one.

    The decision surface must not depend on which read it is taken from, so a
    disagreement is a hard failure rather than something to write up.
    """
    for name, read in reads.items():
        if read["beats"] != primary["beats"]:
            raise ValueError(
                f"{name}: primary and like-for-like reads disagree on the rule's verdict — "
                "the decision surface must not depend on the read"
            )
