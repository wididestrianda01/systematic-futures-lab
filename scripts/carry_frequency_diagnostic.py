"""Carry construction diagnostic — the family's headline number is a rebalancing-frequency artifact.

`carry` runs `sign(-basis)` refreshed every session through the shared overlay. That sign moves with
the front leg's own price as much as with the term structure, so at daily frequency it churns
(turnover ~0.18/day on the OOT window) and the Sharpe in the comparison tables is a property of the
churn rather than of the carry premium: the identical rule held from each month's first session earns
the other way round, and the sign-inverse of the daily rule earns a large positive Sharpe. The
literature's version of this rule is the per-security timing construction rebalanced monthly
(Koijen, Moskowitz, Pedersen & Vrugt, "Carry", JFE 2018), which is the second row here.

This is a diagnostic, not part of the comparison set: no family is added to the tables, no rule is
re-decided, and this script reaches no date the committed reads did not already measure. It exists so
`docs/findings/memo.md` and `docs/methods/carry.md` can quote the frequency comparison from a
committed artifact.

Run: uv run python scripts/carry_frequency_diagnostic.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from systematic_futures.comparison import headline_rows, resolved, table_for
from systematic_futures.data.panels import develop_validate, load_frozen, oot_window
from systematic_futures.protocol import PROTOCOL, Method

RESULTS = Path("results/out_of_sample")


def month_start(signal: pd.DataFrame) -> pd.DataFrame:
    """Each signal held from its month's first session to the next month's — the monthly rebalance."""
    return signal.groupby([signal.index.year, signal.index.month]).transform("first")


def constructions(basis: pd.DataFrame) -> dict[str, Method]:
    """The committed rule, the same rule at monthly frequency, and the committed rule's inverse."""
    daily = -np.sign(basis)
    return {
        "daily_sign_neg_basis": Method(daily),
        "month_start_sign_neg_basis": Method(month_start(daily)),
        "daily_sign_basis_inverse": Method(-daily),
    }


def main() -> int:
    wide, basis_wide = load_frozen()
    window, oot = develop_validate(wide), oot_window(wide)

    # Baseline rows must reproduce the committed carry row: signals on the same panel the phase
    # script used, metrics masked the same way, so this file cannot silently diverge from the tables.
    committed = {
        "develop_validate": float(
            headline_rows(pd.read_csv("results/families/tables.csv"), PROTOCOL.headline_bps).loc[
                "carry", "sharpe"
            ]
        ),
        "oot": float(
            headline_rows(
                pd.read_csv("results/out_of_sample/tables.csv"), PROTOCOL.headline_bps
            ).loc["carry", "sharpe"]
        ),
    }

    rows = []
    for label, closes, mask in (("develop_validate", window, None), ("oot", wide, oot.index)):
        methods = resolved(constructions(basis_wide.reindex(index=closes.index)), closes)
        table = table_for(methods, closes, protocol=PROTOCOL, dates=mask)
        table["window"] = label
        got = float(
            headline_rows(table, PROTOCOL.headline_bps).loc["daily_sign_neg_basis", "sharpe"]
        )
        assert abs(got - committed[label]) < 1e-9, f"{label}: {got} != committed {committed[label]}"
        rows.append(table)

    out = pd.concat(rows, ignore_index=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    out.to_csv(RESULTS / "carry_frequency.csv", index=False)

    headline = headline_rows(out, PROTOCOL.headline_bps)[["sharpe", "turnover", "dsr"]]
    print(headline.to_string())
    print(f"\nwrote {RESULTS}/carry_frequency.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
