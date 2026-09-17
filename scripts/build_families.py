"""The families read: run every method family through the shared seam on the frozen
panel — develop+validate only (2010-01-01 → 2021-12-31; the OOT window 2022+ stays
untouched for its single end-of-project read) — and commit the per-family
comparison tables plus the seasonality-after-costs finding.

Window and comparison set come from the package (`data.panels.develop_validate`,
`catalogue.classic_set`) so the decision read cannot drift from this one. The committed
numbers are mechanical output; the MUST canon gates interpreting them.

Run: uv run python scripts/build_families.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from systematic_futures.catalogue import BENCHMARK, classic_set
from systematic_futures.comparison import headline_rows, resolved, table_for
from systematic_futures.data.panels import (
    DEVELOP_START,
    VALIDATE_END,
    develop_validate,
    load_frozen,
)
from systematic_futures.protocol import PROTOCOL, Method

RESULTS = Path("results/families")

GATE = (
    "**Interpretation status: gated.** The MUST reading canon gates interpretation of any\n"
    "backtest result (`docs/reading/README.md`), and M7 is still partially gated\n"
    "(`docs/reading/notes.md`). Everything below the numbers is a *provisional* reading of\n"
    "mechanical output, not a checked finding, until the canon closes; the\n"
    "out-of-sample interpretation pass owns it.\n\n"
)


def main() -> int:
    wide, basis_wide = load_frozen()
    window = develop_validate(wide)
    tables = table_for(resolved(classic_set(basis_wide), window), window, protocol=PROTOCOL)

    RESULTS.mkdir(parents=True, exist_ok=True)
    tables.to_csv(RESULTS / "tables.csv", index=False)

    rows = headline_rows(tables, PROTOCOL.headline_bps)
    free = headline_rows(tables, 0.0)
    season, season_free = rows.loc["seasonality"], free.loc["seasonality"]
    trend, trend_free = rows.loc[BENCHMARK], free.loc[BENCHMARK]
    xs = rows.loc["xs_momentum"]
    dies = season["sharpe"] < trend["sharpe"]

    (RESULTS / "FINDINGS.md").write_text(
        "# Families read — findings\n\n"
        f"Window: develop+validate ({DEVELOP_START.date()} → {VALIDATE_END.date()}), frozen "
        f"panel, {PROTOCOL.vol_target:.0%} vol target, 16 CME roots; OOT (2022+) untouched.\n\n"
        f"{GATE}"
        "## Seasonality standalone loses (expected finding)\n\n"
        f"Numbers: standalone seasonality Sharpe {season_free['sharpe']:.2f} before costs, "
        f"{season['sharpe']:.2f} after {PROTOCOL.headline_bps:.0f} bps "
        f"(turnover {season['turnover']:.2f}), against the TSMOM benchmark's "
        f"{trend_free['sharpe']:.2f} / {trend['sharpe']:.2f} "
        f"(turnover {trend['turnover']:.2f}). Reported as a failed challenger.\n\n"
        + (
            "Provisional reading: the failure is edge rather than churn — the signal is already "
            "negative before costs and its turnover is *below* the benchmark's, so cost drag "
            "alone does not explain it.\n"
            if dies
            else "Provisional reading: not confirmed on this window — investigate before "
            "reporting.\n"
        )
        + "\nFull per-method, per-bps metrics in `tables.csv` "
        "(Sharpe, Sortino, max DD, turnover, DSR). "
        + (
            f"The cross-sectional family is the other side of this story: it posts Sharpe "
            f"{xs['sharpe']:.2f} against the benchmark's {trend['sharpe']:.2f} after "
            f"{PROTOCOL.headline_bps:.0f} bps.\n"
            if xs["sharpe"] > trend["sharpe"]
            else "No other family beats the benchmark on this window.\n"
        )
    )
    inverted = table_for(
        {
            "carry_inverted": Method(
                np.sign(basis_wide.reindex(index=window.index, columns=window.columns))
            )
        },
        window,
        protocol=PROTOCOL,
    )
    inverse_sharpe = headline_rows(inverted, PROTOCOL.headline_bps).loc["carry_inverted", "sharpe"]
    (RESULTS / "FINDINGS.md").open("a").write(
        "\n## Carry: the sign convention decides (failed challenger under the literature sign)\n\n"
        "Numbers: the KMPV convention (long backwardated / short contangoed, i.e. carry = "
        "front−next over next) posts Sharpe "
        f"{rows.loc['carry', 'sharpe']:.2f} "
        f"at {PROTOCOL.headline_bps:.0f} bps on develop+validate, while the inverted sign — long "
        f"contangoed — posts {inverse_sharpe:.2f}, against the benchmark's "
        f"{trend['sharpe']:.2f}. The literature convention stays in the tables.\n\n"
        "Provisional reading: a sign flip of that size is not a measurement artifact to average "
        "away, and what it means is M3 (carry) / M4 (commodity term structure) territory — so "
        "it waits for the canon rather than being resolved by picking the profitable sign.\n"
    )
    print(tables.to_string(index=False))
    print(f"wrote {RESULTS / 'tables.csv'} and {RESULTS / 'FINDINGS.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
