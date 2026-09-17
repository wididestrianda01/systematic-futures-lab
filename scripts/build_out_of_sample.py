"""The out-of-sample read: the one read of that window, and the series the notebook plots.

Every family — the classic set plus 6a/6b — measured once on the out-of-sample window
(2022 → 2024Q1, the frozen snapshot's end) through the same pipeline seam the earlier reads
used. This is the project's single touch of those dates: nothing before this script reads them,
and nothing after it recomputes them.

The walk-forward schedule is extended across the whole frozen panel and the metrics are read on
the OOT dates only (`run`'s `dates` mask). Every fold trains and tunes on prior data only, so the
ML variants stay the protocol the decision read declared rather than a configuration invented
after seeing OOT numbers, and the last fold — the one whose test block covers 2022 — trains
entirely inside develop+validate. Signals are computed on the full panel, so no family starts 2022
with a truncated lookback.

Two reads, as in the decision read: `tables.csv` puts every method on the identical window;
`tables_like_for_like.csv` re-measures each ML variant on its own covered dates. The run asserts
both return the same rule verdict, so the OOT read cannot become a second decision surface.

`OOT.md` records the numbers and the pre-declared rule applied to them as **out-of-time evidence
about the decision read's verdict** — it does not re-decide it, and interpretation stays gated by
the reading canon and the interpretation pass.

`curves.csv` carries the per-method daily net return series at the headline cost level for both
windows: derived statistics only, never prices — the analysis notebook plots from it, so the
notebook needs no market data.

Run: uv run python scripts/build_out_of_sample.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from systematic_futures.catalogue import (
    BENCHMARK,
    LABELS,
    ML_VARIANTS,
    full_set,
    walk_forward_folds,
)
from systematic_futures.comparison import (
    headline_rows,
    like_for_like,
    protocol_meta,
    resolved,
    table_for,
)
from systematic_futures.data.panels import develop_validate, load_frozen, oot_window
from systematic_futures.decision import require_same_verdict, verdict
from systematic_futures.engine import curve
from systematic_futures.engine.metrics import sharpe
from systematic_futures.protocol import PROTOCOL
from systematic_futures.reporting import (
    benchmark_line,
    family_names,
    outcome_split,
    outcome_table,
)

RESULTS = Path("results/out_of_sample")


def outcome_doc(primary: dict, sensitivity: dict[str, dict], meta: pd.DataFrame) -> str:
    winners, losers = outcome_split(primary, ML_VARIANTS)
    if winners:
        outcome = (
            "**Out-of-time outcome: the rule is met by "
            + family_names(winners, LABELS)
            + "** on the OOT window, the same way it was met in develop+validate."
        )
        if losers:
            outcome += (
                " " + family_names(losers, LABELS) + " do not beat the benchmark out of time."
            )
    else:
        outcome = (
            "**Out-of-time outcome: the rule fails for both variants** on the OOT window: the "
            "decision-read winners do not carry their edge past 2021, and that is the finding, not a "
            "reason to retune."
        )
    coverage = {v: [f"{meta.loc[v, 'signal_coverage']:.0%}"] for v in ML_VARIANTS}
    return (
        "# The out-of-sample read (2022 → 2024Q1)\n\n"
        "**One touch.** These are the only numbers in the project measured on dates after "
        "2021-12-31. The decision read in `results/decision/DECISION.md` stands as it was "
        "decided; what follows is out-of-time evidence about it, not a reopening.\n\n"
        f"Rule: the pre-declared rule of `results/decision/DECISION_RULE.md`, applied here to the "
        f"{PROTOCOL.headline_bps:.0f} bps row of the primary read on the OOT window.\n\n"
        f"{benchmark_line(primary, BENCHMARK)}\n\n"
        f"{outcome_table(primary, sensitivity, ML_VARIANTS, LABELS, extra_headers=('OOT coverage',), extra_cells=coverage)}\n\n"
        f"{outcome}\n\n"
        "**Protocol.** The walk-forward schedule is the decision read's, extended across the whole "
        "frozen panel; each fold is refit on prior data only and 6b's search runs inside its "
        "fold's training set, so no fold saw an out-of-sample date before predicting it. The "
        "fold covering 2022 trains entirely inside develop+validate. Trial counts are unchanged "
        "as multiple-testing inputs: 6a searched nothing (trials = 1), 6b = folds x trials "
        f"({PROTOCOL.splits} x {PROTOCOL.n_trials} = {PROTOCOL.splits * PROTOCOL.n_trials}).\n\n"
        "**Reading.** What the OOT numbers mean (regime attribution, the trend drought the "
        "benchmark carries into 2022, what the ML variants generalized) is interpretation, and "
        "it stays with the memo behind the reading canon and the "
        "interpretation pass. Deliberately not written here.\n\n"
        "Robustness read (`tables_like_for_like.csv`): the verdict is asserted identical on both "
        "reads before this file is written.\n"
    )


def main() -> int:
    wide, basis_wide = load_frozen()
    oot = oot_window(wide)
    window = develop_validate(wide)
    print(
        f"OOT window: {oot.index.min().date()} → {oot.index.max().date()} ({len(oot.index)} dates)"
    )

    folds = walk_forward_folds(wide.index, protocol=PROTOCOL)
    assert max(fold.test.max() for fold in folds) == wide.index.max(), (
        "last fold must cover the end"
    )

    resolved_oot = resolved(full_set(basis_wide, protocol=PROTOCOL), wide)
    meta = protocol_meta(resolved_oot, oot.index)

    RESULTS.mkdir(parents=True, exist_ok=True)
    full = table_for(resolved_oot, wide, protocol=PROTOCOL, dates=oot.index).join(meta, on="method")
    full.to_csv(RESULTS / "tables.csv", index=False)

    like_tables = like_for_like(
        resolved_oot, wide, variants=ML_VARIANTS, protocol=PROTOCOL, dates=oot.index
    )
    like_tables.to_csv(RESULTS / "tables_like_for_like.csv", index=False)

    curves = []
    # The develop+validate series is the families run's own series: signals computed on that window,
    # so the warm-up and the ML fold geometry are the ones those tables were built from. The OOT
    # series is the one the tables above were built from: full-panel signals, metrics masked to the
    # OOT dates. Each curve is then re-measured and asserted equal to its table's Sharpe before
    # anything is written — a plot in the notebook cannot disagree with the number printed beside it.
    window_signals = resolved(full_set(basis_wide, protocol=PROTOCOL), window)
    reference = headline_rows(pd.read_csv("results/decision/tables.csv"), PROTOCOL.headline_bps)
    for name, method in window_signals.items():
        returns = curve(
            method.signal, window, vol_target=PROTOCOL.vol_target, bps=PROTOCOL.headline_bps
        )
        got, want = sharpe(returns), float(reference.loc[name, "sharpe"])
        assert abs(got - want) < 1e-9, f"{name}: curve Sharpe {got} != the table's {want}"
        curves.append(
            pd.DataFrame(
                {
                    "window": "develop_validate",
                    "method": name,
                    "date": returns.index,
                    "ret": returns.to_numpy(),
                }
            )
        )
    for name, method in resolved_oot.items():
        returns = curve(
            method.signal,
            wide,
            vol_target=PROTOCOL.vol_target,
            bps=PROTOCOL.headline_bps,
            dates=oot.index,
        )
        got = sharpe(returns)
        want = float(headline_rows(full, PROTOCOL.headline_bps).loc[name, "sharpe"])
        assert abs(got - want) < 1e-9, f"{name}: OOT curve Sharpe {got} != the OOT table's {want}"
        curves.append(
            pd.DataFrame(
                {
                    "window": "oot",
                    "method": name,
                    "date": returns.index,
                    "ret": returns.to_numpy(),
                }
            )
        )
    pd.concat(curves, ignore_index=True).round({"ret": 8}).to_csv(
        RESULTS / "curves.csv", index=False
    )

    primary = verdict(headline_rows(full, PROTOCOL.headline_bps), ML_VARIANTS, benchmark=BENCHMARK)
    sensitivity = {
        variant: verdict(
            headline_rows(like_tables[like_tables["window"] == variant], PROTOCOL.headline_bps),
            ML_VARIANTS,
            benchmark=BENCHMARK,
        )
        for variant in ML_VARIANTS
    }
    require_same_verdict(primary, sensitivity)
    (RESULTS / "OOT.md").write_text(outcome_doc(primary, sensitivity, meta))

    print()
    print(
        headline_rows(full, PROTOCOL.headline_bps)[
            ["sharpe", "dsr", "turnover", "signal_coverage"]
        ].to_string()
    )
    print(f"\nwrote {RESULTS}/tables.csv, tables_like_for_like.csv, curves.csv, OOT.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
