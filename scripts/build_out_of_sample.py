"""the out-of-sample read numbers: the one read of the out-of-sample window, and the series the notebook plots.

Every family — the classic set plus 6a/6b — measured once on the out-of-sample window
(2022 → 2024Q1, the frozen snapshot's end) through the same pipeline seam the earlier phases
used. This is the project's single touch of those dates: nothing before this script reads them,
and nothing after it recomputes them.

The walk-forward schedule is extended across the whole frozen panel and the metrics are read on
the OOT dates only (`run`'s `dates` mask). Every fold trains and tunes on prior data only, so the
ML variants stay the protocol the decision read declared rather than a configuration invented after seeing
OOT numbers, and the last fold — the one whose test block covers 2022 — trains entirely inside
develop+validate. Signals are computed on the full panel, so no family starts 2022 with a
truncated lookback.

Two reads, as in the decision read: `tables.csv` puts every method on the identical window;
`tables_like_for_like.csv` re-measures each ML variant on its own covered dates. The run asserts
both return the same rule verdict, so the OOT read cannot become a second decision surface.

`OOT.md` records the numbers and the pre-declared rule applied to them as **out-of-time evidence
about the decision-read verdict** — it does not re-decide it, and interpretation stays gated by the
reading canon (ticket 32) and the decision-read interpretation ticket (34).

`curves.csv` carries the per-method daily net return series at the headline cost level for both
windows: derived statistics only, never prices — the analysis notebook plots from it, so the
notebook needs no market data.

Run: uv run python scripts/build_out_of_sample.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from systematic_futures.data.panels import develop_validate, load_frozen, oot_window
from systematic_futures.decision import require_same_verdict, verdict
from systematic_futures.engine import curve
from systematic_futures.engine.metrics import sharpe
from systematic_futures.harness import (
    BENCHMARK,
    HEADLINE_BPS,
    VOL_TARGET,
    classic_set,
    headline_rows,
    table_for,
)
from systematic_futures.ml import leakage_violations, lgbm_defaults, lgbm_tuned, purged_walk_forward

RESULTS = Path("results/results.1")
HORIZON, SPLITS, EMBARGO, N_TRIALS = 5, 5, 10, 20
ML_VARIANTS = ("ml_defaults", "ml_tuned")
LABELS = {"ml_defaults": "6a", "ml_tuned": "6b"}


def outcome_doc(primary: dict, sensitivity: dict[str, dict], meta: pd.DataFrame) -> str:
    rows = "\n".join(
        f"| {LABELS[v]} `{v}` | {primary['variant_sharpe'][v]:.2f} | "
        f"{primary['variant_dsr'][v]:.4g} | {'yes' if primary['beats'][v] else 'no'} | "
        f"{sensitivity[v]['variant_dsr'][v]:.4g} | {'yes' if sensitivity[v]['beats'][v] else 'no'} | "
        f"{meta.loc[v, 'signal_coverage']:.0%} |"
        for v in ML_VARIANTS
    )
    winners = [v for v in ML_VARIANTS if primary["beats"][v]]
    losers = [v for v in ML_VARIANTS if not primary["beats"][v]]
    if winners:
        outcome = (
            "**Out-of-time outcome: the rule is met by "
            + ", ".join(f"{LABELS[v]} `{v}`" for v in winners)
            + "** on the OOT window, the same way it was met in develop+validate."
        )
        if losers:
            outcome += (
                " "
                + ", ".join(f"{LABELS[v]} `{v}`" for v in losers)
                + " do not beat the benchmark out of time."
            )
    else:
        outcome = (
            "**Out-of-time outcome: the rule fails for both variants** on the OOT window — the "
            "decision-read winners do not carry their edge past 2021, and that is the finding, not a "
            "reason to retune."
        )
    return (
        "# the out-of-sample read — the out-of-sample read (2022 → 2024Q1)\n\n"
        "**One touch.** These are the only numbers in the project measured on dates after "
        "2021-12-31. The decision-read verdict in `results/decision/DECISION.md` stands as it was "
        "decided; what follows is out-of-time evidence about it, not a reopening.\n\n"
        f"Rule: the pre-declared rule of `results/decision/DECISION_RULE.md`, applied here to the "
        f"{HEADLINE_BPS:.0f} bps row of the primary read on the OOT window.\n\n"
        f"Benchmark `{BENCHMARK}`: Sharpe {primary['benchmark_sharpe']:.2f}, "
        f"DSR {primary['benchmark_dsr']:.4g}.\n\n"
        "| variant | Sharpe | DSR (primary) | beats benchmark | DSR (own dates) | beats benchmark "
        "| OOT coverage |\n|---|---|---|---|---|---|---|\n"
        f"{rows}\n\n"
        f"{outcome}\n\n"
        "**Protocol.** The walk-forward schedule is the decision-read one extended across the whole "
        "frozen panel; each fold is refit on prior data only and 6b's search runs inside its "
        "fold's training set, so no fold saw an out-of-sample date before predicting it. The "
        "fold covering 2022 trains entirely inside develop+validate. Trial counts are unchanged "
        "as multiple-testing inputs: 6a searched nothing (trials = 1), 6b = folds x trials "
        f"({SPLITS} x {N_TRIALS} = {SPLITS * N_TRIALS}).\n\n"
        "**Reading.** What the OOT numbers mean — regime attribution, the trend drought the "
        "benchmark carries into 2022, what the ML variants generalized — is interpretation, and "
        "it stays with the memo (ticket 38) behind the reading canon (ticket 32) and the decision-read "
        "interpretation (ticket 34). Deliberately not written here.\n\n"
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

    folds = purged_walk_forward(wide.index, n_splits=SPLITS, horizon=HORIZON, embargo=EMBARGO)
    problems = leakage_violations(folds, wide.index, horizon=HORIZON, embargo=EMBARGO)
    assert not problems, f"splitter leaks across the OOT boundary: {problems[:3]}"
    assert max(fold.test.max() for fold in folds) == wide.index.max(), (
        "last fold must cover the end"
    )

    methods = {
        **classic_set(basis_wide),
        "ml_defaults": lgbm_defaults(basis_wide, horizon=HORIZON, n_splits=SPLITS, embargo=EMBARGO),
        "ml_tuned": lgbm_tuned(
            basis_wide, horizon=HORIZON, n_splits=SPLITS, embargo=EMBARGO, n_trials=N_TRIALS
        ),
    }
    signals = {name: method(wide) for name, method in methods.items()}
    trials = {name: getattr(method, "trials", 1) for name, method in methods.items()}
    meta = pd.DataFrame(
        {
            name: {
                "trials": trials[name],
                "signal_coverage": float((signals[name].loc[oot.index] != 0).any(axis=1).mean()),
            }
            for name in methods
        }
    ).T

    RESULTS.mkdir(parents=True, exist_ok=True)
    full = table_for(signals, wide, trials=trials, dates=oot.index).join(meta, on="method")
    full.to_csv(RESULTS / "tables.csv", index=False)

    like = []
    for variant in ML_VARIANTS:
        covered = signals[variant].index[(signals[variant] != 0).any(axis=1)]
        dates = oot.index.intersection(covered)
        table = table_for(signals, wide, trials=trials, dates=dates)
        table["window"] = variant
        table["window_dates"] = len(dates)
        like.append(table)
        print(f"like-for-like window ({variant}): {len(dates)} of {len(oot.index)} OOT dates")
    like_tables = pd.concat(like, ignore_index=True)
    like_tables.to_csv(RESULTS / "tables_like_for_like.csv", index=False)

    curves = []
    # The develop+validate series is the develop+validate run's own series: signals computed on that window,
    # so the warm-up and the ML fold geometry are the ones those tables were built from. The OOT
    # series is the one the tables above were built from: full-panel signals, metrics masked to the
    # OOT dates. Each curve is then re-measured and asserted equal to its table's Sharpe before
    # anything is written — a plot in the notebook cannot disagree with the number printed beside it.
    window_signals = {name: method(window) for name, method in methods.items()}
    reference = headline_rows(pd.read_csv("results/decision/tables.csv"))
    for name, signal in window_signals.items():
        returns = curve(signal, window, vol_target=VOL_TARGET, bps=HEADLINE_BPS)
        got, want = sharpe(returns), float(reference.loc[name, "sharpe"])
        assert abs(got - want) < 1e-9, f"{name}: curve Sharpe {got} != decision-read table's {want}"
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
    for name, signal in signals.items():
        returns = curve(signal, wide, vol_target=VOL_TARGET, bps=HEADLINE_BPS, dates=oot.index)
        got = sharpe(returns)
        want = float(headline_rows(full).loc[name, "sharpe"])
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

    primary = verdict(headline_rows(full), ML_VARIANTS)
    sensitivity = {
        variant: verdict(headline_rows(like_tables[like_tables["window"] == variant]), ML_VARIANTS)
        for variant in ML_VARIANTS
    }
    require_same_verdict(primary, sensitivity)
    (RESULTS / "OOT.md").write_text(outcome_doc(primary, sensitivity, meta))

    print()
    print(headline_rows(full)[["sharpe", "dsr", "turnover", "signal_coverage"]].to_string())
    print(f"\nwrote {RESULTS}/tables.csv, tables_like_for_like.csv, curves.csv, OOT.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
