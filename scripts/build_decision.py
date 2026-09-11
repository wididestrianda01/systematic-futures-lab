"""the decision read harness: variants 4a/4b against the TSMOM benchmark through the identical seam.

Window: develop+validate (data through 2021-12-31). The walk-forward out-of-fold
folds ARE the decision window; the single OOT read (2022 → 2024Q1) stays
untouched for its end-of-project touch in the out-of-sample read.

Two readings of the same numbers, both committed:

- `tables.csv` — the primary, pre-declared read: every method on the identical
  window, so the ML variants sit flat on the days their folds do not cover (the
  `signal_coverage` column says how much of the window that is).
- `tables_like_for_like.csv` — every method re-measured on each ML variant's own
  covered dates, so a variant is not scored on a diluted series against a fully
  populated benchmark. Same accounting path: the engine's `dates` mask.

The pre-declared decision rule goes to `DECISION_RULE.md` in the same run with no
outcome text — the numbers live in the two tables and ticket 34 applies the rule.
Trial counts are the honest multiple-testing inputs: 6a searched nothing
(trials = 1), 6b searched n_splits x n_trials configurations per run.

Run: uv run python scripts/build_decision.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from systematic_futures.data.panels import load_frozen
from systematic_futures.engine import run
from systematic_futures.methods import (
    baseline,
    carry,
    seasonal_tilt,
    seasonality,
    tsmom,
    xs_momentum,
)
from systematic_futures.ml import leakage_violations, lgbm_defaults, lgbm_tuned, purged_walk_forward

RESULTS = Path("results/phase4")
VALIDATE_END = "2021-12-31"  # develop+validate end; the OOT window starts 2022
VOL_TARGET = 0.10
HORIZON, SPLITS, EMBARGO = 5, 5, 10
N_TRIALS = 20
BENCHMARK = "tsmom"
ML_VARIANTS = ("ml_defaults", "ml_tuned")
HEADLINE_BPS = 2.0

RULE = f"""# the decision read decision rule (pre-declared in the spec, before this phase's numbers existed)

**Rule.** Each ML variant (4a `ml_defaults`, 4b `ml_tuned`) must beat the
`{BENCHMARK}` benchmark on **decision-window Deflated Sharpe Ratio after costs**
— the {HEADLINE_BPS:.0f} bps row of the tables — to be reported as a winner.
The decision window is the walk-forward **out-of-fold** folds inside
develop+validate: the single out-of-sample read (2022 → 2024Q1) stays untouched
until the out-of-sample read, so it cannot be the decision window without spending the touch.

A variant that fails this test is documented as a **failed challenger** with the
numbers that failed it — not retuned until it passes.

**Which table decides.** `tables.csv` is the primary read: every method on the
identical window. `tables_like_for_like.csv` repeats the comparison on each ML
variant's own covered dates — the sensitivity read, because a signal that is
flat by construction on part of the window has its Sharpe diluted by roughly
sqrt(coverage) and must not be judged on that dilution alone. The rule is
applied to the primary read, with the sensitivity reported beside it.

**Multiple-testing inputs.** The DSR column is deflated by the configurations
each variant's selection actually evaluated: 4a searched none
(trials = 1), 4b searched folds x trials = {SPLITS} x {N_TRIALS} configurations
(trials = {SPLITS * N_TRIALS}), counted whether or not a fold's search was later
skipped for lack of data.

Outcome and interpretation: ticket 34 (blocked by the M7 reading gate).
"""


def main() -> int:
    wide, basis_wide = load_frozen()
    window = wide.loc[:VALIDATE_END]
    assert wide.index.max() > pd.Timestamp(VALIDATE_END), "loader must return the full panel"
    assert window.index.max() <= pd.Timestamp(VALIDATE_END), "OOT window must stay untouched"

    folds = purged_walk_forward(window.index, n_splits=SPLITS, horizon=HORIZON, embargo=EMBARGO)
    problems = leakage_violations(folds, window.index, horizon=HORIZON, embargo=EMBARGO)
    assert not problems, f"splitter leaks on the frozen window: {problems[:3]}"

    methods = {
        "ml_defaults": lgbm_defaults(basis_wide, horizon=HORIZON, n_splits=SPLITS, embargo=EMBARGO),
        "ml_tuned": lgbm_tuned(
            basis_wide, horizon=HORIZON, n_splits=SPLITS, embargo=EMBARGO, n_trials=N_TRIALS
        ),
        BENCHMARK: tsmom,
        "baseline": baseline,
        "xs_momentum": xs_momentum,
        "carry": carry(basis_wide),
        "seasonality": seasonality,
        "seasonal_tilt": seasonal_tilt,
    }
    signals = {name: method(window) for name, method in methods.items()}
    meta = {
        name: {
            "trials": getattr(method, "trials", 1),
            "signal_coverage": float((signals[name] != 0).any(axis=1).mean()),
        }
        for name, method in methods.items()
    }

    def table_for(dates: pd.DatetimeIndex | None = None) -> pd.DataFrame:
        records = []
        for name, signal in signals.items():
            table = run(
                signal, window, vol_target=VOL_TARGET, trials=meta[name]["trials"], dates=dates
            )
            table["method"] = name
            records.append(table.reset_index().rename(columns={"index": "bps"}))
        return pd.concat(records, ignore_index=True)

    full = table_for()
    full = full.join(pd.DataFrame(meta).T, on="method")
    RESULTS.mkdir(parents=True, exist_ok=True)
    full.to_csv(RESULTS / "tables.csv", index=False)

    like = []
    for variant in ML_VARIANTS:
        dates = signals[variant].index[(signals[variant] != 0).any(axis=1)]
        table = table_for(dates)
        table["window"] = variant
        table["window_dates"] = len(dates)
        like.append(table)
        print(f"like-for-like window ({variant}): {len(dates)} dates")
    pd.concat(like, ignore_index=True).to_csv(RESULTS / "tables_like_for_like.csv", index=False)

    (RESULTS / "DECISION_RULE.md").write_text(RULE)
    print(full[full["bps"] == HEADLINE_BPS].to_string(index=False))
    print(f"\nwrote {RESULTS}/tables.csv, tables_like_for_like.csv, DECISION_RULE.md")

    headline = full[full["bps"] == HEADLINE_BPS][
        ["method", "sharpe", "dsr", "trials", "signal_coverage"]
    ]
    print("\ndecision-window inputs (ticket 34 applies the rule):")
    print(headline.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
