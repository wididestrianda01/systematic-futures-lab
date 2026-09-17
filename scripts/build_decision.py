"""the decision read comparison: variants 6a/6b against the TSMOM benchmark through the shared seam.

Window: develop+validate (2010-01-01 → 2021-12-31, `data.panels.develop_validate`).
The walk-forward out-of-fold folds ARE the decision window; the single OOT read
(2022 → 2024Q1) stays untouched for its end-of-project touch in the out-of-sample read. The
comparison set (`catalogue`) and the table shape (`comparison`), so the families read and the decision read
cannot drift apart.

Two readings of the same numbers, both committed:

- `tables.csv` — the primary, pre-declared read: every method on the identical
  window, so the ML variants sit flat on the days their folds do not cover (the
  `signal_coverage` column says how much of the window that is).
- `tables_like_for_like.csv` — a robustness read: every method re-measured on each
  ML variant's own covered dates, so a variant is not scored on a diluted series
  against a fully populated benchmark. Same accounting path: the engine's `dates`
  mask. The run asserts both readings return the same verdict, so this read can
  never become a second decision surface (one table was asked for).

`DECISION_RULE.md` carries the pre-declared rule verbatim; `DECISION.md` records
that rule applied to these numbers. Trial counts are the honest multiple-testing
inputs: 6a searched nothing (trials = 1), 6b searched n_splits x n_trials
configurations per run. Interpretation (what purge/embargo and the DSR deflation
changed) is left for the interpretation pass, gated by the M7 canon.

Run: uv run python scripts/build_decision.py
"""

from __future__ import annotations

from pathlib import Path

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
from systematic_futures.data.panels import (
    DEVELOP_START,
    VALIDATE_END,
    develop_validate,
    load_frozen,
)
from systematic_futures.decision import require_same_verdict, verdict
from systematic_futures.protocol import PROTOCOL
from systematic_futures.reporting import (
    benchmark_line,
    family_names,
    outcome_split,
    outcome_table,
)

RESULTS = Path("results/phase4")

RULE = f"""# the decision read decision rule (pre-declared in the plan, before this phase's numbers existed)

**Rule.** Each ML variant (6a `ml_defaults`, 6b `ml_tuned`) must beat the
`{BENCHMARK}` benchmark on **decision-window Deflated Sharpe Ratio after costs**
— the {PROTOCOL.headline_bps:.0f} bps row of the tables — to be reported as a winner.
The decision window is the walk-forward **out-of-fold** folds inside
develop+validate: the single out-of-sample read (2022 → 2024Q1) stays untouched
until the out-of-sample read, so it cannot be the decision window without spending the touch.

A variant that fails this test is documented as a **failed challenger** with the
numbers that failed it — not retuned until it passes.

**Reads.** `tables.csv` is the pre-declared primary read: every method on the
identical window, and the read the rule is applied to. `tables_like_for_like.csv`
re-measures every family on each ML variant's own covered dates — a robustness
read, not a second decision surface: it exists because a signal that is flat by
construction on part of the window has its Sharpe diluted by roughly
sqrt(coverage), and the harness asserts both reads return the same verdict
before either is reported.

**Multiple-testing inputs.** The DSR column is deflated by the configurations
each variant's selection actually evaluated: 6a searched none
(trials = 1), 6b searched folds x trials = {PROTOCOL.splits} x {PROTOCOL.n_trials} configurations
(trials = {PROTOCOL.splits * PROTOCOL.n_trials}), counted whether or not a fold's search was later
skipped for lack of data.

**Scope.** One comparison set, one accounting path, two reads of it. The applied
outcome is `DECISION.md`; the interpretation stays for later (M7-gated).
"""


def outcome_doc(primary: dict, sensitivity: dict[str, dict]) -> str:
    winners, losers = outcome_split(primary, ML_VARIANTS)
    if winners:
        verdict_text = (
            "**Verdict: the rule is met by "
            + family_names(winners, LABELS)
            + "**, which beat the benchmark's DSR after costs and are reported as winners."
        )
        if losers:
            verdict_text += (
                " "
                + family_names(losers, LABELS)
                + " still failed and are documented as failed challengers."
            )
    else:
        verdict_text = (
            "**Verdict: the rule fails for both variants — documented as failed challengers**, "
            "with the numbers above, not retuned until they pass."
        )
    return (
        "# the decision read decision (the pre-declared rule applied)\n\n"
        f"Rule: `DECISION_RULE.md`, applied to the {PROTOCOL.headline_bps:.0f} bps row of the primary "
        f"read (`tables.csv`) on the develop+validate window "
        f"({DEVELOP_START.date()} → {VALIDATE_END.date()}).\n\n"
        f"{benchmark_line(primary, BENCHMARK)}\n\n"
        f"{outcome_table(primary, sensitivity, ML_VARIANTS, LABELS)}\n\n"
        f"{verdict_text}\n\n"
        "Robustness read (`tables_like_for_like.csv`, each variant on its own covered dates): "
        "the verdict is identical on both reads — asserted in the harness, not asserted here, "
        "because the two reads disagreeing must stop the run rather than be written up.\n\n"
        "Interpretation — what the purged/embargoed folds and the DSR deflation actually "
        "changed versus a naive same-window split, and where the variants overfit — is left for later, "
        "gated by the M7 reading canon. Deliberately not written here.\n"
    )


def main() -> int:
    wide, basis_wide = load_frozen()
    window = develop_validate(wide)

    walk_forward_folds(window.index, protocol=PROTOCOL)  # the schedule every family fits on

    methods = resolved(full_set(basis_wide, protocol=PROTOCOL), window)
    meta = protocol_meta(methods, window.index)

    RESULTS.mkdir(parents=True, exist_ok=True)
    full = table_for(methods, window, protocol=PROTOCOL).join(meta, on="method")
    full.to_csv(RESULTS / "tables.csv", index=False)

    like_tables = like_for_like(
        methods, window, variants=ML_VARIANTS, protocol=PROTOCOL, dates=window.index
    )
    like_tables.to_csv(RESULTS / "tables_like_for_like.csv", index=False)

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

    (RESULTS / "DECISION_RULE.md").write_text(RULE)
    (RESULTS / "DECISION.md").write_text(outcome_doc(primary, sensitivity))

    headline = headline_rows(full, PROTOCOL.headline_bps)[
        ["sharpe", "dsr", "trials", "signal_coverage"]
    ]
    print(headline.to_string())
    print(f"\nwrote {RESULTS}/tables.csv, tables_like_for_like.csv, DECISION_RULE.md, DECISION.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
