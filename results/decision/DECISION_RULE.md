# the decision read decision rule (pre-declared in the spec, before this phase's numbers existed)

**Rule.** Each ML variant (4a `ml_defaults`, 4b `ml_tuned`) must beat the
`tsmom` benchmark on **decision-window Deflated Sharpe Ratio after costs**
— the 2 bps row of the tables — to be reported as a winner.
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
(trials = 1), 4b searched folds x trials = 5 x 20 configurations
(trials = 100), counted whether or not a fold's search was later
skipped for lack of data.

Outcome and interpretation: ticket 34 (blocked by the M7 reading gate).
