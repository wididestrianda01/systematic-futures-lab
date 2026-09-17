# Decision rule (pre-declared in the plan, before this read's numbers existed)

**Rule.** Each ML variant (6a `ml_defaults`, 6b `ml_tuned`) must beat the
`tsmom` benchmark on **decision-window Deflated Sharpe Ratio after costs**
(the 2 bps row of the tables) to be reported as a winner.
The decision window is the walk-forward **out-of-fold** folds inside
develop+validate: the single out-of-sample read (2022 → 2024Q1) stays untouched
until it is taken, so it cannot be the decision window without spending the touch.

A variant that fails this test is documented as a **failed challenger** with the
numbers that failed it, not retuned until it passes.

**Reads.** `tables.csv` is the pre-declared primary read: every method on the
identical window, and the read the rule is applied to. `tables_like_for_like.csv`
re-measures every family on each ML variant's own covered dates, a robustness
read, not a second decision surface: it exists because a signal that is flat by
construction on part of the window has its Sharpe diluted by roughly
sqrt(coverage), and the harness asserts both reads return the same verdict
before either is reported.

**Multiple-testing inputs.** The DSR column is deflated by the configurations
each variant's selection actually evaluated: 6a searched none
(trials = 1), 6b searched folds x trials = 5 x 20 configurations
(trials = 100), counted whether or not a fold's search was later
skipped for lack of data.

**Scope.** One comparison set, one accounting path, two reads of it. The applied
outcome is `DECISION.md`; the interpretation stays for later.
