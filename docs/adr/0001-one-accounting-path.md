# One accounting path through the pipeline seam

Every method, and every parameter search, is measured by crossing
`engine.run(method, data) → metrics table`; nothing restates the vol overlay, the held
position path, the cost charge or the metric definitions, and the engine exposes none of them
(`account` derives the held path itself instead of accepting one). Chosen because the ML
tuning objective and the committed comparison tables must be the same arithmetic: the objective
used to score its inner validation block through a private copy of the pipeline, which could
drift from the metric the decision rule reads, with no test between them. The cost, measured at
real panel scale (3800 × 16), is that the seam computes the whole metrics table per trial —
turnover, Sortino, max drawdown and DSR columns the tuner never reads — which is milliseconds
against a LightGBM fit, accepted for the guarantee.

## Considered options

- **A private scoring path in the tuning objective** — rejected: two accounting paths, no test
  between them, and the search that picks the reported parameters scored by different code than
  the tables that report them.
- **`account(..., held=...)`**, an already-resolved held path, so the trade charged and the trade
  the reported turnover counts were provably one object — rejected: it widens the interface for
  one caller (12 of 13 call sites ignore it) to save 0.26 ms of a 2.18 ms call. The agreement is
  the pure function `traded_notional(held_positions(exposure, closes))`, identical on both sides.

## Scope: which interface may carry a family's inputs

The refusal above is about the *engine's* interface, and it holds: nothing an engine call could derive
for itself is accepted as an argument. The *comparison table*'s interface is a different seam and takes
the opposite decision — `comparison.table_for` receives each family as a `protocol.Method` carrying the
trial count its selection evaluated, rather than a callable plus a parallel `trials` mapping. The reason
is this ADR's own: three call sites threaded that mapping by hand, the number sets the DSR bar the rule
is read against, and a family could omit it and still report an undeflated DSR. Narrow where widening
would save one caller a derivation it can do itself; explicit where the value is an input no caller can
derive.
