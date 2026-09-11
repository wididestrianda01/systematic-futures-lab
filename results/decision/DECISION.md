# the decision read decision (the pre-declared rule applied)

Rule: `DECISION_RULE.md`, applied to the 2 bps row of the primary read (`tables.csv`) on the develop+validate window (2010-01-01 → 2021-12-31).

Benchmark `tsmom`: Sharpe -0.08, DSR 0.3726.

| variant | Sharpe | DSR (primary) | beats benchmark | DSR (own dates) | beats benchmark |
|---|---|---|---|---|---|
| 6a `ml_defaults` | 0.80 | 0.9989 | yes | 0.9985 | yes |
| 6b `ml_tuned` | 1.55 | 0.9999 | yes | 0.9998 | yes |

**Verdict: the rule is met by 6a `ml_defaults`, 6b `ml_tuned`**, which beat the benchmark's DSR after costs and are reported as winners.

Robustness read (`tables_like_for_like.csv`, each variant on its own covered dates): the verdict is identical on both reads — asserted in the harness, not asserted here, because the two reads disagreeing must stop the run rather than be written up.

Interpretation — what the purged/embargoed folds and the DSR deflation actually changed versus a naive same-window split, and where the variants overfit — is ticket 34's, gated by the M7 reading canon (ticket 32). Deliberately not written here.
