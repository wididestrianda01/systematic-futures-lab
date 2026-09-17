# The out-of-sample read (2022 → 2024Q1)

**One touch.** These are the only numbers in the project measured on dates after 2021-12-31. The decision read in `results/decision/DECISION.md` stands as it was decided; what follows is out-of-time evidence about it, not a reopening.

Rule: the pre-declared rule of `results/decision/DECISION_RULE.md`, applied here to the 2 bps row of the primary read on the OOT window.

Benchmark `tsmom`: Sharpe 0.55, DSR 0.7966.

| variant | Sharpe | DSR (primary) | beats benchmark | DSR (own dates) | beats benchmark | OOT coverage |
|---|---|---|---|---|---|---|
| 6a `ml_defaults` | 0.07 | 0.5433 | no | 0.5533 | no | 99% |
| 6b `ml_tuned` | 0.39 | 0.02606 | no | 0.02936 | no | 99% |

**Out-of-time outcome: the rule fails for both variants** on the OOT window: the decision-read winners do not carry their edge past 2021, and that is the finding, not a reason to retune.

**Protocol.** The walk-forward schedule is the decision read's, extended across the whole frozen panel; each fold is refit on prior data only and 6b's search runs inside its fold's training set, so no fold saw an out-of-sample date before predicting it. The fold covering 2022 trains entirely inside develop+validate. Trial counts are unchanged as multiple-testing inputs: 6a searched nothing (trials = 1), 6b = folds x trials (5 x 20 = 100).

**Reading.** What the OOT numbers mean (regime attribution, the trend drought the benchmark carries into 2022, what the ML variants generalized) is interpretation, and it stays with the findings memo. Deliberately not written here.

Robustness read (`tables_like_for_like.csv`): the verdict is asserted identical on both reads before this file is written.
