"""The comparison's two declared contracts: its protocol and its methods.

A **protocol** is everything a comparison was measured under — the walk-forward fold geometry, the
label horizon and search budget, the headline cost level, and the overlay. It is declared once here
and read by the phase scripts, the catalogue and the ML variants' tuning objective, so the search that
picks a variant's parameters and the tables that report them cannot be measured under different
arithmetic (ADR 0001). Changing a field is a *protocol* change, not a parameter change: the
out-of-sample read no longer extends the schedule the rule was decided under.

A **method** is one family in seam-compatible form: the wide continuous-close panel in, a signal table
out, plus the number of configurations its selection actually evaluated. That trial count is the
Deflated Sharpe Ratio's deflation input, so it belongs to the contract every consumer reads rather than
to an attribute a family may or may not remember to set.

The overlay's own numbers live in `engine.sizing`, the module that implements the overlay; the record
here carries the values this comparison chose.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

from systematic_futures.engine.sizing import CAP, VOL_TARGET

HORIZON = 5  # label length in sessions
SPLITS = 5  # walk-forward folds
EMBARGO = 10  # sessions excluded from training after each test block
N_TRIALS = 20  # search configurations per fold
HEADLINE_BPS = 2.0  # the cost level a decision is read at


@dataclass(frozen=True)
class Protocol:
    """The geometry, cost level and overlay one comparison was measured under."""

    horizon: int = HORIZON
    splits: int = SPLITS
    embargo: int = EMBARGO
    n_trials: int = N_TRIALS
    headline_bps: float = HEADLINE_BPS
    vol_target: float = VOL_TARGET
    cap: float = CAP


PROTOCOL = Protocol()
"""The comparison protocol the phase tables, the rule and the out-of-sample read all share."""

Signal = Callable[[pd.DataFrame], pd.DataFrame] | pd.DataFrame


@dataclass(frozen=True)
class Method:
    """One family at the pipeline seam: a signal source plus the trials its selection evaluated.

    `signal` is a callable (wide closes -> signal table) or a signal table itself; `trials` is what the
    family's selection actually evaluated — 1 for a family that searched nothing — counted whether or
    not a configuration was later discarded.
    """

    signal: Signal
    trials: int = 1
