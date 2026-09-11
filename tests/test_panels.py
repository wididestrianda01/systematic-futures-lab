"""Panel measurement on the real thing: a ragged union calendar.

The frozen panel is 16 roots with staggered history starts and quote holes
(CME Sunday rows, holidays) — ~44% of its cells are empty. Every return and
trailing window must be measured over the symbol's own observations: a plain
`pct_change` on the union grid turns each hole into two lost returns and any
window crossing a hole into NaN, which silently starves trailing-window signals
and the vol overlay. These tests pin the per-symbol semantics; on a dense grid
they reduce to the pandas equivalents the method fixtures already cover.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from systematic_futures.data.panels import (
    consecutive_returns,
    develop_validate,
    forward_return,
    held_positions,
    trailing_mean,
    trailing_return,
    trailing_std,
)
from systematic_futures.engine import account
from systematic_futures.engine.metrics import traded_notional

HOLE = slice(20, 23)  # B does not quote these three sessions


def ragged_panel(n: int = 60) -> pd.DataFrame:
    idx = pd.bdate_range("2020-01-01", periods=n)
    frame = pd.DataFrame(
        {
            "A": 100.0 * 1.001 ** np.arange(n),
            "B": 100.0 * 1.002 ** np.arange(n),
        },
        index=idx,
    )
    frame.iloc[HOLE, frame.columns.get_loc("B")] = np.nan
    return frame


def test_union_grid_pct_change_loses_returns_across_a_hole():
    """The red case the per-symbol measurement replaces."""
    closes = ragged_panel()
    naive = closes.pct_change()
    assert naive["B"].iloc[HOLE].isna().all()
    assert np.isnan(naive["B"].iloc[23])  # the observation after the hole is poisoned too
    assert np.isclose(naive["A"].iloc[23], 0.001)  # A's grid is untouched


def test_consecutive_returns_span_each_symbols_own_observations():
    closes = ragged_panel()
    rets = consecutive_returns(closes)
    assert np.isclose(rets["B"].iloc[23], closes["B"].iloc[23] / closes["B"].iloc[19] - 1.0)
    assert rets["B"].iloc[HOLE].isna().all()
    assert np.isclose(rets["A"].iloc[23], 0.001)


def test_trailing_windows_count_the_symbols_own_observations():
    closes = ragged_panel()
    assert np.isclose(
        trailing_return(closes, 2)["B"].iloc[23], closes["B"].iloc[23] / closes["B"].iloc[18] - 1.0
    )
    assert not np.isnan(trailing_std(consecutive_returns(closes), 5).loc[closes.index[23], "B"])
    mean_a = closes["A"].iloc[3:24].mean()
    assert np.isclose(trailing_mean(closes, 21).loc[closes.index[23], "A"], mean_a)


def test_engine_earns_the_return_spanning_a_quote_hole():
    closes = ragged_panel()
    constant = pd.DataFrame(1.0, index=closes.index, columns=closes.columns)
    rets = account(constant, closes)

    spanning = closes["B"].iloc[23] / closes["B"].iloc[19] - 1.0
    assert np.isclose(rets["B"].iloc[23], spanning)  # not zero: the hole must not eat the return
    assert rets["B"].iloc[HOLE].fillna(0.0).eq(0.0).all()  # nothing happens without a quote

    # The overlay's own shape: no exposure where the symbol has no quote. The held
    # position must survive the gap rather than silently going flat.
    gap = constant.copy()
    gap.iloc[HOLE, gap.columns.get_loc("B")] = 0.0
    held = account(gap, closes)
    assert np.isclose(held["B"].iloc[23], spanning)

    churn = traded_notional(held_positions(gap, closes))
    assert np.isclose(churn["B"].iloc[HOLE].sum() + churn["B"].iloc[23], 0.0)  # no phantom trading


def test_forward_return_spans_each_symbols_own_observations():
    closes = ragged_panel()
    ahead = forward_return(closes, 2)
    # From B's last observation before the hole, two of B's own observations ahead is
    # the date after the hole's return, not the hole's calendar neighbour.
    assert np.isclose(ahead["B"].iloc[19], closes["B"].iloc[24] / closes["B"].iloc[19] - 1.0)
    assert ahead["B"].iloc[20:23].isna().all()  # B has no observation there, so no return


def test_develop_validate_bounds_both_ends():
    """The window contract: both ends applied, and a panel that does not extend past
    them is an error — the derived panel starts in the 1970s, so an end-bounded slice
    alone would run every metric on pre-develop history."""
    panel = pd.DataFrame({"X": 1.0}, index=pd.bdate_range("2005-01-03", "2025-12-31"))
    window = develop_validate(panel)
    assert window.index.min() == pd.Timestamp("2010-01-01")
    assert window.index.max() == pd.Timestamp("2021-12-31")

    with pytest.raises(ValueError, match="develop start"):
        develop_validate(pd.DataFrame({"X": 1.0}, index=pd.bdate_range("2010-01-04", "2021-12-31")))
    with pytest.raises(ValueError, match="OOT window"):
        develop_validate(pd.DataFrame({"X": 1.0}, index=pd.bdate_range("2005-01-03", "2021-12-31")))
