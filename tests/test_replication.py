"""MOP replication: the paper's formula, its formation timing, and its cross-section rule.

These pin the three things that decide whether a replication number means
anything — the volatility weights of Eq. (1), the month the signal is formed in
versus the month it is held (getting this backwards is a look-ahead and roughly
doubles the Sharpe), and MOP's S_t (a contract not yet in the strategy is absent
from the average, not averaged in as a flat zero).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from systematic_futures.replication import (
    AVAILABLE,
    TABLE1,
    UNAVAILABLE,
    VOL_COM,
    VOL_SCALE,
    VOL_TARGET,
    ex_ante_vol,
    factor_returns,
    live_instruments,
    mop_weights,
)

DAYS = 800


def alternating_panel(symbols=("A", "B"), n: int = DAYS) -> pd.DataFrame:
    """Deterministic ±1% walk: every return is exactly +0.01 or -0.01."""
    idx = pd.bdate_range("2010-01-01", periods=n)
    return pd.DataFrame(
        {s: 100.0 * np.cumprod(1.0 + 0.01 * np.resize([1, -1], n)) for s in symbols}, index=idx
    )


def rising_panel(n: int = DAYS) -> pd.DataFrame:
    """One symbol compounding 0.1%/session — a signal that is long from its first lookback on."""
    idx = pd.bdate_range("2010-01-01", periods=n)
    return pd.DataFrame({"A": 100.0 * 1.001 ** np.arange(n)}, index=idx)


def paper_sigma(x: np.ndarray, delta: float, scale: int) -> np.ndarray:
    """MOP Eq. (1) transcribed with loops: rbar_t is the same weighted mean, at its own date.

    Written out rather than borrowed from pandas so the test does not depend on
    `ewm`'s recursion, initialisation or NaN handling — those are what the
    implementation is being checked for.
    """
    out = np.full(len(x), np.nan)
    mean = np.full(len(x), np.nan)
    for t in range(len(x)):
        weights = delta ** np.arange(t, -1, -1)
        weights = weights / weights.sum()  # the paper's weights, renormalised over the history
        mean[t] = (weights * x[: t + 1]).sum()
        out[t] = np.sqrt(scale * (weights * (x[: t + 1] - mean[: t + 1]) ** 2).sum())
    return out


def test_ex_ante_vol_is_the_papers_weighted_sum_of_squared_returns():
    idx = pd.bdate_range("2010-01-01", periods=400)
    walk = 0.01 * np.resize([1, -1], len(idx))
    rets = pd.DataFrame({"A": walk, "B": walk / 2}, index=idx)
    sigma = ex_ante_vol(rets)
    expected = paper_sigma(walk, delta=VOL_COM / (1 + VOL_COM), scale=VOL_SCALE)

    for t in (250, 399):
        assert sigma.iloc[t, 0] == pytest.approx(expected[t], rel=1e-9)
    assert np.isnan(sigma["A"].iloc[:59]).all()  # unused until the estimate has 60 observations
    assert not np.isnan(sigma["A"].iloc[59])


def test_weights_hold_the_signal_formed_at_the_previous_month_end():
    closes = rising_panel()
    weights = mop_weights(closes)["A"]

    month_ends = closes.groupby(closes.index.to_period("M")).tail(1)
    month_ends.index = month_ends.index.to_period("M")
    trailing = month_ends / month_ends.shift(12) - 1.0  # written out again, not imported

    assert weights.loc["2010-01-01":"2011-01-31"].isna().all()  # no 12-month history yet
    for period, weight in weights.dropna().items():
        formed_in = period.to_period("M") - 1
        if pd.isna(trailing.get(formed_in.iloc[0] if hasattr(formed_in, "iloc") else formed_in)):
            continue
        assert np.sign(weight) == np.sign(trailing[formed_in])
        assert abs(weight) > 0


def test_signal_formed_from_a_month_end_ignores_that_months_later_sessions():
    """Perturbing a month's own sessions must not move that month's holding."""
    closes = alternating_panel(("A",))
    base = mop_weights(closes)
    perturbed = closes.copy()
    perturbed.loc["2012-06-04":"2012-06-29"] *= 1.5  # sessions inside the June holding
    moved = mop_weights(perturbed)

    june = base.loc["2012-06-04":"2012-06-29"].index
    pd.testing.assert_frame_equal(base.loc[june], moved.loc[june])  # June holds May's signal
    assert not np.allclose(
        base.loc["2012-07-02":"2012-07-31"], moved.loc["2012-07-02":"2012-07-31"]
    )


def test_exposure_sizes_each_instrument_to_the_target_volatility():
    closes = alternating_panel()
    weights = mop_weights(closes, vol_target=VOL_TARGET)
    sigma = ex_ante_vol(closes.pct_change())
    july = weights.loc["2012-07-02"]
    expected = (VOL_TARGET / sigma.loc["2012-06-29"]).abs()

    assert july.abs()["A"] == pytest.approx(expected["A"], rel=1e-12)
    assert july.abs()["B"] == pytest.approx(expected["B"], rel=1e-12)
    assert july.abs().nunique() == 1  # one constant size held through the month
    assert weights.loc["2012-07-02":"2012-07-31"].abs().nunique().eq(1).all()


def test_a_contract_that_is_not_yet_in_the_strategy_does_not_dilute_the_factor():
    early = alternating_panel(("A",), n=DAYS)
    late = alternating_panel(("B",), n=400)
    late.index = pd.bdate_range(early.index[400], periods=400)
    closes = pd.concat([early, late], axis=1)

    live = live_instruments(closes)
    book = factor_returns(closes)
    solo = per_symbol_returns(closes)

    assert live.loc[closes.index[300]] == 1  # only A is in the strategy
    assert live.loc[closes.index[-1]] == 2  # B has joined by the end of the panel
    # with B absent the factor is A's own return — not half of it
    assert book.loc[closes.index[300]] == pytest.approx(solo.loc[closes.index[300], "A"])
    # and once B is in, it is the two-instrument mean
    assert book.loc[closes.index[-1]] == pytest.approx(solo.loc[closes.index[-1]].mean())


def per_symbol_returns(closes: pd.DataFrame) -> pd.DataFrame:
    """The summands `factor_returns` averages, taken from the engine rather than recomputed."""
    from systematic_futures.data.panels import held_positions
    from systematic_futures.engine import account

    weights = mop_weights(closes)
    return account(weights, closes).where(held_positions(weights, closes).shift(1).notna())


def test_universe_boundary_is_explicit():
    """Every Table 1 row is mapped to a free series or says why not, with no stem reused."""
    assert len(TABLE1) == 55  # 24 commodity + 9 equity + 13 bond + 9 currency
    assert len(AVAILABLE) + len(UNAVAILABLE) == len(TABLE1)
    assert len(AVAILABLE) == len({i.stem for i in TABLE1 if i.stem})
    for instrument in TABLE1:
        assert (instrument.stem is None) != (instrument.why == "")
