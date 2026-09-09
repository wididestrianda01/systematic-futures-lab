import numpy as np
import pandas as pd

from systematic_futures.data.continuous import back_adjust
from systematic_futures.data.roll_calendar import build_roll_calendar, extract_contract_prices
from systematic_futures.data.synthetic import make_synthetic_multiple_prices

OHLC = ["open", "high", "low", "close"]


def target_returns(raw, cal):
    """The position holds front(t) from t−1 close through t close, so the true daily
    return is close(front(t), t) / close(front(t), t−1) — including roll days."""
    wide = raw.pivot(index="date", columns="raw_symbol", values="close").sort_index()
    cal = cal.sort_values("date").reset_index(drop=True)
    rets = []
    for i in range(1, len(cal)):
        prev, cur = cal.iloc[i - 1], cal.iloc[i]
        leg = cur["front"]
        rets.append(wide.at[cur["date"], leg] / wide.at[prev["date"], leg] - 1)
    return pd.Series(rets, index=cal["date"].iloc[1:])


def naive_returns(raw, cal):
    """Unadjusted stitch: pct_change of whatever contract the calendar holds that day."""
    held = cal.merge(
        raw[["date", "raw_symbol", "close"]],
        left_on=["date", "front"], right_on=["date", "raw_symbol"],
    ).sort_values("date")
    return held["close"].pct_change().iloc[1:].reset_index(drop=True), held


def test_continuous_return_equals_held_return_through_rolls():
    mp = make_synthetic_multiple_prices(("ES",), start="2020-01-01", end="2021-06-30", seed=5)
    cal = build_roll_calendar(mp)
    raw = extract_contract_prices(mp)
    cont = back_adjust(raw, cal)
    got = cont.sort_values("date")["close"].pct_change().iloc[1:].reset_index(drop=True)
    want = target_returns(raw, cal).reset_index(drop=True)
    np.testing.assert_allclose(got, want, rtol=1e-9)


def test_adjustment_removes_the_planted_gap():
    mp = make_synthetic_multiple_prices(("ES",), start="2020-01-01", end="2020-06-30", seed=7)
    cal = build_roll_calendar(mp)
    raw = extract_contract_prices(mp)
    cont = back_adjust(raw, cal).sort_values("date").reset_index(drop=True)
    naive, held = naive_returns(raw, cal)
    want = target_returns(raw, cal).reset_index(drop=True)
    flips = held["front"].ne(held["front"].shift()).to_numpy()[1:]
    assert flips.sum() >= 5  # monthly contracts, half-year window
    # the unadjusted stitch carries the roll gap; the adjusted series does not
    assert not np.allclose(naive[flips], want[flips])
    np.testing.assert_allclose(naive[~flips], want[~flips], rtol=1e-9)
    # newest period is unadjusted (cumulative factor = 1)
    np.testing.assert_allclose(
        cont["close"].iloc[-4:].to_numpy(), held["close"].iloc[-4:].to_numpy(), rtol=1e-12
    )


def test_columns_and_no_gaps():
    mp = make_synthetic_multiple_prices(("ES",), seed=3)
    cal = build_roll_calendar(mp)
    raw = extract_contract_prices(mp)
    cont = back_adjust(raw, cal)
    assert list(cont.columns) == ["date", "symbol", "contract", "close"]
    assert cont["close"].notna().all()
    assert len(cont) == len(cal)
