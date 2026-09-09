import numpy as np
import pandas as pd

from systematic_futures.data.continuous import back_adjust
from systematic_futures.data.roll_calendar import build_roll_calendar
from systematic_futures.data.synthetic import (
    make_synthetic_contract_prices,
    make_synthetic_definitions,
)

OHLC = ["open", "high", "low", "close"]


def hand_raw(days, prices_per_contract):
    frames = []
    for contract, close in prices_per_contract.items():
        frames.append(
            pd.DataFrame(
                {
                    "date": days,
                    "raw_symbol": contract,
                    "open": close - 0.5,
                    "high": close + 1.0,
                    "low": close - 1.0,
                    "close": close,
                    "volume": 1,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def hand_scenario():
    """4 contracts, constant +3 premium per successor; front flips on days[12] and days[22]."""
    days = pd.bdate_range("2024-01-01", periods=30)
    t = np.arange(len(days), dtype=float)
    defs = pd.DataFrame(
        {
            "symbol": "A",
            "raw_symbol": ["A1", "A2", "A3", "A4"],
            "expiration": [days[16], days[26], days[29], days[29]],
            "last_trade_date": [days[14], days[24], days[28], days[29]],
        }
    )
    raw = hand_raw(days, {"A1": 100 + t, "A2": 103 + t, "A3": 106 + t, "A4": 109 + t})
    cal = build_roll_calendar(defs, n_bd=2, start=days[5])
    return days, raw, cal


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
    _, raw, cal = hand_scenario()
    cont = back_adjust(raw, cal)
    got = cont.sort_values("date")["close"].pct_change().iloc[1:].reset_index(drop=True)
    want = target_returns(raw, cal).reset_index(drop=True)
    np.testing.assert_allclose(got, want, rtol=1e-12)


def test_adjustment_removes_the_planted_gap():
    _, raw, cal = hand_scenario()
    cont = back_adjust(raw, cal).sort_values("date").reset_index(drop=True)
    naive, held = naive_returns(raw, cal)
    want = target_returns(raw, cal).reset_index(drop=True)
    flips = held["front"].ne(held["front"].shift()).to_numpy()[1:]
    assert flips.sum() == 2
    # the unadjusted stitch carries the basis jump; the adjusted series does not
    assert not np.allclose(naive[flips], want[flips])
    np.testing.assert_allclose(naive[~flips], want[~flips], rtol=1e-12)
    # newest period (from the last flip onward) carries factor 1.0
    np.testing.assert_allclose(
        cont["close"].iloc[-4:], held["close"].iloc[-4:], rtol=1e-15
    )


def test_generated_data_roundtrip():
    defs = make_synthetic_definitions(("ES",), start="2020-01", n_months=6)
    raw = make_synthetic_contract_prices(defs, start="2019-12-01", end="2021-06-30", seed=5)
    cal = build_roll_calendar(defs, n_bd=5, start="2019-12-15")
    cont = back_adjust(raw, cal)
    got = cont.sort_values("date")["close"].pct_change().iloc[1:].reset_index(drop=True)
    want = target_returns(raw, cal).reset_index(drop=True)
    np.testing.assert_allclose(got, want, rtol=1e-9)
    # OHLC adjusted consistently: (high − low) scales by the same factor as (close ratio)
    assert list(cont.columns) == [
        "date", "symbol", "contract", "open", "high", "low", "close", "volume"
    ]
