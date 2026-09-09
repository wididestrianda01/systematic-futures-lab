import pandas as pd

from systematic_futures.data.roll_calendar import build_roll_calendar
from systematic_futures.data.synthetic import make_synthetic_definitions


def hand_definitions(n_days=30):
    """4 contracts, LTDs on days[14]/[24]/[28]/[29]; n_bd=2 → rolls on days[12]/[22]/[26]/[27]."""
    days = pd.bdate_range("2024-01-01", periods=n_days)
    return days, pd.DataFrame(
        {
            "symbol": "A",
            "raw_symbol": ["A1", "A2", "A3", "A4"],
            "expiration": [days[16], days[26], days[29], days[29]],
            "last_trade_date": [days[14], days[24], days[28], days[29]],
        }
    )


def test_front_flips_on_roll_dates():
    days, defs = hand_definitions()
    cal = build_roll_calendar(defs, n_bd=2, start=days[5])
    cal = cal.set_index("date")
    # coverage: business days from days[5] through the last roll, while a successor exists
    assert cal.index.min() == days[5]
    assert cal.index.max() == days[25]  # last day strictly before the final roll
    # flips happen exactly on the computed roll dates
    assert cal.loc[days[11], "front"] == "A1" and cal.loc[days[12], "front"] == "A2"
    assert cal.loc[days[21], "front"] == "A2" and cal.loc[days[22], "front"] == "A3"
    assert cal.loc[days[12], "next"] == "A3" and cal.loc[days[5], "next"] == "A2"
    # both legs always present
    assert cal["front"].notna().all() and cal["next"].notna().all()


def test_roll_dates_strictly_increasing():
    _, defs = hand_definitions()
    defs["symbol"] = "B"
    try:
        build_roll_calendar(defs.assign(last_trade_date=defs["last_trade_date"].iloc[::-1].to_numpy()), n_bd=2)
    except ValueError as e:
        assert "strictly increasing" in str(e)
    else:
        raise AssertionError("non-monotonic rolls must fail")


def test_generated_definitions_roundtrip():
    defs = make_synthetic_definitions(("ES",), start="2020-01", n_months=6)
    cal = build_roll_calendar(defs, n_bd=5)
    flips = (cal["front"] != cal["front"].shift()).sum() - 1  # first row is not a flip
    assert flips == len(defs) - 3  # first contract never front (grid starts at its roll), last has no successor
    expected_rolls = (defs["last_trade_date"] - pd.tseries.offsets.BDay(5)).dt.normalize()
    observed = cal["date"][cal["front"] != cal["front"].shift()].tolist()
    assert observed[1:] == expected_rolls.sort_values().tolist()[1:5]
