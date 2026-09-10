import pandas as pd
import pytest

from systematic_futures.data.roll_calendar import build_roll_calendar
from systematic_futures.data.synthetic import make_synthetic_multiple_prices


def month_index(contract_id: int) -> int:
    """YYYYMM00 → absolute month index (for successor checks across year ends)."""
    return (contract_id // 10000) * 12 + (contract_id // 100) % 100


def test_calendar_is_a_projection_of_observed_legs():
    mp = make_synthetic_multiple_prices(("ES",), start="2020-01-01", end="2021-06-30")
    cal = build_roll_calendar(mp)
    days = pd.bdate_range("2020-01-01", "2021-06-30")
    assert list(cal.columns) == ["date", "symbol", "front", "next"]
    assert len(cal) == len(days)  # intraday dupes collapsed, legacy null-front row dropped
    assert cal["date"].is_monotonic_increasing
    assert cal["front"].notna().all() and cal["next"].notna().all()


def test_front_transitions_match_generator_rule():
    mp = make_synthetic_multiple_prices(("A",), start="2020-01-01", end="2020-06-30")
    cal = build_roll_calendar(mp)
    by_date = cal.set_index("date")
    # before the roll: front = January's contract; from the first bd ≥ Jan 18: February's
    assert by_date.loc[pd.Timestamp("2020-01-17"), "front"] == 20200100
    cutoff = pd.bdate_range("2020-01-18", "2020-01-22")[0]
    assert by_date.loc[cutoff, "front"] == 20200200
    # forward is always the front's successor month (year-boundary safe)
    delta = by_date["next"].map(month_index) - by_date["front"].map(month_index)
    assert (delta == 1).all()


def test_non_monotonic_front_raises():
    mp = make_synthetic_multiple_prices(
        ("A",), start="2020-01-01", end="2020-03-31", intraday=False
    )
    bad = mp.iloc[:-20].copy()
    bad.loc[bad.index[-1], "PRICE_CONTRACT"] = 20190100  # front jumps backwards
    with pytest.raises(ValueError, match="non-decreasing"):
        build_roll_calendar(bad)


def test_legacy_null_front_row_dropped():
    mp = make_synthetic_multiple_prices(
        ("A",), start="2020-01-01", end="2020-02-29", intraday=False
    )
    assert mp["PRICE_CONTRACT"].isna().sum() == 1  # the legacy leading row exists
    cal = build_roll_calendar(mp)
    assert cal["date"].min() == pd.Timestamp("2020-01-01")
