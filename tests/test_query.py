import pandas as pd

from systematic_futures.data.query import (
    CONTINUOUS_COVERAGE_QUERY,
    COVERAGE_QUERY,
    ROLL_COUNTS_QUERY,
    query,
)

MP_COLUMNS = [
    "DATETIME",
    "CARRY",
    "CARRY_CONTRACT",
    "PRICE",
    "PRICE_CONTRACT",
    "FORWARD",
    "FORWARD_CONTRACT",
]


def mp_row(day, front_id, front_px):
    return {
        "DATETIME": f"{day:%Y-%m-%d %H:%M:%S}",
        "CARRY": front_px - 1,
        "CARRY_CONTRACT": front_id - 100,
        "PRICE": front_px,
        "PRICE_CONTRACT": front_id,
        "FORWARD": front_px + 2,
        "FORWARD_CONTRACT": front_id + 100,
    }


def build_store(tmp_path):
    dates = pd.bdate_range("2020-01-01", periods=5)
    mp = pd.DataFrame(
        [mp_row(d, 20200100, 100.0) for d in dates[:4]]
        + [mp_row(d, 20200200, 101.0) for d in dates[4:]]  # front rolls on the last day
    )
    mp["symbol"] = "ES"
    mp = mp[MP_COLUMNS + ["symbol"]]
    (tmp_path / "raw" / "multiple_prices").mkdir(parents=True)
    mp.to_csv(tmp_path / "raw" / "multiple_prices" / "SP500.csv", index=False)
    (tmp_path / "raw" / "roll_calendars").mkdir(parents=True)
    pd.DataFrame(
        {"contract": [20200100, 20200200], "roll_date": [str(dates[4].date()), ""]}
    ).to_csv(tmp_path / "raw" / "roll_calendars" / "SP500_rollcalendar.csv", index=False)
    (tmp_path / "derived").mkdir()
    pd.DataFrame(
        {
            "date": dates,
            "symbol": "ES",
            "front": [20200100] * 4 + [20200200],
            "next": [20200200] * 4 + [20200300],
        }
    ).to_parquet(tmp_path / "derived" / "roll_calendar.parquet")
    (tmp_path / "derived" / "continuous").mkdir()
    pd.DataFrame({"date": dates, "symbol": "ES", "contract": 20200100, "close": 1.0}).to_parquet(
        tmp_path / "derived" / "continuous" / "SP500.parquet"
    )
    (tmp_path / "derived" / "basis").mkdir()
    pd.DataFrame(
        {"date": dates, "symbol": "ES", "front_close": 100.0, "next_close": 102.0, "basis": 0.02}
    ).to_parquet(tmp_path / "derived" / "basis" / "SP500.parquet")
    return tmp_path


def test_coverage_per_instrument(tmp_path):
    store = build_store(tmp_path)
    df = query(store, COVERAGE_QUERY)
    assert df.loc[0, "instrument"] == "SP500"
    assert df.loc[0, "rows"] == 5
    assert str(df.loc[0, "first_date"])[:10] == "2020-01-01"
    assert str(df.loc[0, "last_date"])[:10] == "2020-01-07"


def test_roll_counts(tmp_path):
    store = build_store(tmp_path)
    df = query(store, ROLL_COUNTS_QUERY)
    assert df.set_index("symbol").at["ES", "rolls"] == 1


def test_continuous_coverage(tmp_path):
    store = build_store(tmp_path)
    df = query(store, CONTINUOUS_COVERAGE_QUERY)
    assert df.loc[0, "year"] == 2020
    assert df.loc[0, "rows"] == 5
