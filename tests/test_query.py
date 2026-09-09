import pandas as pd

from systematic_futures.data.query import (
    COVERAGE_QUERY,
    OI_SUMMARY_QUERY,
    ROLL_COUNTS_QUERY,
    query,
)


def build_store(tmp_path):
    dates = pd.bdate_range("2020-01-01", periods=5)
    ohl = pd.DataFrame(
        {"date": dates, "raw_symbol": "ESH0", "open": 1.0, "high": 1.0, "low": 1.0,
         "close": 1.0, "volume": 1}
    )
    (tmp_path / "raw" / "glbx" / "ohlcv-1d").mkdir(parents=True)
    ohl.to_parquet(tmp_path / "raw" / "glbx" / "ohlcv-1d" / "ES.parquet")
    (tmp_path / "raw" / "iceus" / "ohlcv-1d").mkdir(parents=True)
    ohl.to_parquet(tmp_path / "raw" / "iceus" / "ohlcv-1d" / "KC.parquet")
    (tmp_path / "raw" / "glbx" / "statistics").mkdir(parents=True)
    pd.DataFrame(
        {"date": dates, "raw_symbol": "ESH0", "open_interest": [10, 11, 12, 13, 14]}
    ).to_parquet(tmp_path / "raw" / "glbx" / "statistics" / "ES.parquet")
    (tmp_path / "raw" / "glbx" / "definition").mkdir(parents=True)
    pd.DataFrame(
        {"raw_symbol": ["ESH0"], "expiration": [dates[0]], "last_trade_date": [dates[0]]}
    ).to_parquet(tmp_path / "raw" / "glbx" / "definition" / "ES.parquet")
    (tmp_path / "derived").mkdir()
    pd.DataFrame(
        {"date": dates, "symbol": "ES",
         "front": ["ESH0"] * 4 + ["ESH1"], "next": ["ESH1"] * 4 + ["ESH2"]}
    ).to_parquet(tmp_path / "derived" / "roll_calendar.parquet")
    (tmp_path / "derived" / "continuous").mkdir()
    pd.DataFrame(
        {"date": dates, "symbol": "ES", "contract": "ESH0", "open": 1.0, "high": 1.0,
         "low": 1.0, "close": 1.0, "volume": 1}
    ).to_parquet(tmp_path / "derived" / "continuous" / "ES.parquet")
    (tmp_path / "derived" / "basis").mkdir()
    pd.DataFrame(
        {"date": dates, "symbol": "ES", "front_close": 1.0, "next_close": 2.0, "basis": 1.0}
    ).to_parquet(tmp_path / "derived" / "basis" / "ES.parquet")
    return tmp_path


def test_coverage_per_root_and_year(tmp_path):
    store = build_store(tmp_path)
    df = query(store, COVERAGE_QUERY)
    assert set(df["root"]) == {"ES", "KC"}
    assert (df["rows"] == 5).all()


def test_roll_counts(tmp_path):
    store = build_store(tmp_path)
    df = query(store, ROLL_COUNTS_QUERY)
    assert df.set_index("symbol").at["ES", "rolls"] == 1


def test_oi_summary(tmp_path):
    store = build_store(tmp_path)
    df = query(store, OI_SUMMARY_QUERY)
    assert df.loc[0, "root"] == "ES"
    assert df.loc[0, "mean_oi"] == 12.0
