import pandas as pd

from systematic_futures.data.synthetic import OHLCV_COLUMNS, make_synthetic_ohlcv


def test_deterministic_same_seed():
    a = make_synthetic_ohlcv(seed=7)
    b = make_synthetic_ohlcv(seed=7)
    pd.testing.assert_frame_equal(a, b)


def test_different_seed_differs():
    a = make_synthetic_ohlcv(seed=7)
    b = make_synthetic_ohlcv(seed=8)
    assert not a.equals(b)


def test_invariants():
    df = make_synthetic_ohlcv(seed=3)
    assert list(df.columns) == OHLCV_COLUMNS
    assert (df[["open", "high", "low", "close"]] > 0).all().all()
    assert (df["high"] >= df[["open", "close"]].max(axis=1)).all()
    assert (df["low"] <= df[["open", "close"]].min(axis=1)).all()
    assert (df["volume"] > 0).all()


def test_parquet_roundtrip(tmp_path):
    df = make_synthetic_ohlcv(seed=1)
    path = tmp_path / "panel.parquet"
    df.to_parquet(path)
    back = pd.read_parquet(path)
    pd.testing.assert_frame_equal(back, df)
