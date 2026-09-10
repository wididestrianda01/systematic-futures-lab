import numpy as np
import pandas as pd

from systematic_futures.data.basis import compute_basis
from systematic_futures.data.roll_calendar import build_roll_calendar, extract_contract_prices
from systematic_futures.data.synthetic import make_synthetic_multiple_prices


def hand_mp():
    ts = pd.Timestamp("2024-01-02 23:00")
    return pd.DataFrame(
        {
            "DATETIME": [ts],
            "CARRY": [99.0],
            "CARRY_CONTRACT": [20231200],
            "PRICE": [100.0],
            "PRICE_CONTRACT": [20240100],
            "FORWARD": [102.0],
            "FORWARD_CONTRACT": [20240200],
            "symbol": "X",
        }
    )


def test_constant_carry_gives_constant_basis():
    mp = hand_mp()
    cal = build_roll_calendar(mp)
    basis = compute_basis(extract_contract_prices(mp), cal)
    assert len(basis) == 1
    np.testing.assert_allclose(basis["basis"], 0.02, rtol=1e-15)
    assert list(basis.columns) == ["date", "symbol", "front_close", "next_close", "basis"]


def test_missing_leg_excluded():
    mp = hand_mp()
    gap = mp.iloc[[0]].copy()
    gap["DATETIME"] = pd.Timestamp("2024-01-03 23:00")
    gap["FORWARD"] = np.nan  # next leg missing that day
    gap["FORWARD_CONTRACT"] = np.nan
    basis = compute_basis(
        extract_contract_prices(pd.concat([mp, gap])), build_roll_calendar(pd.concat([mp, gap]))
    )
    assert len(basis) == 1  # the gap day is excluded, never zero-filled
    assert basis["basis"].notna().all()


def test_generated_data_full_coverage():
    mp = make_synthetic_multiple_prices(("ES",), start="2020-01-01", end="2021-06-30", seed=5)
    cal = build_roll_calendar(mp)
    basis = compute_basis(extract_contract_prices(mp), cal)
    assert len(basis) == len(cal)
    assert basis["basis"].notna().all()
    assert basis["basis"].abs().max() < 0.10  # premium structure keeps basis sane
