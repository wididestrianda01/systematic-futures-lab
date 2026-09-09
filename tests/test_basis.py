import numpy as np
import pandas as pd

from systematic_futures.data.basis import compute_basis
from systematic_futures.data.roll_calendar import build_roll_calendar
from systematic_futures.data.synthetic import (
    make_synthetic_contract_prices,
    make_synthetic_definitions,
)


def test_constant_carry_gives_constant_basis():
    days = pd.bdate_range("2024-01-01", periods=3)
    raw = pd.concat(
        [
            pd.DataFrame({"date": days, "raw_symbol": "F", "close": 100.0}),
            pd.DataFrame({"date": days, "raw_symbol": "N", "close": 102.0}),
        ]
    )
    cal = pd.DataFrame({"date": days, "symbol": "X", "front": "F", "next": "N"})
    basis = compute_basis(raw, cal)
    np.testing.assert_allclose(basis["basis"], 0.02, rtol=1e-15)
    assert list(basis.columns) == ["date", "symbol", "front_close", "next_close", "basis"]


def test_missing_leg_excluded():
    days = pd.bdate_range("2024-01-01", periods=3)
    raw = pd.DataFrame(
        {
            "date": [days[0], days[1], days[0], days[1], days[2]],
            "raw_symbol": ["F", "F", "N", "N", "N"],
            "close": [100.0, 100.0, 102.0, 102.0, 100.0],
        }
    )
    cal = pd.DataFrame(
        {
            "date": days,
            "symbol": "X",
            "front": ["F", "F", "N"],
            "next": ["N", "N", "M"],
        }
    )
    basis = compute_basis(raw, cal)
    assert len(basis) == 2  # day 2's next leg (M) is missing → excluded
    assert basis["basis"].notna().all()
    np.testing.assert_allclose(basis["basis"], 0.02, rtol=1e-15)


def test_generated_data_has_full_coverage():
    defs = make_synthetic_definitions(("ES",), start="2020-01", n_months=6)
    raw = make_synthetic_contract_prices(defs, start="2019-12-01", end="2021-06-30", seed=5)
    cal = build_roll_calendar(defs, n_bd=5, start="2019-12-15")
    basis = compute_basis(raw, cal)
    assert len(basis) == len(cal)
    assert basis["basis"].abs().max() < 0.10  # premium structure keeps basis sane
