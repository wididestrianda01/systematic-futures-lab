import json
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from conftest import continuous_wide

from systematic_futures.engine import account, curve, run
from systematic_futures.engine.metrics import ANN, max_drawdown, sharpe, sortino
from systematic_futures.engine.sizing import vol_target_positions

GOLDEN = Path(__file__).parent / "golden" / "engine_core.json"


def constant_long(closes):
    return pd.DataFrame(1.0, index=closes.index, columns=closes.columns)


# --- seam accounting (ticket 13) -------------------------------------------


def test_constant_long_earns_continuous_returns_including_rolls():
    wide = continuous_wide()
    r = account(constant_long(wide), wide)
    raw = wide.pct_change()
    np.testing.assert_allclose(r.iloc[1:].to_numpy(), raw.iloc[1:].to_numpy(), rtol=1e-12)


def test_metrics_hand_calculated():
    r = pd.Series([0.01, -0.02, 0.03])
    # mean = 0.02/3; std(ddof=1) = 0.02516611...; sortino downside dev from losses only
    np.testing.assert_allclose(sharpe(r), 4.205259864302774, rtol=1e-12)
    np.testing.assert_allclose(sortino(r), 9.165151389911678, rtol=1e-12)
    # equity [1.01, 0.9898, 1.019394]: deepest fall is 0.9898/1.01 - 1 = -2%
    assert abs(max_drawdown(r) - -0.02) < 1e-12


def test_seam_metrics_shape_and_alignment_guard():
    wide = continuous_wide(("ES",), seed=7)
    table = run(constant_long(wide), wide, bps_grid=(0.0, 2.0, 10.0))
    assert list(table.columns) == ["sharpe", "sortino", "max_dd", "turnover", "dsr"]
    assert table.index.tolist() == [0.0, 2.0, 10.0]
    assert table.loc[10.0, "sharpe"] < table.loc[0.0, "sharpe"]
    with pytest.raises(ValueError):
        run(constant_long(wide).iloc[1:], wide)


# --- vol overlay (ticket 14) ------------------------------------------------


def test_vol_overlay_hand_calculated():
    idx = pd.bdate_range("2020-01-01", periods=4)
    closes = pd.DataFrame({"X": [100.0, 101.0, 100.0, 102.0]}, index=idx)
    sig = pd.DataFrame(1.0, index=idx, columns=["X"])
    e = vol_target_positions(sig, closes, vol_target=0.10, cap=1.0, lookback=2)["X"]
    # returns: r1=1%, r2=100/101-1, r3=2%; trailing 2-day std (ddof=1) annualized
    r1, r2, r3 = 0.01, 100.0 / 101.0 - 1.0, 0.02
    sd2 = abs(r1 - r2) / sqrt(2) * sqrt(ANN)
    sd3 = abs(r2 - r3) / sqrt(2) * sqrt(ANN)
    expected = [0.0, 0.0, min(1.0, 0.10 / sd2), min(1.0, 0.10 / sd3)]
    np.testing.assert_allclose(e.to_numpy(), expected, rtol=1e-12)


def test_position_caps_clip():
    idx = pd.bdate_range("2020-01-01", periods=40)
    closes = pd.DataFrame({"X": 100.0 * np.exp(np.cumsum(np.full(40, 0.001)))}, index=idx)
    sig = pd.DataFrame(3.0, index=idx, columns=["X"])
    e = vol_target_positions(sig, closes, vol_target=0.10, cap=1.0, lookback=2)
    assert e["X"].abs().max() == 1.0


# --- cost model (ticket 15) -------------------------------------------------


def test_cost_model_hand_calculated():
    idx = pd.bdate_range("2020-01-01", periods=3)
    closes = pd.DataFrame({"X": [100.0, 110.0, 90.0]}, index=idx)
    e = pd.DataFrame({"X": [1.0, 1.0, -1.0]}, index=idx)
    r = account(e, closes, bps=2.0)["X"]
    # entry day: flat -> +1 costs 2e-4; hold day earns 10%; flip day: +1 -> -1
    # trades 2 units at 2 bps, and earns the held contract's -18.18%
    np.testing.assert_allclose(
        r.to_numpy(), [-2e-4, 0.10, (90.0 / 110.0 - 1.0) - 4e-4], rtol=1e-15, atol=1e-18
    )


def test_cost_sensitivity_rows():
    wide = continuous_wide(("ES",), seed=11)
    table = run(constant_long(wide), wide)  # default grid 0/2/5/10 bps
    assert table.index.tolist() == [0.0, 2.0, 5.0, 10.0]
    assert table["sharpe"].is_monotonic_decreasing
    assert table["turnover"].nunique() == 1  # turnover independent of cost level


def test_turnover_prices_the_cost_column():
    """The reported turnover is the base the charge is taken on, per unit of book capital:
    the mean daily return given up between two cost levels is exactly `turnover x bps x 1e-4`.
    Summing traded notional across symbols instead of averaging it reports `n_symbols` times
    the drag the column explains (16x on the project panel)."""
    wide = continuous_wide()
    cl = constant_long(wide)
    table = run(cl, wide, bps_grid=(0.0, 2.0))
    drag = curve(cl, wide, bps=0.0).mean() - curve(cl, wide, bps=2.0).mean()
    assert abs(drag - table.loc[2.0, "turnover"] * 2e-4) < 1e-15

    # and in Sharpe terms, to the accuracy of the (slightly different) cost-level stds
    ann_vol = curve(cl, wide).std(ddof=1) * sqrt(ANN)
    implied = (table.loc[0.0, "sharpe"] - table.loc[2.0, "sharpe"]) * ann_vol / (2e-4 * ANN)
    assert abs(implied / table.loc[2.0, "turnover"] - 1.0) < 0.02


def test_date_mask_evaluates_on_those_days_only():
    """The like-for-like read: metrics on a subset of days, same accounting path."""
    wide = continuous_wide(("ES",), seed=13)
    sig = constant_long(wide)
    mask = wide.index[-20:]

    masked = run(sig, wide, dates=mask)
    manual = sharpe(account(sig, wide).mean(axis=1).reindex(mask))
    assert np.isclose(masked.loc[0.0, "sharpe"], manual)
    assert not np.isclose(masked.loc[0.0, "sharpe"], run(sig, wide).loc[0.0, "sharpe"])
    full_mask = run(sig, wide, dates=wide.index)
    pd.testing.assert_frame_equal(full_mask, run(sig, wide))


def test_curve_reads_the_same_accounting_as_run():
    """The notebook's plot input and the table's Sharpe are one accounting path: the curve at a
    cost level is exactly the daily return series whose Sharpe `run` reports, masked the same way."""
    wide = continuous_wide(("ES",), seed=17)
    sig = constant_long(wide)
    daily = curve(sig, wide, vol_target=0.10, bps=2.0)
    assert np.isclose(sharpe(daily), run(sig, wide, vol_target=0.10).loc[2.0, "sharpe"])

    mask = wide.index[-20:]
    masked = curve(sig, wide, vol_target=0.10, bps=2.0, dates=mask)
    assert np.isclose(
        sharpe(masked), run(sig, wide, vol_target=0.10, dates=mask).loc[2.0, "sharpe"]
    )
    assert masked.index.equals(mask)


# --- golden seam outputs -----------------------------------------------------


def test_golden_seam_outputs():
    wide = continuous_wide()
    cl = constant_long(wide)
    calls = {
        "no_overlay": {"bps_grid": (0.0,)},
        "vol_overlay": {"bps_grid": (0.0,), "vol_target": 0.10},
        "costs": {"vol_target": 0.10},
    }
    golden = json.loads(GOLDEN.read_text())
    assert set(golden) == set(calls)
    for section, kwargs in calls.items():
        table = run(cl, wide, **kwargs)
        for bps_str, metrics in golden[section]["rows"].items():
            for metric, want in metrics.items():
                got = table.loc[float(bps_str), metric]
                assert abs(got - want) < 1e-12, (section, bps_str, metric, got, want)
