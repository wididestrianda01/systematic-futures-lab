"""Per-family golden fixtures — each family's sign and scale pinned at the seam.

Hand-calculated on constructed panels whose properties are exact (constant
trailing vol, known return paths), so overlay scale, daily PnL, and turnover
verify without re-implementing the engine. Ticket 19: baseline; 20: TSMOM
horizons; 21: TSMOM ensemble.
"""

from math import sqrt

import numpy as np
import pandas as pd
import pytest
from conftest import continuous_wide

from systematic_futures.engine import account, run
from systematic_futures.engine.metrics import ANN, traded_notional
from systematic_futures.engine.sizing import vol_target_positions
from systematic_futures.methods import (
    baseline,
    carry,
    horizon_signal,
    seasonal_tilt,
    seasonality,
    tsmom,
    xs_momentum,
)
from systematic_futures.methods.tsmom import VOL_SCALAR

VOL_TARGET = 0.10
LOOKBACK = 30
EXPECTED_SD = 0.01 * sqrt(LOOKBACK / (LOOKBACK - 1))


def alternating_panel(n=45):
    """One symbol whose daily returns alternate exactly +-1%: every full 30-day
    trailing window holds 15 up / 15 down moves, so its ddof=1 std is
    EXPECTED_SD and the overlay scale is hand-known."""
    idx = pd.bdate_range("2020-01-01", periods=n)
    steps = np.where(np.arange(n) % 2 == 0, 1.01, 0.99)
    steps[0] = 1.0
    return pd.DataFrame({"X": 100.0 * np.cumprod(steps)}, index=idx)


# --- ticket 19: baseline -----------------------------------------------------


def test_baseline_signal_is_constant_long():
    wide = continuous_wide()
    sig = baseline(wide)
    assert (sig == 1.0).all().all()
    assert sig.index.equals(wide.index) and sig.columns.equals(wide.columns)


def test_baseline_overlay_scale_and_return_path_hand_calculated():
    closes = alternating_panel()
    rets = closes.pct_change()

    # Overlay spec written out by hand: 10% vol target over trailing-30d
    # annualized std, capped, flat until the window fills.
    expected_E = (
        (VOL_TARGET / (rets.rolling(LOOKBACK).std(ddof=1) * sqrt(ANN))).clip(-1.0, 1.0).fillna(0.0)
    )
    first_full = expected_E.index[LOOKBACK]  # 30 returns exist from here

    E = vol_target_positions(baseline(closes), closes, VOL_TARGET, 1.0, LOOKBACK)
    np.testing.assert_allclose(E.to_numpy(), expected_E.to_numpy(), rtol=1e-9)
    assert E.iloc[:LOOKBACK].abs().to_numpy().sum() == 0.0
    scale = VOL_TARGET / (EXPECTED_SD * sqrt(ANN))
    assert abs(E.iloc[LOOKBACK, 0] - scale) < 1e-9  # pinned scale, uncapped
    assert (E.loc[first_full:] > 0).all().all()  # long sign holds

    r = account(E, closes, 0.0).mean(axis=1)
    expected_r = (expected_E.shift(1) * rets).fillna(0.0)["X"]
    np.testing.assert_allclose(r.to_numpy(), expected_r.to_numpy(), rtol=1e-12)


def test_baseline_turnover_is_single_entry_trade():
    closes = alternating_panel()
    E = vol_target_positions(baseline(closes), closes, VOL_TARGET, 1.0, LOOKBACK)
    tn = traded_notional(E)["X"]
    assert tn.iloc[:LOOKBACK].sum() == 0.0  # flat until the window fills
    assert abs(tn.iloc[LOOKBACK] - E.iloc[LOOKBACK, 0]) < 1e-9  # one entry trade
    assert tn.iloc[LOOKBACK + 1 :].abs().sum() < 1e-9  # then zero (constant vol)


def test_baseline_through_seam():
    table = run(baseline, continuous_wide(), vol_target=VOL_TARGET)
    assert list(table.index) == [0.0, 2.0, 5.0, 10.0]
    assert list(table.columns) == ["sharpe", "sortino", "max_dd", "turnover", "dsr"]
    assert table.notna().all().all()


# --- ticket 20: TSMOM horizon signals ----------------------------------------


def monotonic_panel(step=1.005, n=90):
    """Strictly compounding panel: every daily return is exactly `step`."""
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame({"X": 100.0 * step ** np.arange(n)}, index=idx)


def test_horizon_sign_rising_falling_and_warmup():
    up = horizon_signal(monotonic_panel(), 21)
    assert up.iloc[:21].isna().all().all()  # NaN = not yet active (flat)
    assert (up.iloc[21:] == 1.0).all().all()  # rising: long, vol 0 clips to cap

    down = horizon_signal(
        pd.DataFrame(
            {"X": 100.0 * 0.995 ** np.arange(90)},
            index=pd.bdate_range("2020-01-01", periods=90),
        ),
        21,
    )
    assert (down.iloc[21:] == -1.0).all().all()  # falling: short
    assert down.iloc[:21].isna().all().all()


def test_horizon_zero_return_is_neutral():
    flat = horizon_signal(
        pd.DataFrame({"X": 100.0}, index=pd.bdate_range("2020-01-01", periods=60)), 21
    )
    assert flat.isna().all().all()  # NaN = no direction, engine holds flat


def test_horizon_vol_scaling_hand_calculated():
    # Alternating +4% / -3% daily returns: trailing 21d return is positive
    # (up days net over down days) while annualized vol is large enough that
    # 0.40/sigma stays uncapped — the scale is observable, not clipped.
    n = 60
    idx = pd.bdate_range("2020-01-01", periods=n)
    steps = np.where(np.arange(n) % 2 == 0, 1.04, 0.97)
    steps[0] = 1.0
    closes = pd.DataFrame({"X": 100.0 * np.cumprod(steps)}, index=idx)

    rets = closes.pct_change()
    expected = (
        np.sign(closes.pct_change(21)) * (VOL_SCALAR / (rets.rolling(21).std(ddof=1) * sqrt(ANN)))
    ).clip(-1.0, 1.0)

    sig = horizon_signal(closes, 21)
    np.testing.assert_allclose(sig.to_numpy(), expected.to_numpy(), rtol=1e-9)
    assert 0.0 < expected.iloc[21, 0] < 1.0  # uncapped: scale actually pinned


def test_horizon_signals_pass_seam_alignment():
    wide = continuous_wide()
    for lookback in (5, 10, 20, 21, 63, 252):
        run(horizon_signal(wide, lookback), wide, vol_target=VOL_TARGET)


# --- ticket 21: TSMOM ensemble ------------------------------------------------


def _trend_panel():
    """Three symbols, 300 days: A monotonic up (all six sleeves +1), B
    monotonic down (all six −1), C down then a 15-day late rise — its four
    short+1m sleeves flip +1 while the 63d/252d sleeves stay −1, pinning the
    1/6 weights."""
    n = 300
    idx = pd.bdate_range("2020-01-01", periods=n)
    steps_c = np.where(np.arange(n) < 285, 0.995, 1.015)
    steps_c[0] = 1.0
    crafted = 100.0 * np.cumprod(steps_c)
    return pd.DataFrame(
        {
            "A": 100.0 * 1.005 ** np.arange(n),
            "B": 100.0 * 0.995 ** np.arange(n),
            "C": crafted,
        },
        index=idx,
    )


def test_tsmom_ensemble_weights_hand_calculated():
    sig = tsmom(_trend_panel())
    last = sig.iloc[-1]
    assert last["A"] == pytest.approx(1.0)  # (+1 ×6)/6
    assert last["B"] == pytest.approx(-1.0)  # (−1 ×6)/6
    assert last["C"] == pytest.approx(2.0 / 6.0)  # (+1 ×4 −1 ×2)/6
    assert (sig.dropna().abs() <= 1.0).all().all()  # inside the signal band


def test_tsmom_ensemble_flat_until_longest_horizon():
    sig = tsmom(_trend_panel())
    assert sig.iloc[:252].isna().all().all()  # flat until all sleeves are live


def test_tsmom_through_seam():
    table = run(tsmom, continuous_wide(), vol_target=VOL_TARGET)
    assert list(table.columns) == ["sharpe", "sortino", "max_dd", "turnover", "dsr"]
    assert table.notna().all().all()


# --- ticket 22: XS momentum ---------------------------------------------------


def _xs_panel():
    """Four symbols, 320 days, constant distinct daily returns so the trailing
    252d ranking is exact and stable: A > B > C > D."""
    n = 320
    idx = pd.bdate_range("2020-01-01", periods=n)
    steps = {"A": 1.02, "B": 1.01, "C": 0.99, "D": 0.98}
    steps = {s: np.concatenate([[1.0], np.full(n - 1, v)]) for s, v in steps.items()}
    return pd.DataFrame({s: 100.0 * np.cumprod(v) for s, v in steps.items()}, index=idx)


def test_xs_momentum_ranks_and_signs_hand_calculated():
    sig = xs_momentum(_xs_panel())
    last = sig.iloc[-1]
    assert last["A"] == pytest.approx(1.0)  # strongest → +1
    assert last["B"] == pytest.approx(1.0 / 3.0)  # (3 − 2.5)/1.5
    assert last["C"] == pytest.approx(-1.0 / 3.0)  # (2 − 2.5)/1.5
    assert last["D"] == pytest.approx(-1.0)  # weakest → −1


def test_xs_momentum_ties_average_rank():
    n = 320
    idx = pd.bdate_range("2020-01-01", periods=n)
    tied = 100.0 * np.cumprod(np.concatenate([[1.0], np.full(n - 1, 0.98)]))
    steps_a = np.concatenate([[1.0], np.full(n - 1, 1.02)])
    panel = pd.DataFrame({"A": 100.0 * np.cumprod(steps_a), "B": tied, "C": tied}, index=idx)
    sig = xs_momentum(panel)
    last = sig.iloc[-1]
    assert last["B"] == last["C"] == pytest.approx(-0.5)  # tied ranks 1.5, 1.5


def test_xs_momentum_neutral_until_lookback_fills():
    sig = xs_momentum(_xs_panel())
    assert sig.iloc[:252].isna().all().all()


def test_xs_momentum_through_seam():
    table = run(xs_momentum, continuous_wide(), vol_target=VOL_TARGET)
    assert table.notna().all().all()


# --- ticket 23: carry ----------------------------------------------------------


def test_carry_sign_and_neutrality_hand_calculated():
    idx = pd.bdate_range("2020-01-01", periods=45)
    closes = pd.DataFrame({"X": 100.0}, index=idx)
    basis_long = pd.DataFrame(
        {
            "date": idx[[5, 10, 15]],
            "symbol": ["X", "X", "X"],
            "front_close": [100.0] * 3,
            "next_close": [98.0, 103.0, 100.0],
            "basis": [-0.02, 0.03, 0.0],
        }
    )
    sig = carry(basis_long)(closes)
    assert sig.iloc[5, 0] == 1.0  # backwardated front → long
    assert sig.iloc[10, 0] == -1.0  # contangoed front → short
    assert sig.iloc[15, 0] == 0.0  # flat basis → no position
    assert sig.drop(idx[[5, 10, 15]]).isna().all().all()  # missing leg → flat


def test_carry_through_seam():
    idx = pd.bdate_range("2020-01-01", periods=45)
    closes = alternating_panel(45)
    half = len(idx) // 2
    basis = pd.DataFrame(
        {
            "date": idx,
            "symbol": "X",
            "front_close": 100.0,
            "next_close": np.where(np.arange(len(idx)) < half, 101.0, 99.0),
            "basis": np.where(np.arange(len(idx)) < half, 0.01, -0.01),
        }
    )
    table = run(carry(basis), closes, vol_target=VOL_TARGET)
    assert table.notna().all().all()
    assert table["turnover"].iloc[0] > 0  # alternating basis flips positions


# --- ticket 24: seasonality standalone -----------------------------------------


def _season_panel():
    """Four and a half years, two symbols. March returns alternate (+3%, −1%)
    for A (mean +1%) and −3%/+1% for B (mean −1%); every other day is flat.
    Scores exist from the fourth March on (3 priors inside the 60m window)."""
    idx = pd.bdate_range("2020-01-01", "2024-06-30")
    up = np.where(np.arange(len(idx)) % 2 == 0, 0.03, -0.01)
    down = np.where(np.arange(len(idx)) % 2 == 0, -0.03, 0.01)
    rets = pd.DataFrame({"A": up, "B": down}, index=idx)
    rets[~(idx.month == 3)] = 0.0
    return (1.0 + rets).cumprod() * 100.0


def test_seasonality_signs_and_warmup():
    sig = seasonality(_season_panel())
    active = sig.loc["2023-03-10":"2023-03-20"]
    assert (active["A"] == 1.0).all()  # strong March → long
    assert (active["B"] == -1.0).all()  # weak March → short
    assert sig.loc["2022-03-15"].isna().all()  # only 2 priors → no score yet
    off = sig.loc["2023-01-17"]
    assert off["A"] == 0.0 and off["B"] == 0.0  # scored 0 → flat, not NaN


def test_seasonality_through_seam():
    table = run(seasonality, _season_panel(), vol_target=VOL_TARGET)
    assert table["turnover"].iloc[0] > 0  # monthly flip in and out of March
    assert table.loc[0.0, "sharpe"] > 0  # in-sample seasonality earns its mean


# --- ticket 25: seasonality tilt on trend ---------------------------------------


def _tilt_panel():
    """Five years, three symbols. Baseline daily return +2% (A, C) or −2% (B).
    March overrides: +2% for A and B (their own strong month, score > 0),
    −0.2% for C (mildly weak month, score < 0). Asserted on each year's first
    March day, where every horizon's trailing window is still dominated by
    baseline days, so the trend value is exactly ±1 and the tilt's 1.5×/0.5×
    is visible: A long × 1.5 (capped +1), B short × 1.5 (capped −1),
    C long × 0.5 = 0.5. From the fourth March the score exists; the third
    March still warms up (tilt inactive, trend untouched)."""
    idx = pd.bdate_range("2020-01-01", "2024-12-31")
    rets = pd.DataFrame({"A": 0.02, "B": -0.02, "C": 0.02}, index=idx)
    march = idx.month == 3
    rets.loc[march, "A"] = 0.02
    rets.loc[march, "B"] = 0.02
    rets.loc[march, "C"] = -0.002
    return (1.0 + rets).cumprod() * 100.0


def test_seasonal_tilt_amplify_dampen_and_warmup():
    sig = seasonal_tilt(_tilt_panel())
    first_march_2023 = sig.loc["2023-03-01"]
    assert first_march_2023["A"] == 1.0  # long × 1.5 → capped at +1
    assert first_march_2023["B"] == -1.0  # short × 1.5 → capped at −1
    assert first_march_2023["C"] == pytest.approx(0.5)  # long × 0.5 → damped
    # warmup: only 2 March priors in 2022, tilt inactive, trend untouched
    assert sig.loc["2022-03-01", "C"] == pytest.approx(1.0)


def test_seasonal_tilt_through_seam():
    table = run(seasonal_tilt, _season_panel(), vol_target=VOL_TARGET)
    assert table["turnover"].iloc[0] > 0  # the tilt churns positions
