"""vectorbt as an independent cross-checker of the engine's account path.

Not load-bearing: vbt re-derives the daily return path (exposure drift, fees on
traded notional, compounding) with its own numba accounting. Tolerances are
loose (1e-3 relative) because vbt sizes orders against the current portfolio
value while the engine charges bps on the exposure delta — a small basis drift
is inherent to the two conventions; exact equality is not the contract.
"""

import numpy as np
import pandas as pd
import pytest
from conftest import continuous_wide

from systematic_futures.engine import account
from systematic_futures.engine.sizing import vol_target_positions
from systematic_futures.methods.tsmom import horizon_signal

TOL = {"rtol": 1e-3, "atol": 1e-5}


def crossover_signals(closes, window=10):
    return pd.DataFrame(
        np.where(closes > closes.rolling(window).mean(), 1.0, -1.0),
        index=closes.index,
        columns=closes.columns,
    )


def vbt_returns(wide, exposure, bps):
    vbt = pytest.importorskip("vectorbt")
    pf = vbt.Portfolio.from_orders(
        wide,
        exposure,
        size_type="target_percent",
        fees=bps / 1e4,
        init_cash=1.0,
        direction="both",
        freq="D",
    )
    return pf.returns().to_numpy()


def test_engine_account_matches_vectorbt_with_overlay_and_costs():
    wide = continuous_wide(seed=11)
    e = vol_target_positions(crossover_signals(wide), wide, vol_target=0.10, cap=1.0, lookback=30)
    np.testing.assert_allclose(
        account(e, wide, bps=2.0).to_numpy(), vbt_returns(wide, e, 2.0), **TOL
    )


def test_engine_account_matches_vectorbt_without_costs():
    wide = continuous_wide(seed=21)
    e = vol_target_positions(crossover_signals(wide), wide, vol_target=0.10, cap=1.0, lookback=30)
    np.testing.assert_allclose(
        account(e, wide, bps=0.0).to_numpy(), vbt_returns(wide, e, 0.0), **TOL
    )


def test_tsmom_sleeve_sweep_agrees_with_vectorbt():
    """The spec's TSMOM parameter sweep, cross-checked sleeve by sleeve: to vbt each
    horizon is its own strategy, so a sign or scale error in one sleeve cannot hide
    behind the ensemble average."""
    wide = continuous_wide(seed=31)
    for lookback in (10, 21, 63):
        e = vol_target_positions(
            horizon_signal(wide, lookback), wide, vol_target=0.10, cap=1.0, lookback=30
        )
        np.testing.assert_allclose(
            account(e, wide, bps=2.0).to_numpy(),
            vbt_returns(wide, e, 2.0),
            err_msg=f"{lookback}d sleeve disagrees with vectorbt",
            **TOL,
        )
