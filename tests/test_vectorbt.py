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


def crossover_signals(closes, window=10):
    return pd.DataFrame(
        np.where(closes > closes.rolling(window).mean(), 1.0, -1.0),
        index=closes.index,
        columns=closes.columns,
    )


def test_engine_account_matches_vectorbt_with_overlay_and_costs():
    vbt = pytest.importorskip("vectorbt")
    wide = continuous_wide(seed=11)
    e = vol_target_positions(crossover_signals(wide), wide, vol_target=0.10, cap=1.0, lookback=30)
    r_engine = account(e, wide, bps=2.0)
    pf = vbt.Portfolio.from_orders(
        wide,
        e,
        size_type="target_percent",
        fees=2.0 / 1e4,
        init_cash=1.0,
        direction="both",
        freq="D",
    )
    np.testing.assert_allclose(r_engine.to_numpy(), pf.returns().to_numpy(), rtol=1e-3, atol=1e-5)


def test_engine_account_matches_vectorbt_without_costs():
    vbt = pytest.importorskip("vectorbt")
    wide = continuous_wide(seed=21)
    e = vol_target_positions(crossover_signals(wide), wide, vol_target=0.10, cap=1.0, lookback=30)
    r_engine = account(e, wide, bps=0.0)
    pf = vbt.Portfolio.from_orders(
        wide,
        e,
        size_type="target_percent",
        fees=0.0,
        init_cash=1.0,
        direction="both",
        freq="D",
    )
    np.testing.assert_allclose(r_engine.to_numpy(), pf.returns().to_numpy(), rtol=1e-3, atol=1e-5)
