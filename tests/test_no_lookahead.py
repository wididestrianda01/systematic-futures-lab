"""No-look-ahead contract guard — regression test at the accounting seam.

Hand-built fixture with an explicit expected path: a signal that turns on at
day t must first earn day t+1's return. A same-day implementation earns day
t's return on day t's signal — caught exactly.
"""

import numpy as np
import pandas as pd

from systematic_futures.engine import account


def test_signal_first_earns_the_next_return():
    idx = pd.bdate_range("2020-01-01", periods=5)
    closes = pd.DataFrame({"X": [100.0, 110.0, 90.0, 105.0, 95.0]}, index=idx)
    # flat at t0, long from t1 on
    sig = pd.DataFrame({"X": [0.0, 1.0, 1.0, 1.0, 1.0]}, index=idx)
    r = account(sig, closes)["X"]
    raw = closes["X"].pct_change()
    # as-of: day0 flat (nothing earned, nothing held); day1 earns sig(t0)=0;
    # day t earns sig(t-1) * R(t) for t >= 2
    expected = [0.0, 0.0, raw.iloc[2], raw.iloc[3], raw.iloc[4]]
    np.testing.assert_allclose(r.to_numpy(), expected, rtol=1e-15, atol=0)


def test_lookahead_implementation_is_caught():
    """Sanity on the guard itself: the same-day (buggy) alignment produces a
    different — non-zero on day 1 — path, i.e. this test would go red on a
    look-ahead regression (demonstrated when the guard was written)."""
    idx = pd.bdate_range("2020-01-01", periods=5)
    closes = pd.DataFrame({"X": [100.0, 110.0, 90.0, 105.0, 95.0]}, index=idx)
    sig = pd.DataFrame({"X": [0.0, 1.0, 1.0, 1.0, 1.0]}, index=idx)
    r = account(sig, closes)["X"]
    raw = closes["X"].pct_change()
    same_day_buggy = sig["X"] * raw  # what a look-ahead bug would produce
    assert r.iloc[1] == 0.0
    assert abs(same_day_buggy.iloc[1] - 0.10) < 1e-12
