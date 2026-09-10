"""Deflated Sharpe Ratio — known-input fixture (Bailey & López de Prado 2014)."""

from statistics import NormalDist

import numpy as np
import pandas as pd

from systematic_futures.engine.core import run
from systematic_futures.engine.metrics import deflated_sharpe

_NORM = NormalDist()


def seeded_returns():
    return pd.Series(np.random.default_rng(3).normal(0.0008, 0.01, 500))


def test_dsr_known_inputs():
    r = seeded_returns()
    # pinned against the closed-form computed independently below
    np.testing.assert_allclose(deflated_sharpe(r, 10), 0.9204222355716409, rtol=1e-12)


def test_dsr_without_trials_equals_psr():
    r = seeded_returns()
    n = len(r)
    sr = r.mean() / r.std(ddof=1)
    g3, g4 = r.skew(), r.kurt() + 3.0
    var_unit = 1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr**2
    psr = _NORM.cdf(sr * np.sqrt(n - 1) / np.sqrt(var_unit))
    np.testing.assert_allclose(deflated_sharpe(r, 1), psr, rtol=1e-12)


def test_dsr_deflates_with_more_trials_and_rejects_noise():
    r = seeded_returns()
    d10, d50 = deflated_sharpe(r, 10), deflated_sharpe(r, 50)
    assert d10 > d50  # more candidate strategies, harsher deflation
    loser = pd.Series(np.random.default_rng(4).normal(-0.001, 0.01, 500))
    assert deflated_sharpe(loser, 10) < 0.01  # negative-Sharpe strategy rejected


def test_dsr_column_in_seam_table():
    idx = pd.bdate_range("2020-01-01", periods=60)
    closes = pd.DataFrame(
        {"X": 100.0 * np.exp(np.cumsum(np.random.default_rng(5).normal(0.0002, 0.01, 60)))},
        index=idx,
    )
    table = run(
        lambda c: pd.DataFrame(1.0, index=c.index, columns=c.columns),
        closes,
        bps_grid=(2.0,),
        trials=5,
    )
    assert "dsr" in table.columns
    assert 0.0 <= table.loc[2.0, "dsr"] <= 1.0
