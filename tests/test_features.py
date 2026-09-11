"""Ticket 28 — the ML feature panel: point-in-time by construction.

Two things are pinned: a couple of features' exact values on a series whose
properties are hand-known, and the as-of guard — perturb the future, and every
feature value at or before the perturbation date must be bit-identical.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from systematic_futures.ml.features import FEATURES, feature_panel, forward_label


def known_panel(n: int = 400) -> pd.DataFrame:
    """A grows exactly 1%/day, B falls exactly 1%/day: ret_5 and the cross-sectional
    ranks are hand-computable, and each sits on a known side of its own average."""
    idx = pd.bdate_range("2015-01-01", periods=n)
    return pd.DataFrame(
        {"A": 100.0 * 1.01 ** np.arange(n), "B": 100.0 * 0.99 ** np.arange(n)}, index=idx
    )


def seeded_panel(n: int = 400, symbols=("A", "B", "C"), seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n)
    rets = rng.normal(scale=0.01, size=(n, len(symbols)))
    return pd.DataFrame(100.0 * np.cumprod(1.0 + rets, axis=0), index=idx, columns=list(symbols))


def test_panel_is_tidy_with_the_declared_features():
    closes = known_panel(n=120)
    panel = feature_panel(closes)
    assert list(panel.columns) == list(FEATURES)
    assert panel.index.names == ["date", "symbol"]
    assert len(panel) == len(closes) * closes.shape[1]


def test_hand_calculated_return_and_rank_features():
    closes = known_panel()
    panel = feature_panel(closes)
    last = panel.loc[closes.index[-1]]

    assert np.isclose(last.loc["A", "ret_5"], 1.01**5 - 1.0)
    assert np.isclose(last.loc["B", "ret_5"], 0.99**5 - 1.0)
    assert last.loc["A", "mom_rank"] == 1.0
    assert last.loc["B", "mom_rank"] == -1.0
    assert last.loc["A", "ma_dist_21"] > 0 > last.loc["B", "ma_dist_21"]


def test_warmup_features_are_nan_not_zero_filled():
    closes = known_panel(n=120)
    panel = feature_panel(closes)
    first = panel.loc[closes.index[0]]
    assert first[["ret_21", "vol_21", "mom_rank"]].isna().all().all()
    assert (panel["ret_252"].isna()).all()  # 120 days never fill a 252-day window


def test_features_do_not_look_ahead():
    closes = seeded_panel()
    basis = closes.pct_change(21)  # trailing by construction: a stand-in basis panel
    cut = closes.index[-60]

    before = feature_panel(closes, basis)
    perturbed, basis_perturbed = closes.copy(), basis.copy()
    perturbed.loc[closes.index > cut] *= 3.0
    basis_perturbed.loc[basis.index > cut] *= 3.0
    after = feature_panel(perturbed, basis_perturbed)

    past = before.loc[before.index.get_level_values("date") <= cut]
    pd.testing.assert_frame_equal(past, after.loc[after.index.get_level_values("date") <= cut])


def test_forward_label_reads_only_its_own_window():
    closes = seeded_panel()
    horizon = 5
    labels = forward_label(closes, horizon=horizon)
    t = closes.index[-30]
    symbol = "A"

    inside = closes.copy()  # a change inside (t, t+h] must move the label
    inside.loc[closes.index > t] *= 1.5
    assert not np.isclose(labels.loc[(t, symbol)], forward_label(inside, horizon).loc[(t, symbol)])

    beyond = closes.copy()  # a change strictly after t+h must not
    beyond.loc[closes.index >= t + pd.Timedelta(days=12)] *= 1.5
    assert np.isclose(labels.loc[(t, symbol)], forward_label(beyond, horizon).loc[(t, symbol)])
