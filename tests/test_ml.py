"""The ML variants at the seam.

Pinned here: out-of-fold-only predictions (the ML analogue of the engine's
no-look-ahead guard — perturb a fold's own test block and nothing before it may
move, for the default and the tuned variant), a planted learnable relationship
the model must actually find, determinism, the honest trial count, and the fact
that both variants run through the one shared seam.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from systematic_futures.engine import run
from systematic_futures.ml import (
    inner_split,
    leakage_violations,
    lgbm_defaults,
    lgbm_tuned,
    purged_walk_forward,
)
from systematic_futures.protocol import Protocol

PROTOCOL_TEST = Protocol(horizon=5, splits=3, embargo=10, n_trials=2)
N = 2400


def ar_panel(n: int = N, symbols=("A", "B", "C", "D", "E", "F"), phi: float = 0.8, seed: int = 3):
    """Returns with genuine positive autocorrelation: a learnable momentum effect,
    which is the point — the model must find it out of sample."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2008-01-01", periods=n)
    shocks = rng.normal(scale=0.01, size=(n, len(symbols)))
    rets = np.zeros_like(shocks)
    for t in range(1, n):
        rets[t] = phi * rets[t - 1] + shocks[t]
    return pd.DataFrame(100.0 * np.cumprod(1.0 + rets, axis=0), index=idx, columns=list(symbols))


def basis_panel(closes: pd.DataFrame) -> pd.DataFrame:
    """Stand-in carry panel: a trailing 21-day return, never forward-looking."""
    return closes.pct_change(21).fillna(0.0)


def fixture(n: int = N):
    closes = ar_panel(n)
    return closes, basis_panel(closes)


def fold_schedule(index: pd.DatetimeIndex):
    return purged_walk_forward(
        index,
        n_splits=PROTOCOL_TEST.splits,
        horizon=PROTOCOL_TEST.horizon,
        embargo=PROTOCOL_TEST.embargo,
    )


def walk_forward_cut(closes: pd.DataFrame) -> pd.Timestamp:
    """End of the middle fold's test block: everything at or before it is predicted
    by folds whose training data lies entirely before it."""
    return fold_schedule(closes.index)[1].test[-1]


@pytest.mark.parametrize(
    ("build", "expected_trials"),
    [
        (lambda basis: lgbm_defaults(basis, protocol=PROTOCOL_TEST), 1),
        (lambda basis: lgbm_tuned(basis, protocol=PROTOCOL_TEST), 3 * 2),
    ],
    ids=["6a defaults", "6b tuned"],
)
def test_every_variant_predicts_out_of_fold_only_and_counts_its_trials(build, expected_trials):
    """The contract both variants get from the one walk-forward scaffold: flat outside
    the folds' own test blocks, and a trial count that counts every configuration the
    variant's selection actually evaluated."""
    closes, basis = fixture()
    method = build(basis)
    assert method.trials == expected_trials

    signals = method.signal(closes)
    cut = walk_forward_cut(closes)
    assert (signals.loc[:cut] != 0).any().any(), "fixture must yield out-of-fold signals"

    perturbed = closes.copy()
    perturbed.loc[closes.index > cut] *= 1.5
    pd.testing.assert_frame_equal(signals.loc[:cut], method.signal(perturbed).loc[:cut])


def test_defaults_learn_a_planted_momentum_effect():
    closes, basis = fixture()
    method = lgbm_defaults(basis, protocol=PROTOCOL_TEST)
    signals = method.signal(closes)
    forward = closes.shift(-PROTOCOL_TEST.horizon) / closes - 1.0

    ics = []
    for date, row in signals.iterrows():
        realized = forward.loc[date]
        if row.abs().sum() > 0 and row.notna().sum() > 2 and realized.notna().sum() > 2:
            ics.append(row.corr(realized, method="spearman"))
    assert np.nanmean(ics) > 0.1, (
        f"out-of-sample rank IC {np.nanmean(ics):.3f} — effect not learned"
    )


def test_defaults_run_through_the_seam_deterministically():
    closes, basis = fixture(n=1800)
    method = lgbm_defaults(basis, protocol=PROTOCOL_TEST)
    assert method.trials == 1

    first = run(method.signal, closes, vol_target=PROTOCOL_TEST.vol_target)
    second = run(method.signal, closes, vol_target=PROTOCOL_TEST.vol_target)
    pd.testing.assert_frame_equal(first, second)
    assert list(first.index) == [0.0, 2.0, 5.0, 10.0]
    assert first["sharpe"].notna().all() and first["turnover"].notna().all()

    signals = method.signal(closes)
    assert signals.index.equals(closes.index) and signals.columns.equals(closes.columns)
    assert (signals.iloc[:252] == 0.0).all().all()  # warmup stays flat, never back-filled
    assert not signals.isna().any().any()


def test_the_tuned_search_is_deterministic():
    closes, basis = fixture()
    method = lgbm_tuned(basis, protocol=PROTOCOL_TEST)
    pd.testing.assert_frame_equal(method.signal(closes), method.signal(closes))


def test_inner_tuning_split_stays_inside_the_fold_training_set():
    """The tuning protocol: the frames the search is handed come from `fold.train`
    alone, purged and embargoed, never from the fold's test block."""
    closes, _ = fixture(n=1800)
    for fold in fold_schedule(closes.index):
        inner = inner_split(fold, protocol=PROTOCOL_TEST)
        assert set(inner.train) <= set(fold.train)
        assert set(inner.test) <= set(fold.train)
        assert not set(inner.test) & set(fold.test)
        assert inner.test.min() > inner.train.max()  # validation follows the inner train
        assert not leakage_violations(
            [inner],
            fold.train,
            horizon=PROTOCOL_TEST.horizon,
            embargo=PROTOCOL_TEST.embargo,
        )


def test_missing_or_misaligned_basis_raises_instead_of_going_flat():
    closes, _ = fixture(n=1200)
    with pytest.raises(ValueError, match="complete feature/label samples"):
        lgbm_defaults(protocol=PROTOCOL_TEST).signal(closes)
    misaligned = basis_panel(closes).rename(columns={name: f"{name}_x" for name in closes.columns})
    with pytest.raises(ValueError, match="complete feature/label samples"):
        lgbm_defaults(misaligned, protocol=PROTOCOL_TEST).signal(closes)
