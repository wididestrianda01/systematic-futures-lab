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

HORIZON, SPLITS, EMBARGO = 5, 3, 10
N_TUNED_TRIALS = 2
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


def walk_forward_cut(closes: pd.DataFrame) -> pd.Timestamp:
    """End of the middle fold's test block: everything at or before it is predicted
    by folds whose training data lies entirely before it."""
    folds = purged_walk_forward(closes.index, n_splits=SPLITS, horizon=HORIZON, embargo=EMBARGO)
    return folds[1].test[-1]


@pytest.mark.parametrize(
    ("build", "expected_trials"),
    [
        (lambda basis: lgbm_defaults(basis, horizon=HORIZON, n_splits=SPLITS, embargo=EMBARGO), 1),
        (
            lambda basis: lgbm_tuned(
                basis, horizon=HORIZON, n_splits=SPLITS, embargo=EMBARGO, n_trials=N_TUNED_TRIALS
            ),
            SPLITS * N_TUNED_TRIALS,
        ),
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

    signals = method(closes)
    cut = walk_forward_cut(closes)
    assert (signals.loc[:cut] != 0).any().any(), "fixture must yield out-of-fold signals"

    perturbed = closes.copy()
    perturbed.loc[closes.index > cut] *= 1.5
    pd.testing.assert_frame_equal(signals.loc[:cut], method(perturbed).loc[:cut])


def test_defaults_learn_a_planted_momentum_effect():
    closes, basis = fixture()
    signals = lgbm_defaults(basis, horizon=HORIZON, n_splits=SPLITS, embargo=EMBARGO)(closes)
    forward = closes.shift(-HORIZON) / closes - 1.0

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
    method = lgbm_defaults(basis, horizon=HORIZON, n_splits=SPLITS, embargo=EMBARGO)
    assert method.trials == 1

    first = run(method, closes, vol_target=0.10)
    second = run(method, closes, vol_target=0.10)
    pd.testing.assert_frame_equal(first, second)
    assert list(first.index) == [0.0, 2.0, 5.0, 10.0]
    assert first["sharpe"].notna().all() and first["turnover"].notna().all()

    signals = method(closes)
    assert signals.index.equals(closes.index) and signals.columns.equals(closes.columns)
    assert (signals.iloc[:252] == 0.0).all().all()  # warmup stays flat, never back-filled
    assert not signals.isna().any().any()


def test_the_tuned_search_is_deterministic():
    closes, basis = fixture()
    method = lgbm_tuned(
        basis, horizon=HORIZON, n_splits=SPLITS, embargo=EMBARGO, n_trials=N_TUNED_TRIALS
    )
    pd.testing.assert_frame_equal(method(closes), method(closes))  # seeded search is reproducible


def test_inner_tuning_split_stays_inside_the_fold_training_set():
    """The tuning protocol: the frames the search is handed come from `fold.train`
    alone, purged and embargoed, never from the fold's test block."""
    closes, _ = fixture(n=1800)
    folds = purged_walk_forward(closes.index, n_splits=SPLITS, horizon=HORIZON, embargo=EMBARGO)
    for fold in folds:
        inner = inner_split(fold, horizon=HORIZON, embargo=EMBARGO)
        assert set(inner.train) <= set(fold.train)
        assert set(inner.test) <= set(fold.train)
        assert not set(inner.test) & set(fold.test)
        assert inner.test.min() > inner.train.max()  # validation follows the inner train
        assert not leakage_violations([inner], fold.train, horizon=HORIZON, embargo=EMBARGO)


def test_missing_or_misaligned_basis_raises_instead_of_going_flat():
    closes, basis = fixture(n=1200)
    with pytest.raises(ValueError, match="complete feature/label samples"):
        lgbm_defaults(horizon=HORIZON, n_splits=SPLITS, embargo=EMBARGO)(closes)
    misaligned = basis.rename(columns={name: f"{name}_x" for name in basis.columns})
    with pytest.raises(ValueError, match="complete feature/label samples"):
        lgbm_defaults(misaligned, horizon=HORIZON, n_splits=SPLITS, embargo=EMBARGO)(closes)
