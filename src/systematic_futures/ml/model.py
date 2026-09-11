"""ML variants — 6a LightGBM defaults, 6b LightGBM + Optuna pruned TPE.

Both are seam-compatible methods: bind the panels at construction (`basis=`,
the carry family's convention), then the callable maps the wide continuous
close panel to a signal table — `run(method, closes)` needs no change.

Signals are **out-of-fold only**: each walk-forward fold fits on its purged,
embargoed training dates and predicts that fold's test dates; every date
outside a test block, and every date whose features are still warming up, is
flat. A date's raw predictions are ranked across the cross-section and centred
(the same map XS momentum uses: strongest long +1, weakest short -1), so the
shared 10% vol overlay owns sizing and the comparison stays apples-to-apples.

Labels are the forward h-day return scaled by trailing vol as-of t
(`features.forward_label`) — the only forward-looking object in the ML package,
and never a model input.

Multiple testing: the Deflated Sharpe Ratio needs the number of configurations
the selection actually evaluated, so `method.trials` rides on the returned
callable — 1 for 6a (defaults, no search), folds x n_trials for 6b. 6b's
tuning happens inside a fold, on a purged inner split of that fold's training
data: a test block never informs its own search.
"""

from __future__ import annotations

from collections.abc import Callable

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMRegressor

from systematic_futures.engine.core import account
from systematic_futures.engine.metrics import sharpe
from systematic_futures.engine.sizing import vol_target_positions
from systematic_futures.methods.ranking import centered_rank
from systematic_futures.ml.cv import Fold, purged_walk_forward
from systematic_futures.ml.features import FEATURES, feature_panel, forward_label

HORIZON = 5  # label length in trading days
N_SPLITS = 5
EMBARGO = 10
SEED = 7
MIN_TRAIN_ROWS = 250
VOL_TARGET = 0.10
CAP = 1.0
TUNING_BPS = 2.0  # the headline cost level the tuning objective sees
STAGES = (50, 150)  # cumulative boosting rounds; the pruner decides after the first
DEFAULT_PARAMS = {
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
}
_LEARNER = {"deterministic": True, "force_row_wise": True, "num_threads": 1, "verbose": -1}

Fit = Callable[[pd.DataFrame, pd.DataFrame, Fold], pd.DataFrame]


def lgbm_defaults(
    basis: pd.DataFrame | None = None,
    *,
    horizon: int = HORIZON,
    n_splits: int = N_SPLITS,
    embargo: int = EMBARGO,
    seed: int = SEED,
    min_train_rows: int = MIN_TRAIN_ROWS,
):
    """Variant 6a: sensible defaults, no search. Returns a method with `trials` = 1."""

    def method(closes: pd.DataFrame) -> pd.DataFrame:
        learner = LGBMRegressor(
            n_estimators=STAGES[-1], random_state=seed, **_LEARNER, **DEFAULT_PARAMS
        )

        def fit(train: pd.DataFrame, test: pd.DataFrame, fold: Fold) -> pd.DataFrame:
            learner.fit(train[FEATURES], train["label"])
            return _wide(learner.predict(test[FEATURES]), test.index, closes)

        return _signals(
            closes,
            basis,
            fit,
            horizon=horizon,
            n_splits=n_splits,
            embargo=embargo,
            min_train_rows=min_train_rows,
        )

    method.trials = 1
    return method


def lgbm_tuned(
    basis: pd.DataFrame | None = None,
    *,
    horizon: int = HORIZON,
    n_splits: int = N_SPLITS,
    embargo: int = EMBARGO,
    n_trials: int = 20,
    seed: int = SEED,
    min_train_rows: int = MIN_TRAIN_ROWS,
):
    """Variant 6b: pruned TPE per fold; `trials` = n_splits x n_trials.

    Conservative by construction: every fold runs a study of `n_trials`
    configurations, counted whether or not a fold was later skipped for lack of
    data, so the DSR deflation never flatters the variant.
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def method(closes: pd.DataFrame) -> pd.DataFrame:
        def fit(train: pd.DataFrame, test: pd.DataFrame, fold: Fold) -> pd.DataFrame:
            inner = _inner_split(fold, horizon=horizon, embargo=embargo)
            dates = train.index.get_level_values("date")
            inner_train, inner_valid = train[dates.isin(inner.train)], train[dates.isin(inner.test)]
            if len(inner_train) < min_train_rows or len(inner_valid) < min_train_rows:
                return pd.DataFrame(index=fold.test, columns=closes.columns, dtype=float)
            params = _tune(closes, inner_train, inner_valid, n_trials=n_trials, seed=seed)
            return _fit_full(closes, train, test, params, seed)

        return _signals(
            closes,
            basis,
            fit,
            horizon=horizon,
            n_splits=n_splits,
            embargo=embargo,
            min_train_rows=min_train_rows,
        )

    method.trials = n_splits * n_trials
    return method


def _inner_split(fold: Fold, *, horizon: int, embargo: int) -> Fold:
    """The tuning split: the last quarter of the fold's training dates is the inner
    validation block, purged and embargoed from the rest.

    Built from `fold.train` alone, so a fold's test block can never reach its own
    search — the property `tests/test_ml.py` pins.
    """
    return purged_walk_forward(fold.train, n_splits=3, horizon=horizon, embargo=embargo)[-1]


def _signals(
    closes: pd.DataFrame,
    basis: pd.DataFrame | None,
    fit: Fit,
    *,
    horizon: int,
    n_splits: int,
    embargo: int,
    min_train_rows: int,
) -> pd.DataFrame:
    """Walk-forward out-of-fold signal panel (0 wherever no fold predicts).

    Folds that cannot be fitted are skipped, but a sample set too small to fit
    any fold is a configuration error, not a flat strategy: it means the basis
    panel is missing or misaligned, or the history is shorter than the longest
    feature window — raise rather than return an all-zero panel whose metrics
    are silently NaN.
    """
    samples = feature_panel(closes, basis).join(forward_label(closes, horizon)).dropna()
    if len(samples) < min_train_rows:
        raise ValueError(
            f"only {len(samples)} complete feature/label samples for {closes.shape[1]} symbols "
            f"over {len(closes)} days: check the basis panel's alignment and the history length"
        )
    folds = purged_walk_forward(closes.index, n_splits=n_splits, horizon=horizon, embargo=embargo)
    signals = pd.DataFrame(0.0, index=closes.index, columns=closes.columns)
    for fold in folds:
        dates = samples.index.get_level_values("date")
        train, test = samples[dates.isin(fold.train)], samples[dates.isin(fold.test)]
        if len(train) < min_train_rows or test.empty:
            continue
        ranked = centered_rank(fit(train, test, fold)).reindex(
            index=fold.test, columns=closes.columns
        )
        signals.loc[ranked.index, ranked.columns] = ranked.fillna(0.0)
    return signals


def _wide(values, index: pd.MultiIndex, closes: pd.DataFrame) -> pd.DataFrame:
    """(date, symbol)-indexed predictions -> date x symbol panel on the close grid."""
    return pd.Series(values, index=index).unstack("symbol").reindex(columns=closes.columns)


def _tune(
    closes: pd.DataFrame,
    inner_train: pd.DataFrame,
    inner_valid: pd.DataFrame,
    *,
    n_trials: int,
    seed: int,
) -> dict:
    """Pruned TPE over LightGBM hyperparameters, scored on the inner validation block."""

    def objective(trial: optuna.Trial) -> float:
        params = _suggest(trial)
        features = inner_train[FEATURES].to_numpy()
        dataset = lgb.Dataset(
            features, inner_train["label"].to_numpy(), params=params, free_raw_data=False
        )
        booster, trained, score = None, 0, float("-inf")
        for stage in STAGES:
            booster = lgb.train(
                _params(params, seed),
                dataset,
                num_boost_round=stage - trained,
                init_model=booster,
                keep_training_booster=True,
            )
            trained = stage
            pred = _wide(booster.predict(inner_valid[FEATURES]), inner_valid.index, closes)
            score = _score(pred, closes, inner_valid)
            trial.report(score, stage)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return score

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=5),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0),
    )
    study.optimize(objective, n_trials=n_trials, catch=(ValueError,))
    return dict(DEFAULT_PARAMS, **study.best_params)


def _suggest(trial: optuna.Trial) -> dict:
    """The search space: the defaults, with the capacity/regularization knobs opened."""
    params = dict(DEFAULT_PARAMS)
    params.update(
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        num_leaves=trial.suggest_int("num_leaves", 7, 63, log=True),
        min_child_samples=trial.suggest_int("min_child_samples", 20, 200, log=True),
        feature_fraction=trial.suggest_float("feature_fraction", 0.4, 1.0),
        bagging_fraction=trial.suggest_float("bagging_fraction", 0.5, 1.0),
        lambda_l2=trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
    )
    return params


def _params(params: dict, seed: int) -> dict:
    return dict(objective="regression", seed=seed, feature_pre_filter=False, **_LEARNER, **params)


def _fit_full(
    closes: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame, params: dict, seed: int
) -> pd.DataFrame:
    """Refit the tuned parameters on the fold's whole training set, predict its test block."""
    booster = lgb.train(
        _params(params, seed),
        lgb.Dataset(train[FEATURES].to_numpy(), train["label"].to_numpy()),
        num_boost_round=STAGES[-1],
    )
    return _wide(booster.predict(test[FEATURES]), test.index, closes)


def _score(pred: pd.DataFrame, closes: pd.DataFrame, valid: pd.DataFrame) -> float:
    """Sharpe of the ranked, vol-targeted, 2 bps-costed signal on the validation block.

    Same pipeline as the harness (engine overlay + engine accounting), so the
    tuning objective is the metric the decision rule uses — restricted to the
    fold's inner validation window.
    """
    signal = centered_rank(pred)
    if signal.empty:
        return float("-inf")
    window = closes.loc[: signal.index.max()]
    panel = pd.DataFrame(0.0, index=window.index, columns=closes.columns)
    panel.loc[signal.index, signal.columns] = signal.fillna(0.0)
    path = account(vol_target_positions(panel, window, VOL_TARGET, CAP), window, TUNING_BPS)
    value = sharpe(path.mean(axis=1).reindex(signal.index).dropna())
    return value if np.isfinite(value) else float("-inf")
