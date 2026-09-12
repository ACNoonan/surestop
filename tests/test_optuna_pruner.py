"""The Optuna pruner: shadow mode first, pruning after, a warning for adaptive samplers."""

import numpy as np
import pytest

optuna = pytest.importorskip("optuna")
from certified_pruner import CertifiedPruner  # noqa: E402

optuna.logging.set_verbosity(optuna.logging.WARNING)
T = 20


def objective(trial):
    x = trial.suggest_float("x", 0.0, 1.0)
    rng = np.random.default_rng(trial.number)
    final = 2.15 + 0.05 * x + rng.normal(0, 0.005)
    for step in range(T):
        value = final + 0.2 * np.exp(-step / 4) + rng.normal(0, 0.004) * np.exp(-step / 8)
        if step == T - 1:
            value = final
        trial.report(float(value), step)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return final


def test_calibrates_in_shadow_mode_then_prunes():
    pruner = CertifiedPruner(n_calibration=40, alpha=0.05)
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0), pruner=pruner)
    study.optimize(objective, n_trials=160)
    states = [t.state for t in study.trials]
    assert all(s == optuna.trial.TrialState.COMPLETE for s in states[:40])
    assert pruner.rule_ is not None and np.isfinite(pruner.rule_.lambda_)
    assert sum(s == optuna.trial.TrialState.PRUNED for s in states) > 0


def test_warns_for_an_adaptive_sampler():
    pruner = CertifiedPruner(n_calibration=20)
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=0), pruner=pruner)
    with pytest.warns(RuntimeWarning, match="not exchangeable"):
        study.optimize(objective, n_trials=2)


def test_from_curves_is_calibrated_before_the_first_trial():
    rng = np.random.default_rng(0)
    curves = 2.17 + rng.normal(0, 0.01, (60, 1)) + 0.2 * np.exp(-np.arange(T) / 4)
    pruner = CertifiedPruner.from_curves(curves, steps=range(T))
    assert pruner.rule_.n_calibration_ == 60 and pruner.steps_ == list(range(T))
