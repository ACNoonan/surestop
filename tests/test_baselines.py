"""The numpy median and ASHA rules in reproduce/baselines_cc03c.py stop where Optuna's pruners stop."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

optuna = pytest.importorskip("optuna")
optuna.logging.set_verbosity(optuna.logging.WARNING)

REPRO = Path(__file__).resolve().parent.parent / "reproduce"
sys.path.insert(0, str(REPRO))
from baselines_cc03c import asha_stops, median_stops  # noqa: E402

N = 150  # runs replayed per order; enough for every rung and the median to be exercised


def curves():
    pool = json.loads((REPRO / "cc03c_pool.json").read_text())
    return np.array([r["val_ce"] for r in pool["runs"]], dtype=float)


def optuna_stops(Y, order, pruner, first_step):
    """Replay the runs in `order` through a real Optuna pruner; return each run's last evaluation."""
    T = Y.shape[1]
    stop = {}

    def objective(trial):
        i = int(order[trial.number])
        for t in range(T):
            trial.report(float(Y[i, t]), t + first_step)
            # No check at the final evaluation: a kill there saves nothing, and neither the
            # baselines nor SureStop score one.
            if t < T - 1 and trial.should_prune():
                stop[i] = t
                raise optuna.TrialPruned()
        stop[i] = T - 1
        return float(Y[i, -1])

    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0), pruner=pruner)
    study.optimize(objective, n_trials=len(order))
    return np.array([stop[int(i)] for i in order])


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("warmup", [0, 17])
def test_median_matches_optuna(seed, warmup):
    Y = curves()
    order = np.random.default_rng(seed).permutation(len(Y))[:N]
    want = optuna_stops(Y, order, optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=warmup), 0)
    got = median_stops(Y, order, startup=5, warmup=warmup)[order]
    assert (got < Y.shape[1] - 1).sum() > 20
    np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize("seed", [0, 1])
def test_asha_matches_optuna(seed):
    Y = curves()
    order = np.random.default_rng(seed).permutation(len(Y))[:N]
    pruner = optuna.pruners.SuccessiveHalvingPruner(min_resource=2, reduction_factor=3,
                                                    min_early_stopping_rate=0)
    want = optuna_stops(Y, order, pruner, 1)  # steps 1..50, so rungs fall at evaluations 2, 6, 18
    got = asha_stops(Y, order)[order]
    assert len(set(got.tolist())) == 4  # kills at all three rungs, plus survivors
    np.testing.assert_array_equal(got, want)
