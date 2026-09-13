"""The bound, a negative control that proves the bound check can fail, and the rule's edges."""

import math

import numpy as np
import pytest

from surestop import KillRule, NotEnoughCalibration

T = 30


def synth(n, rng):
    """Exchangeable learning curves: a final value, plus a decaying offset and noise that shrink with step."""
    finals = rng.normal(2.17, 0.013, n)
    t = np.arange(T)
    offset = rng.uniform(0.05, 0.3, n)[:, None] * np.exp(-t / 6.0)
    noise = rng.normal(0, 1, (n, T)) * 0.012 * np.exp(-t / 10.0)
    Y = finals[:, None] + offset + noise
    Y[:, -1] = finals
    return Y


def mean_joint_rate(alpha_fit, reps=400, n_cal=200, n_test=200, seed=0, **kw):
    rng = np.random.default_rng(seed)
    rates = []
    for _ in range(reps):
        cal, test = synth(n_cal, rng), synth(n_test, rng)
        rule = KillRule(alpha=alpha_fit, good_threshold=2.159, **kw).fit(cal)
        rates.append(rule.evaluate(test)["joint_false_kill_rate"])
    rates = np.array(rates)
    return rates.mean(), rates.std(ddof=1) / math.sqrt(reps)


def test_joint_false_kill_rate_is_bounded():
    mean, se = mean_joint_rate(0.05)
    assert mean <= 0.05 + 3 * se, (mean, se)


def test_negative_control_the_bound_check_can_fail():
    # A rule that leaks 2x its budget: certify at 0.10, hold the result to 0.05. The assertion
    # the real test makes must fail here. (At 0.25 the synthetic pool has too few good runs for a
    # finite threshold, so the rule kills everything — too easy a control to prove anything.)
    mean, se = mean_joint_rate(0.10, reps=400)
    assert not (mean <= 0.05 + 3 * se), (mean, se)


def test_the_rule_actually_saves_compute():
    rng = np.random.default_rng(1)
    rule = KillRule(alpha=0.05, good_threshold=2.159).fit(synth(300, rng))
    assert rule.evaluate(synth(300, rng))["savings"] > 0.2


def test_min_peek_blocks_earlier_kills():
    rng = np.random.default_rng(2)
    rule = KillRule(alpha=0.05, good_threshold=2.159, min_peek=12).fit(synth(300, rng))
    steps = [rule.kill_step(c) for c in synth(300, rng)]
    assert all(s is None or s >= 12 for s in steps)
    assert any(s is not None for s in steps)


def test_should_kill_agrees_with_kill_step():
    rng = np.random.default_rng(3)
    rule = KillRule(alpha=0.05, good_threshold=2.159, min_peek=4).fit(synth(300, rng))
    for c in synth(50, rng):
        first = next((t for t in range(T) if rule.should_kill(c[: t + 1])), None)
        assert first == rule.kill_step(c)


def test_hold_until_resolved_moves_the_first_peek_later():
    rng = np.random.default_rng(4)
    cal = synth(300, rng)
    held = KillRule(alpha=0.05, good_threshold=2.159, hold_until_resolved=0.9).fit(cal)
    assert held.first_peek_ > 0
    assert held.resolution_[held.first_peek_] >= 0.9


def test_too_few_calibration_runs_raises():
    with pytest.raises(NotEnoughCalibration):
        KillRule(alpha=0.05).fit(synth(18, np.random.default_rng(5)))


def test_maximize_mirrors_minimize():
    rng = np.random.default_rng(6)
    cal, test = synth(200, rng), synth(100, rng)
    lo = KillRule(alpha=0.05, good_threshold=2.159).fit(cal)
    hi = KillRule(alpha=0.05, good_threshold=-2.159, maximize=True).fit(-cal)
    assert [lo.kill_step(c) for c in test] == [hi.kill_step(-c) for c in test]
