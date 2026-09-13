# Stopping bad training runs early, with a bound on stopping good ones

Most of a hyperparameter sweep's compute goes to runs that were never going to win. Early stopping
saves that compute. Every pruner in common use (median, percentile, successive halving,
Hyperband) is a heuristic, and none of them says how often it kills a run that would have ended
well. If you care about that number, you have to measure it yourself.

We built a rule that states it up front. You pick α, and the rule guarantees that the expected
fraction of runs that were good *and* got killed stays at or below α. We tested it on 400 real
fine-tuning runs under a pre-registered protocol. It kept the bound and saved about half the
evaluation compute. It also exposed a flaw in how it decides, which a small change fixes on the
same data. That fix still needs a fresh test.

## The rule

A run is *good* if its final validation loss ends at or below a threshold τ. At every evaluation,
a predictor forecasts the final loss from the curve so far. In the version we recommend, the
forecast is simply the latest observed value.

To calibrate, take n runs that have already finished. For each good one, record the worst moment
of its trajectory: the largest gap between forecast and τ at any evaluation. Runs that are not good
contribute nothing. Set λ to the ⌈(1−α)(n+1)⌉-th smallest of these scores. A new run is killed at
the first evaluation where its forecast exceeds τ + λ.

This is conformal risk control applied to the loss "good and killed". Taking the worst moment
across all evaluations means one threshold covers every look at a run, so no correction for
multiple looks is needed. The guarantee requires the calibration runs and the new runs to be
exchangeable. Random sampling gives that; adaptive samplers such as TPE do not.

## What we measured

The experiment, CC-03c, was pre-registered before any run finished:

- **Runs:** 400 LoRA runs of Qwen2.5-0.5B on dolly-15k, 750 steps each, validation loss measured
  50 times per run.
- **Configs:** drawn at random from a frozen hyperparameter prior (`hyperparameter-prior.md`).
- **τ:** pinned from an earlier pool before the first run landed.
- **Splits:** two, one random and one past-to-future by night, each about half calibration and
  half test.

The pre-registered predictor was a power law fitted to each curve. Its result:

| | false kills | joint rate (95% CI) | eval compute saved |
|---|---|---|---|
| random split | 13 of 200 | 6.5% (3.5–10.9%) | 52.2% |
| past-to-future split | 11 of 203 | 5.4% (2.7–9.5%) | 52.8% |

Both intervals contain the 5% target. The gate we used could have failed: a rule that kills every
run immediately was scored the same way, and it failed on both splits. The earlier version of this
experiment could not make that claim. Its split drew too few good runs for any breach to be
possible.

## What went wrong

**Every kill fired at the 3rd of 50 evaluations.** The early ranking of runs doesn't match the
final ranking well. On the calibration runs, raw losses only correlate at 0.90 with the final
ranking from evaluation 18. So the rule was mostly acting on noise that happens to be cheap to
act on. It stayed within budget because the bound held.

- **A third of good runs still died.** The bound covers the joint rate, and good runs were only
  about a sixth of all runs, so a 5% joint budget allows about a third of them to be killed. Most
  of those were borderline; the single best run survived on both splits.
- **The fitted curve made things worse.** The power-law predictor ranked runs less well than the
  latest observed value at every early evaluation. We therefore do not ship it.

## The predictor we ship, on the same splits

The package's default predictor is the latest observed value. It was scored on the same two
splits as an ungated reference row when the experiment closed, and this is what the reproduction
script checks:

| | false kills | joint rate | eval compute saved |
|---|---|---|---|
| random split | 18 of 200 | 9.0% | 78.0% |
| past-to-future split | 7 of 203 | 3.4% | 75.4% |

The random split is a breach. Over 2000 random resplits of the same 400 runs, the same rule
averages a 5.0% joint rate and lands on a breach 8.5% of the time, so the 18 sits in that
distribution's tail rather than showing a leak. A correct rule at this sample size breaches on
about 7% of splits.

## A fix, not yet confirmed

Holding kills until evaluation 18 and forecasting from the latest value, on the same 400 runs:

- **Savings:** 57.5% on the random split and 55.9% on the past-to-future split, with the same
  false-kill counts as above.
- **Over 2000 resplits:** joint rate 5.0%, savings 60%, and the best run killed 13.6% of the time,
  against 24.8% without the hold.

These runs had already been seen, so that result is a hypothesis. Its test needs a fresh pool of
runs, and a follow-up experiment is sized for 200.

## What this does not claim

- **The approach is established.** Calibrating a stop threshold on a predictor with a conformal
  guarantee is an established pattern. Xie et al. (arXiv:2602.13935) use the same device, a
  threshold on the maximum of a running score calibrated on one class only, to stop LLM
  reasoning traces, with a bound on the probability that any single trace stops falsely. Related
  work does the same for mixed-integer solvers and LLM agent episodes. As far as our search found,
  nobody has applied it to pruning training runs, and no major tuning library ships a pruner with a
  stated error guarantee. `prior-art.md` has the sweep.
- **One setting.** One model, one dataset, one horizon. Other models could have more spread
  between runs, or less.
- **False alarms.** Even with a correct rule, a breach test on 200 runs flags about 7% of random
  splits. A single breach is weak evidence on its own.

## Try it

`surestop` v0.2 wraps the rule as a numpy class and an Optuna pruner. The pruner runs the
first trials unpruned to calibrate, then freezes. `reproduce/reproduce_cc03c.py` checks that the
package reproduces the last-value rows above exactly, on both pre-registered splits, from the
curves in `reproduce/cc03c_pool.json`.

```python
study = optuna.create_study(sampler=optuna.samplers.RandomSampler(),
                            pruner=ConformalPruner(n_calibration=60, alpha=0.05))
```
