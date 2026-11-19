# SureStop

**A guard for a fleet of training runs. It stops the bad ones early, with a stated bound on how
often it stops one that would have ended well.** You pick α and say what "ended well" means. The
rule then bounds either the fraction of all runs that were good and got killed, or the chance
that a good run gets killed.

Every pruner in common use (median, percentile, successive halving, Hyperband) is a heuristic,
and none says how often it kills a run that would have finished well. This one does, under a
condition you can check: the runs you calibrate on and the runs you judge must come from the same
random process.

It was measured on 400 real LoRA fine-tuning runs under a pre-registered protocol. The curves are
in this repository, and two scripts check that the package reproduces every number quoted here.

## When to use it, and when not

**Use it when every good run matters.** The bound protects each good run, so it earns its cost
where losing one is expensive:

- seed replicates for a paper, where each finished run is a data point;
- nightly retrains or per-customer fine-tunes, where a killed good run is a missed delivery;
- training with high seed variance, where a run that starts slowly can still end well.

These jobs also meet the rule's two requirements without effort. They repeat one process, so the
runs are exchangeable, and they come with an acceptance bar, which is the threshold the rule
needs. None of these settings has been measured yet. The measurement below is a hyperparameter
sweep.

**Don't use it to find the single best configuration in a sweep.** A sweep needs one winner and
can afford to lose most of its good runs. On the 400 measured runs, successive halving killed 79%
of the good runs, saved 88% of the compute, and still ended within 0.0005 nats of the best run.
SureStop killed far fewer good runs and saved less
([the comparison](#against-the-heuristic-pruners)). If that trade doesn't matter to you, use
ASHA.

## Install

```
pip install surestop            # numpy + scipy
pip install "surestop[optuna]"  # adds the Optuna pruner
```

## Use

Plain numpy, on any training loop:

```python
from surestop import KillRule

# completed_curves: array of shape (runs, evaluations), lower is better
rule = KillRule(alpha=0.05, good_threshold=2.16).fit(completed_curves)

rule.should_kill(curve_so_far)     # True -> stop this run now
rule.kill_step(full_curve)         # where it would have fired, or None
rule.evaluate(held_out_curves)     # false kills, joint rate, savings
```

With Optuna, the first `n_calibration` trials run unpruned to calibrate, then the rule freezes:

```python
import optuna
from surestop import ConformalPruner

study = optuna.create_study(sampler=optuna.samplers.RandomSampler(),
                            pruner=ConformalPruner(n_calibration=60, alpha=0.05))
```

Pass `maximize=True` (or a `direction="maximize"` study) when higher is better.

Both classes take `guarantee="conditional"` to bound the chance that a good run is killed. The
default, `guarantee="joint"`, bounds the fraction of all runs that are good and killed.
[Two guarantees](#two-guarantees) explains the difference. The conditional guarantee is new in
0.3.0. Until that release is on PyPI, install it with
`pip install git+https://github.com/ACNoonan/surestop`.

The same pruner is in the OptunaHub registry as
[`pruners/conformal`](https://github.com/optuna/optunahub-registry/tree/main/package/pruners/conformal),
so an Optuna user can load it without installing surestop:

```python
import optunahub

module = optunahub.load_module(package="pruners/conformal")
pruner = module.ConformalPruner(n_calibration=60, alpha=0.05)
```

The registry entry is the 0.2.0 pruner, which has the joint guarantee only.

## The rule

A run is *good* if its final value ends at or below a threshold τ. At every evaluation, a
predictor forecasts the final value from the curve so far. The default predictor is the latest
observed value.

To calibrate, take n runs that have already finished. For each good one, record its worst moment:
the largest gap between forecast and τ at any evaluation. Runs that are not good contribute
nothing. Set λ to the ⌈(1−α)(n+1)⌉-th smallest of these scores. A new run is killed at the first
evaluation where its forecast exceeds τ + λ.

This is conformal risk control on the loss "good and killed". Taking the worst moment across all
evaluations means one threshold covers every look at a run, so there is no correction for
multiple looks. The guarantee needs the calibration runs and the new runs to be exchangeable.
Random sampling gives that. Adaptive samplers such as TPE do not.

### Two guarantees

**Joint (default).** The rule above. Of all the runs you judge, the expected fraction that are
good and killed is at most α. When good runs are rare this allows a large share of them to die.
In the measurement below good runs were a sixth of all runs, so a 5% joint budget let about a
third of the good runs be killed.

**Conditional (`guarantee="conditional"`).** A good run is killed with probability at most α.
The rule ranks the scores of the m good calibration runs alone and sets λ to the
⌈(1−α)(m+1)⌉-th smallest. A new good run is exchangeable with those m, so its score is equally
likely to fall in any of the m+1 positions among them, and it lands above the k-th smallest with
probability 1 − k/(m+1), which that choice of k keeps at or below α.

The conditional rule costs more in two ways. It needs at least ⌈1/α⌉ − 1 good calibration runs,
which is 19 at α = 0.05 and about 120 calibration runs when a sixth of runs are good. It also
saves less compute, because a threshold that spares 95% of good runs sits higher than one that
spares two thirds of them.

## What was measured

Experiment CC-03c, pre-registered before any run finished:

- **Runs:** 400 LoRA fine-tunes of Qwen2.5-0.5B on dolly-15k, 750 steps each, validation loss
  measured 50 times per run.
- **Configs:** i.i.d. draws from a frozen hyperparameter prior (`docs/hyperparameter-prior.md`).
- **τ:** pinned from an earlier pool before the first run landed.
- **Splits:** two, one random and one past-to-future by night, each about half calibration and
  half test.

The pre-registered rule used a fitted power-law predictor. On both splits it kept the bound and
saved about half the evaluation compute:

| split | false kills | joint rate (95% CI) | eval compute saved |
|---|---|---|---|
| random | 13 of 200 | 6.5% (3.5–10.9%) | 52.2% |
| past-to-future | 11 of 203 | 5.4% (2.7–9.5%) | 52.8% |

A rule that kills every run immediately was scored by the same gate and failed on both splits.

**The flaw that run found:** every kill fired at the 3rd of 50 evaluations. Early rankings only
match final rankings from about evaluation 18. The rule stayed within budget because the bound
held. The fitted power law ranked runs worse than the latest observed value at every early
evaluation.

**What this package ships instead,** and what the reproduction script checks:

| predictor | hold until | random split | past-to-future split | 2000 resplits |
|---|---|---|---|---|
| last value | none | 18 of 200 (9.0%), 78% saved | 7 of 203 (3.4%), 75% saved | joint 5.0%, breach 8.5% |
| last value | eval 18 | 18 of 200 (9.0%), 57% saved | 7 of 203 (3.4%), 56% saved | joint 5.0%, breach 8.6% |

The random split's 18 of 200 is a breach. It sits at the 95th percentile of that rule's own
resplit distribution, whose mean is 5.0%, so it reads as a split tail rather than a leak. A
correct rule at this sample size lands on a breach in about 7% of splits.

The held version was chosen *after* seeing these 400 runs, so its numbers are a hypothesis, and
the evidence on whether it protects the best runs is mixed. Over the 2000 resplits it killed the
best test run 13.6% of the time, against 24.8% without the hold. In the whole-sweep comparison
below it killed the pool's single best run 58.6% of the time, against 1.0% without the hold. That
run ranks 69th of 400 at evaluation 18 and reaches 1st only at the end, and a held rule sets a
tighter threshold for the later evaluations. The hold is off by default.

Full account: `docs/writeup.md`. Run the check yourself:

```
python reproduce/reproduce_cc03c.py
```

## Against the heuristic pruners

This comparison is exploratory. It was run after the experiment closed, on the same 400 curves,
and was not pre-registered.

All 400 runs arrive in a random order and each pruner decides from the first run. SureStop pays
for its calibration runs in full. Median and ASHA are Optuna's `MedianPruner` and
`SuccessiveHalvingPruner`, and a test replays runs through Optuna itself to check that the
stopping points agree exactly. The table averages 500 orders, at α = 0.05.

| pruner | compute saved | good runs killed | top-5 runs killed | regret (nats) |
|---|---|---|---|---|
| SureStop joint, 60 calibration | 62.9% | 26.6% | 16.9% | 0.0000 |
| SureStop joint, 60 calibration, hold 18 | 46.7% | 25.9% | 13.5% | 0.0001 |
| SureStop conditional, 200 calibration | 28.8% | 1.7% | 0.0% | 0.0000 |
| Median, 5 startup, warmup 18 | 58.0% | 62.6% | 41.4% | 0.0002 |
| Median, 5 startup, no warmup | 91.8% | 83.6% | 86.3% | 0.0006 |
| ASHA, η = 3, rungs 2/6/18 | 87.8% | 79.4% | 83.0% | 0.0005 |

"Good runs killed" is a share of all 65 good runs in the pool. "Regret" is the best surviving
final loss minus the pool's best.

- **SureStop keeps the good runs and the heuristics don't.** It kills about a quarter of them
  under the joint guarantee and under 2% under the conditional one. The heuristics kill 63% to
  84%.
- **In this pool that protection changes the sweep's result by almost nothing.** Final losses
  span 2.1575 to 2.2170, with a standard deviation of 0.013 nats, and the worst regret in the
  table is 0.0006 nats. The good runs are close to interchangeable, so losing most of them costs
  a sweep nothing it can measure.
- **The conditional rule's saving is all in the second half.** Its first 200 runs calibrate and
  run in full. On the 200 it judges, it skips about 58% of the evaluations and kills about 3% of
  the good runs. A fleet that calibrates once and then runs for months keeps that rate.

Run it yourself:

```
python reproduce/baselines_cc03c.py --check
```

## What the bound does not cover

- **It needs exchangeable runs.** With a random or quasi-random sampler the bound holds. With
  TPE, Gaussian-process or CMA-ES samplers it does not. The Optuna pruner warns once and carries
  on, but do not rely on the bound there.
- **The default bounds the joint rate, and the share of good runs killed can be much larger.**
  When good runs are rare, a large share of them can still die: about a third in the measurement
  above, mostly borderline ones. Use `guarantee="conditional"` if that matters.
- **It does not make a sweep find a better winner.** See
  [the comparison](#against-the-heuristic-pruners).
- **An estimated τ gives a measured bound rather than a proven one.** `good_quantile` estimates τ from the
  calibration runs. Simulations put the realized rate at 0.88 to 0.97 times α. Pass a real
  `good_threshold` when you have one.
- **α = 0.05 needs at least 19 calibration runs** under the joint guarantee and 19 *good*
  calibration runs under the conditional one. The certified threshold gets loose with few good
  runs. Sweeps of a few dozen trials are too small for this.
- **One setting only so far.** One model, one dataset, one horizon.

## Layout

| path | what |
|---|---|
| `src/surestop/` | `KillRule` (numpy/scipy) and `ConformalPruner` (Optuna) |
| `tests/` | both bounds hold on synthetic exchangeable runs, the same assertions fail on a rule that leaks 2× its budget, and the baseline pruners match Optuna's |
| `reproduce/` | the 400 curves, the pre-registered splits and targets, and two scripts with their committed output: the reproduction and the comparison against median and ASHA |
| `docs/` | the write-up, the prior-art sweep, and the frozen hyperparameter prior the curves were drawn from |
| `optunahub/` | the source of the OptunaHub registry entry `pruners/conformal` (0.2.0, joint guarantee only) |

## Related work

Calibrating a stop threshold on a predictor with a conformal guarantee is an established pattern. Xie et al.
(arXiv:2602.13935) use the same device, a threshold on the maximum of a running score calibrated
on one class only, to stop LLM reasoning traces. Related work does the same for mixed-integer
solvers and LLM agent episodes. As far as a search found, nobody has applied it to pruning
training runs, and no major tuning library ships a pruner with a stated error guarantee.
`docs/prior-art.md` has the sweep.

## Citing

Concept DOI: [10.5281/zenodo.22726440](https://doi.org/10.5281/zenodo.22726440), which always resolves
to the newest version. `CITATION.cff` has the full entry. The curves are a measurement, released
under the same MIT licence as the code.
