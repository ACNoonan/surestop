# certified-pruner

**Stop bad training runs early, with a stated bound on how often you stop one that would have
ended well.** You pick α. The rule kills runs so that the expected fraction of runs that were
good *and* got killed stays at or below α.

Every pruner in common use (median, percentile, successive halving, Hyperband) is a heuristic,
and none says how often it kills a run that would have finished well. This one does, under a
condition you can check: the runs you calibrate on and the runs you judge must come from the same
random process.

It was measured on 400 real LoRA fine-tuning runs under a pre-registered protocol. The curves are
in this repository, and a script checks that the package reproduces the recorded numbers exactly.

## Install

```
pip install certified-pruner            # numpy + scipy
pip install "certified-pruner[optuna]"  # adds the Optuna pruner
```

## Use

Plain numpy, on any training loop:

```python
from certified_pruner import KillRule

# completed_curves: array of shape (runs, evaluations), lower is better
rule = KillRule(alpha=0.05, good_threshold=2.16, min_peek=17).fit(completed_curves)

rule.should_kill(curve_so_far)     # True -> stop this run now
rule.kill_step(full_curve)         # where it would have fired, or None
rule.evaluate(held_out_curves)     # false kills, joint rate, savings
```

With Optuna, the first `n_calibration` trials run unpruned to calibrate, then the rule freezes:

```python
import optuna
from certified_pruner import CertifiedPruner

study = optuna.create_study(sampler=optuna.samplers.RandomSampler(),
                            pruner=CertifiedPruner(n_calibration=60, alpha=0.05))
```

Pass `maximize=True` (or a `direction="maximize"` study) when higher is better.

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
correct rule at this sample size lands on a breach in about 7% of splits. The held version was
chosen *after* seeing these 400 runs, so its numbers are a hypothesis. It halves
how often the single best run is killed at the same budget. Its confirmation needs a fresh pool.

Full account: `docs/writeup.md`. Run the check yourself:

```
python reproduce/reproduce_cc03c.py
```

## What the bound does not cover

- **It needs exchangeable runs.** With a random or quasi-random sampler the bound holds. With
  TPE, Gaussian-process or CMA-ES samplers it does not. The Optuna pruner warns once and carries
  on, but do not rely on the bound there.
- **It bounds the joint rate rather than the share of good runs killed.** When good runs are rare, a large
  share of them can still die: about a third in the measurement above, mostly borderline ones.
  Lower α if that matters.
- **An estimated τ gives a measured bound rather than a proven one.** `good_quantile` estimates τ from the
  calibration runs. Simulations put the realized rate at 0.88 to 0.97 times α. Pass a real
  `good_threshold` when you have one.
- **α = 0.05 needs at least 19 calibration runs**, and the certified threshold gets loose with
  few good runs. Sweeps of a few dozen trials are too small for this.
- **One setting only so far.** One model, one dataset, one horizon.

## Layout

| path | what |
|---|---|
| `src/certified_pruner/` | `KillRule` (numpy/scipy) and `CertifiedPruner` (Optuna) |
| `tests/` | the bound holds on synthetic exchangeable runs, and the same assertion fails on a rule that leaks 2× its budget |
| `reproduce/` | the 400 curves, the pre-registered splits and targets, the script, and its committed output |
| `docs/` | the write-up, the prior-art sweep, and the frozen hyperparameter prior the curves were drawn from |
| `optunahub/` | the package as an OptunaHub registry entry |

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
