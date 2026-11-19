# Structure — surestop

**Kind: artifact (public package and measurement).** The research work that produced it lives in a
private repository. This repository holds what a reader can install, run, and check.

## The one rule

> **Everything here must run for a stranger, from a clean checkout, and show the recorded numbers
> are the package's numbers.**

`reproduce/reproduce_cc03c.py` is that check. Its committed output,
`reproduce_cc03c_RESULTS.txt`, is what a reader diffs their run against. If the two disagree, the
package has drifted from what was measured. `reproduce/baselines_cc03c.py` is the same check for
the comparison against median pruning and ASHA, which is exploratory.

## Map

```
README.md                 what this is, how to use it, what was measured, what the bound omits
STRUCTURE.md              this file
LICENSE                   MIT
pyproject.toml            builds surestop from src/
src/surestop/
  core.py                 KillRule: fit on completed curves, decide on partial ones; joint or
                          conditional guarantee
  optuna_pruner.py        ConformalPruner: shadow mode for n_calibration trials, then frozen
tests/
  test_core.py            both bounds hold on exchangeable synthetic runs; a 2× leak fails the same checks
  test_optuna_pruner.py   shadow mode, pruning after, the adaptive-sampler warning
  test_baselines.py       the numpy median and ASHA rules stop where Optuna's pruners stop
reproduce/
  cc03c_pool.json         400 curves × 50 evaluations, with each run's sampled config
  cc03c_targets.json      alpha, tau, both pre-registered splits, and every number recorded at close
  reproduce_cc03c.py      re-runs the shipped rule on the splits and compares
  reproduce_cc03c_RESULTS.txt   committed output
  baselines_cc03c.py      whole-sweep comparison: SureStop against median pruning and ASHA
  baselines_cc03c_RESULTS.txt   committed output
docs/
  writeup.md              the measurement, what went wrong, the comparison with the heuristics, and both guarantees
  prior-art.md            the sweep that found the mechanism taken and the application open
  hyperparameter-prior.md the frozen prior every curve was drawn from
optunahub/package/pruners/conformal/
                          the registry entry, self-contained. Merged into optuna/optunahub-registry as
                          pruners/conformal (PR #417, 2026-10-07); keep this copy identical to it.
                          It is the 0.2.0 pruner: joint guarantee only, until a registry PR updates it
.github/workflows/test.yml   tests plus the reproduction and the comparison, on three Python versions
```
