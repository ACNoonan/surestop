# Structure — certified-pruner

**Kind: artifact (public package and measurement).** The research work that produced it lives in a
private repository. This repository holds what a reader can install, run, and check.

## The one rule

> **Everything here must run for a stranger, from a clean checkout, and show the recorded numbers
> are the package's numbers.**

`reproduce/reproduce_cc03c.py` is that check. Its committed output,
`reproduce_cc03c_RESULTS.txt`, is what a reader diffs their run against. If the two disagree, the
package has drifted from what was measured.

## Map

```
README.md                 what this is, how to use it, what was measured, what the bound omits
STRUCTURE.md              this file
LICENSE                   MIT
pyproject.toml            builds certified-pruner from src/
src/certified_pruner/
  core.py                 KillRule: fit on completed curves, decide on partial ones
  optuna_pruner.py        CertifiedPruner: shadow mode for n_calibration trials, then frozen
tests/
  test_core.py            the bound holds on exchangeable synthetic runs; a 2× leak fails the same check
  test_optuna_pruner.py   shadow mode, pruning after, the adaptive-sampler warning
reproduce/
  cc03c_pool.json         400 curves × 50 evaluations, with each run's sampled config
  cc03c_targets.json      alpha, tau, both pre-registered splits, and every number recorded at close
  reproduce_cc03c.py      re-runs the shipped rule on the splits and compares
  reproduce_cc03c_RESULTS.txt   committed output
docs/
  writeup.md              the measurement, what went wrong, and the untested fix
  prior-art.md            the sweep that found the mechanism taken and the application open
  hyperparameter-prior.md the frozen prior every curve was drawn from
optunahub/package/pruners/certified_pruner/
                          the registry entry, self-contained, ready to copy into an optunahub-registry fork
.github/workflows/test.yml   tests plus the reproduction, on three Python versions
```
