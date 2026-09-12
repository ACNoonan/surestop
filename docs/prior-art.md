# Prior art — is a certified early-kill pruner for training runs occupied?

*Sweep run 2026-09-11. Question: does anything already stop or prune ML training runs early with
a finite-sample guarantee on killing a run that would have finished good? This is CC-03c's
construction: a final-loss predictor at every checkpoint, a conformal-risk-control threshold
calibrated on completed runs, and a bound on the expected false-kill rate across all checkpoints.*

## Verdict

**The mechanism is not new. The application appears open, at moderate confidence.**

- **Mechanism.** Predict the final outcome from a partial trajectory, then calibrate a stop
  threshold with a distribution-free guarantee. Three 2026 papers use this pattern elsewhere:
  - LLM reasoning traces (2602.13935), with the same max-over-checkpoints statistic
  - mixed-integer solver runs (2602.01476)
  - LLM agent episodes (Doomed from the Start)
- **Application.** Nothing found applies it to training runs or HPO trials. The two real
  guarantees found for discarding a good configuration early (CVST 2015, SQRS 2022) use classical
  sequential tests over cross-validation resampling, not training checkpoints.
- **Libraries.** No major HPO library ships a pruner with a stated error-rate guarantee. The
  nearest is Optuna's `WilcoxonPruner`, whose docs advise a Pocock correction; the correction is
  advice, not a default.

**What this licenses.** A tool or a measurement write-up: "certified pruning of training runs,
measured on 400 real LoRA runs". **Not** a methods-novelty claim.

2602.13935 was read in full after this sweep. Its statistic is the closest match to ours, and the write-up credits it.

## Instrument

- **scout.py:** two runs, seven phrasings, 1,813 works.
  - The first run's recall control failed (2208.02814 did not surface), so its results were
    discarded as evidence.
  - The second run surfaced both controls, 2208.02814 and 2409.15844.
  - 75 titles matched an early-stop/prune/learning-curve filter; five abstracts were pulled.
- **Breadth agent (Sonnet):** 30 web queries across the conformal, AutoML, sequential-testing,
  bandit and LLM-practice vocabularies, plus the docs for Optuna, Ray Tune, W&B, SMAC3, DEHB,
  Syne Tune, Ax/BoTorch and Hyperopt.
- **Not reached:** non-English work, theses, a paywalled Springer version of SQRS (its arXiv
  preprint was read instead), and AutoML workshop-only papers.

## Sources and depth

Depth vocabulary: `full · fetch-summary · snippet · unsearched`. `full` is claimed only where a full-text
extraction and a written read exist; those live in the private research repository, not here.

| source | what it is | verdict vs CC-03c | depth |
|---|---|---|---|
| 2409.15844 Adaptive Learn-then-Test | FWER/FDR on the final selected HP set; its adaptive rounds re-evaluate already-trained candidates | adjacent | full |
| 2606.25601 Farzaneh & Simeone monograph | certifies the final selection; puts successive halving, Hyperband and freeze-thaw explicitly out of scope; never cites CRC | different problem | full on ch. 1–3, 6, 8 · snippet on ch. 4, 5, 7 · appendices B/C unsearched |
| 2404.16795 ifBO / FT-PFN | pauses and resumes, never discards; no guarantee; its predictor could be a sharper backbone (inference) | adjacent | full |
| 2602.13935 Statistical Early Stopping for Reasoning Models | "Maxwise Conformal Stopping": a conformal threshold on the max over checkpoints, bounding a false early stop; it halts reasoning tokens, not training | adjacent, closest statistic | full (read in full after this sweep, before any public claim) |
| 2602.01476 Conformal Prediction for Early Stopping in MIP | a learned optimality-gap predictor plus a conformal stop threshold; solver runs | adjacent | snippet |
| Doomed from the Start (2026) | a recall-controlled abort cascade on hidden-state probes; LLM agent episodes | adjacent | snippet |
| Krueger, Panknin & Braun 2015, CVST (JMLR 16) | bound on dropping a winning config; sequential tests over CV data-subset size | adjacent | fetch-summary (agent-read, no extraction saved) |
| 2112.12438 SQRS | parametric SLRT racing over resampling iterations; classical ML | adjacent | fetch-summary (agent-read, no extraction saved) |
| 2207.03017 ACHO | conformal intervals as an acquisition function; no early termination | different problem | fetch-summary (agent-read) |
| 2305.03623 Conformal quantile regression for HPO | conformal surrogate for multi-fidelity BO; kill rule not confirmed | likely different problem | fetch-summary |
| 2502.04206, 2208.02922, 2407.17358, Makarova et al. 2022 | LTT survey; ACE; Quantile LTT; whole-campaign termination | different or unconfirmed | snippet |
| Optuna `WilcoxonPruner` | per-trial Wilcoxon test; docs advise a Pocock correction for the nominal false-positive rate | nearest library feature | fetch-summary (docs) |
