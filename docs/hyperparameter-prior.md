# The frozen hyperparameter prior every curve in `reproduce/cc03c_pool.json` was drawn from

*Copied from the private research repository. Version 2, frozen 2026-07-25. Internal execution notes removed; the sampled and fixed quantities are verbatim.*


*The load-bearing exchangeability object: runs are exchangeable within a
prior version because every config is an i.i.d. draw from THIS document.
Any edit after this freeze = v3 = a new calibration pool. v2 frozen on
Adam's explicit confirmation, 2026-07-25 ("good to freeze v2").*

**v1 → v2 (smoke-bar re-scope, per CC-03 §4 NULL (re-scope)):** the v1
smoke measured ~5.5 runs/10h-night against the ≥8 bar (draw 1 batch-16
complete at 77.8 min; draw 2 batch-64 rate-projected 183 min; linear-in-
accumulation model for the rest). Three changes, everything else verbatim:
1. **Horizon 750 optimizer steps** (was 1,500), eval every 15 steps —
   still exactly 50 evals, curve dimensionally identical.
2. **Val-CE measured on a frozen 500-example subset** of the frozen val
   split: `val_sub = val[default_rng(2027).permutation(1500)[:500]]`,
   sorted-index sha256/16 `e0ca2599b434cb1a` (eval cost dominated the
   short-horizon run budget).
3. Smoke mode = 15 steps + 1 eval.
Projected ~12 runs/night; the v2 smoke re-measures before any pool
accrues. Same night-seed protocol, so v2 smoke draws 1–3 are the SAME
hyperparameter configs as v1's, at the new horizon.

## Fixed (not sampled)

- **Base model:** Qwen2.5-0.5B (chosen for
  runs-per-night on one GPU)
- **Task:** SFT via LoRA on **databricks-dolly-15k**, frozen split drawn
  once at freeze time with seed 2026: 12,000 train / 1,500 val (rest unused)
- **Metric:** val cross-entropy on the frozen 500-example eval subset
  (`val_sub`, hash above) — the "curve"
- **Horizon:** fixed 750 optimizer steps; eval every 15 steps → **50 val
  evals per run** (curve indices 1..50, final = eval 50 — dimensionally
  identical to the CC-01b/CC-02 machinery)
- **Precision/settings:** bf16, cosine LR schedule to zero, no restarts
  (mirrors LCBench's annealing), max seq len 1024, gradient checkpointing on
- **Per-run training seed:** drawn from the config RNG and recorded (part of
  the config draw, so exchangeability covers seed variation too)

## Sampled hyperparameters (i.i.d. per run)

| Param | Distribution |
|---|---|
| learning rate | log-uniform [1e-5, 1e-3] |
| LoRA rank | uniform on {4, 8, 16, 32} |
| LoRA alpha | 2 × rank (deterministic, not sampled) |
| effective batch size | uniform on {8, 16, 32, 64} (via grad accumulation) |
| warmup fraction | uniform [0, 0.10] |
| weight decay | log-uniform [1e-5, 1e-1] |
| LoRA dropout | uniform [0, 0.30] |

Six sampled dims + recorded seed — deliberately close to LCBench's seven, so
the "wide random prior over a sane box" character carries over.

## RNG protocol

One master seed sequence per night: `night_seed = 20260725 + night_index`
(night_index increments every armed night, gaps allowed). Configs for that
night are draws 1..N from `numpy.default_rng(night_seed)` in a fixed field
order (lr, rank, batch, warmup, wd, dropout, train_seed). N = whatever the
night completes; partial nights are fine (exchangeability is per-draw, and
incomplete final runs are discarded, never partially used).

---
**Freeze record (2026-07-25):**
- Adam confirmed freeze; header updated.
- Dataset: `databricks-dolly-15k.jsonl`, **15,011 rows**,
  SHA256 `2df9083338b4abd6bceb5635764dab5d833b393b55759dffb0959b6fcbf794ec`
  (re-downloadable from Hugging Face, `databricks/databricks-dolly-15k`).
- Split drawn once, `numpy.default_rng(2026)` permutation of 15,011:
  train = first 12,000 (sorted-index sha256/16 `1127a5d3b6a49be6`),
  val = next 1,500 (`3af277058dfaa1da`), 1,511 unused.
