"""SureStop against the heuristic pruners, on the 400 CC-03c curves, over a whole sweep.

This comparison is exploratory. It was run after CC-03c closed, on curves that had already been
seen, and it was not pre-registered. Read it as a description of this pool.

**The sweep.** All 400 runs arrive one at a time in a random order, and each pruner decides
from the first run. SureStop pays for its calibration runs in full, as it would in use: the
first `n_cal` runs finish unpruned, then the rule freezes. The table averages 500 orders.

**The arms.**
- SureStop, joint and conditional guarantee, alpha = 0.05, tau pinned as in CC-03c.
- Median: Optuna's `MedianPruner`. A run is pruned when its best value so far is worse than
  the median, at the same evaluation, of the runs that completed before it.
- ASHA: Optuna's `SuccessiveHalvingPruner`, reduction factor 3, rungs at evaluations 2, 6
  and 18. A run passes a rung only if it is in the top third of the runs that reached it.

The median and ASHA rules are re-implemented here in numpy so this script needs only numpy and
scipy. `tests/test_baselines.py` replays runs through Optuna's own pruners and checks that the
stopping points agree exactly.

**The columns.**
- saved: evaluations skipped, as a share of 400 x 50.
- good killed: share of the 65 runs ending at or below tau that were killed.
  Calibration runs count in the denominator and cannot be killed, so among the runs a
  200-calibration arm actually judges, the share is about double the figure shown.
- top-5 killed: share of the five best runs that were killed.
- best killed: how often the single best run (index 199) was killed.
- regret: best surviving final loss minus the pool's best final loss, in nats.

Run:  python reproduce/baselines_cc03c.py           prints the table
      python reproduce/baselines_cc03c.py --check   also compares it with the committed output
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
from surestop import KillRule  # noqa: E402

ORDERS = 500
ASHA_RUNGS = (1, 5, 17)  # evaluations 2, 6, 18: min_resource 2, reduction factor 3


def surestop_stops(Y: np.ndarray, order: np.ndarray, tau: float, n_cal: int, min_peek: int,
                   guarantee: str = "joint", alpha: float = 0.05) -> np.ndarray:
    """Index of each run's last evaluation: T-1 if it finished, else where the rule fired."""
    n, T = Y.shape
    stop = np.full(n, T - 1)
    rule = KillRule(alpha=alpha, guarantee=guarantee, good_threshold=tau,
                    min_peek=min_peek).fit(Y[order[:n_cal]])
    for i in order[n_cal:]:
        k = rule.kill_step(Y[i])
        if k is not None:
            stop[i] = k
    return stop


def median_stops(Y: np.ndarray, order: np.ndarray, startup: int = 5, warmup: int = 0) -> np.ndarray:
    """Optuna MedianPruner(n_startup_trials=startup, n_warmup_steps=warmup), lower is better."""
    n, T = Y.shape
    stop = np.full(n, T - 1)
    done: list[int] = []
    for i in order:
        if len(done) >= startup:
            med = np.median(Y[done], axis=0)
            fire = np.flatnonzero((np.minimum.accumulate(Y[i]) > med)[warmup: T - 1])
            if fire.size:
                stop[i] = warmup + fire[0]
        if stop[i] == T - 1:
            done.append(int(i))
    return stop


def asha_stops(Y: np.ndarray, order: np.ndarray, rungs: tuple[int, ...] = ASHA_RUNGS,
               eta: int = 3) -> np.ndarray:
    """Optuna SuccessiveHalvingPruner(min_resource=2, reduction_factor=eta), lower is better."""
    n, T = Y.shape
    stop = np.full(n, T - 1)
    seen: dict[int, list[float]] = {r: [] for r in rungs}
    for i in order:
        for r in rungs:
            seen[r].append(float(Y[i, r]))
            ranked = np.sort(seen[r])
            if Y[i, r] > ranked[max(1, len(ranked) // eta) - 1]:
                stop[i] = r
                break
    return stop


def table() -> str:
    pool = json.loads((HERE / "cc03c_pool.json").read_text())
    tau = json.loads((HERE / "cc03c_targets.json").read_text())["tau_good"]
    Y = np.array([r["val_ce"] for r in pool["runs"]], dtype=float)
    n, T = Y.shape
    finals = Y[:, -1]
    good = finals <= tau
    top5 = np.argsort(finals)[:5]
    best = int(top5[0])

    arms = {
        "SureStop joint, 60 cal, no hold": lambda o: surestop_stops(Y, o, tau, 60, 2),
        "SureStop joint, 60 cal, hold 18": lambda o: surestop_stops(Y, o, tau, 60, 17),
        "SureStop joint, 200 cal, hold 18": lambda o: surestop_stops(Y, o, tau, 200, 17),
        "SureStop conditional, 200 cal, no hold": lambda o: surestop_stops(Y, o, tau, 200, 2, "conditional"),
        "SureStop conditional, 200 cal, hold 18": lambda o: surestop_stops(Y, o, tau, 200, 17, "conditional"),
        "Median, 5 startup, warmup 18": lambda o: median_stops(Y, o, 5, 17),
        "Median, 5 startup, no warmup": lambda o: median_stops(Y, o, 5, 0),
        "ASHA, eta 3, rungs 2/6/18": lambda o: asha_stops(Y, o),
    }
    rng = np.random.default_rng(0)
    rows: dict[str, list[tuple[float, ...]]] = {k: [] for k in arms}
    for _ in range(ORDERS):
        order = rng.permutation(n)
        for name, arm in arms.items():
            stop = arm(order)
            killed = stop < T - 1
            rows[name].append((
                1 - (stop + 1).sum() / (n * T),
                (killed & good).sum() / good.sum(),
                killed[top5].mean(),
                float(killed[best]),
                finals[~killed].min() - finals.min(),
            ))

    out = [
        f"pool {n} runs x {T} evaluations · tau {tau:.6f} · {int(good.sum())} good runs · "
        f"{ORDERS} random orders · alpha 0.05",
        f"final loss: min {finals.min():.4f} · median {np.median(finals):.4f} · "
        f"max {finals.max():.4f} · sd {finals.std():.4f}",
        "",
        f"{'arm':40s} {'saved':>7s} {'good killed':>12s} {'top-5 killed':>13s} "
        f"{'best killed':>12s} {'regret':>8s} {'regret p95':>11s}",
    ]
    for name, r in rows.items():
        a = np.array(r)
        m = a.mean(axis=0)
        out.append(f"{name:40s} {100 * m[0]:6.1f}% {100 * m[1]:11.1f}% {100 * m[2]:12.1f}% "
                   f"{100 * m[3]:11.1f}% {m[4]:8.4f} {np.percentile(a[:, 4], 95):11.4f}")
    means = {name: np.array(r).mean(axis=0) for name, r in rows.items()}
    G = int(good.sum())
    out += ["", "break-even cost of one killed good run, in full runs of compute:",
            "c* = (heuristic saved - SureStop saved) x n / ((heuristic killed - SureStop killed) x good)",
            "above c*, SureStop costs less in total; 'dominates' = saves at least as much and kills fewer",
            "", f"{'arm':40s} {'vs ASHA':>9s} {'vs median, warmup 18':>21s}"]
    for name, m in means.items():
        if not name.startswith("SureStop"):
            continue
        cells = []
        for rival in ("ASHA, eta 3, rungs 2/6/18", "Median, 5 startup, warmup 18"):
            h = means[rival]
            d_saved, d_killed = (h[0] - m[0]) * n, (h[1] - m[1]) * G
            cells.append("dominates" if d_saved <= 0 and d_killed > 0 else f"{d_saved / d_killed:.1f}")
        out.append(f"{name:40s} {cells[0]:>9s} {cells[1]:>21s}")
    out += ["", "EXPLORATORY: post hoc on a closed pool, not pre-registered."]
    return "\n".join(out) + "\n"


def main() -> int:
    text = table()
    print(text, end="")
    if "--check" in sys.argv:
        want = (HERE / "baselines_cc03c_RESULTS.txt").read_text()
        if text != want:
            print("\nDOES NOT MATCH baselines_cc03c_RESULTS.txt — do not quote that table")
            return 1
        print("\nMATCHES baselines_cc03c_RESULTS.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
