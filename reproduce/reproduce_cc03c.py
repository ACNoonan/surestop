"""The package must reproduce CC-03c's numbers exactly, from the curves in this directory.

CC-03c was a pre-registered experiment: 400 LoRA fine-tuning runs, two calibration/test splits
fixed before any outcome was read, alpha = 0.05, tau pinned in advance. This script re-runs the
kill rule the package ships (last observed value as predictor) on those splits and compares
against the numbers recorded when the experiment closed. If any number differs, the package has
drifted from the construction that was measured, and none of the measured numbers may be quoted
for it.

Three checks per split:

  1. last value, no hold           == the "LASTVAL" reference row recorded at close
  2. last value, hold pinned at 18 == the exploratory Addendum E1 row
  3. last value, hold_until_resolved=0.90 from peek 3 reproduces E1's hold step and its row

The pre-registered rule itself used a fitted power-law predictor, which this package does not
ship (it ranked runs worse than the last value at every early step). Its numbers are printed
from the targets file for comparison, not reproduced.

Run:  python reproduce/reproduce_cc03c.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
from surestop import KillRule  # noqa: E402


def main() -> int:
    pool = json.loads((HERE / "cc03c_pool.json").read_text())
    targets = json.loads((HERE / "cc03c_targets.json").read_text())
    Y = np.array([r["val_ce"] for r in pool["runs"]], dtype=float)
    assert Y.shape == (400, 50), Y.shape
    alpha, tau = targets["alpha"], targets["tau_good"]
    ok = True
    print(f"pool {Y.shape[0]} runs x {Y.shape[1]} evaluations · alpha {alpha} · tau {tau:.6f}\n")
    for arm in ("R", "T"):
        t = targets["arms"][arm]
        cal, test = np.array(t["cal_idx"]), np.array(t["test_idx"])
        pre = t["preregistered_pow3"]
        print(f"arm {arm} — {t['split']}")
        print(f"  pre-registered pow3 rule (not shipped): fk {pre['false_kills']} "
              f"· joint {pre['joint_fk']:.4f} · savings {pre['savings']:.4f} · {pre['row']}")
        e1 = t["last_value_hold_0.90"]
        checks = [
            ("last value, no hold", dict(min_peek=2), t["last_value_no_hold"], None),
            (f"last value, hold pinned at {e1['k_start']}", dict(min_peek=e1["k_start"] - 1), e1, None),
            ("last value, hold_until_resolved 0.90", dict(min_peek=2, hold_until_resolved=0.90),
             e1, e1["k_start"]),
        ]
        for name, kw, want, want_k in checks:
            rule = KillRule(alpha=alpha, good_threshold=tau, **kw).fit(Y[cal])
            got = rule.evaluate(Y[test])
            match = (got["false_kills"] == want["false_kills"]
                     and abs(got["savings"] - want["savings"]) < 1e-9
                     and (want_k is None or rule.first_peek_ + 1 == want_k))
            ok &= match
            print(f"  {name:38s} fk {got['false_kills']:2d} (want {want['false_kills']:2d}) "
                  f"· savings {got['savings']:.4f} (want {want['savings']:.4f})"
                  + (f" · hold step {rule.first_peek_ + 1} (want {want_k})" if want_k else "")
                  + f"  {'OK' if match else 'MISMATCH'}")
        null = t["null_rule_kill_everything"]
        print(f"  null rule, kill everything at peek 3: fk {null['false_kills']} "
              f"· joint {null['joint_fk']:.4f} · {null['verdict']}  (control: the gate caught it)\n")
    rs = targets["resplits_last_value_hold_0.90"]
    print(f"2000 resplits, last value with hold 0.90: joint mean {rs['joint_mean']:.4f} "
          f"· breach rate {rs['breach_rate']:.3f} · savings mean {rs['savings_mean']:.3f} "
          f"· best run killed {rs['best_killed_rate']:.3f}  (recorded at close, not recomputed here)\n")
    print("REPRODUCES CC-03c" if ok else "DOES NOT REPRODUCE — do not quote CC-03c numbers for this package")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
