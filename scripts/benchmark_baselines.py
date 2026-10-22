"""Phase-1 benchmark: compare penalty baselines under an identical solver budget.

Usage (from the project root):
    python scripts/benchmark_baselines.py --sizes 10 12 --count 8 --seeds 2
Writes results/baselines_raw.csv and results/baselines_summary.csv.

Protocol (keep it identical for every later method, including GRAL):
  * TRAIN split (seed range 1_000_000+) is used only to tune the fixed lambda.
  * TEST split (seed range 9_000_000+) is never used for tuning.
  * Every method gets the same Budget and the same run seeds.
"""
import argparse
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gral import (Budget, run_episode, make_split, burma14, FixedPenalty,  # noqa: E402
                  ScalarAdaptive, PerConstraintRule, ALMDualAscent)

TRAIN_SEED0, TEST_SEED0 = 1_000_000, 9_000_000


def _job(args):
    inst, ctrl, budget, seed = args
    return run_episode(inst, ctrl, budget, seed=seed).row()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[10, 12])
    ap.add_argument("--count", type=int, default=8, help="test instances per size")
    ap.add_argument("--seeds", type=int, default=2, help="independent runs per instance")
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--reads", type=int, default=32)
    ap.add_argument("--sweeps", type=int, default=200)
    ap.add_argument("--burma", action="store_true", help="also run burma14")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    budget = Budget(a.rounds, a.reads, a.sweeps)

    # 1) tune the fixed lambda on TRAIN instances only (cost is reported)
    t0 = time.perf_counter()
    train = make_split(a.sizes, count=2, seed0=TRAIN_SEED0)
    grid = (0.25, 0.5, 1, 2, 4)
    jobs = [(inst, FixedPenalty(l), budget, 0) for l in grid for inst in train]
    with Pool(a.workers) as p:
        rows = p.map(_job, jobs)
    tune = pd.DataFrame(rows)
    tune["lam"] = [j[1].lam for j in jobs]
    tune["score"] = np.where(tune.found_feasible, tune.gap_pct, 1000.0)
    lam_star = float(tune.groupby("lam").score.mean().idxmin())
    tune_secs = time.perf_counter() - t0
    print(f"grid-tuned lambda = {lam_star}  (tuning took {tune_secs:.0f}s, "
          f"{len(jobs)} episodes)")
    tune.to_csv(os.path.join(a.out, "tuning_fixed_lambda.csv"), index=False)

    # 2) evaluate every baseline on the TEST split
    test = make_split(a.sizes, count=a.count, seed0=TEST_SEED0)
    if a.burma:
        test.append(burma14())
    methods = [FixedPenalty(1.0), FixedPenalty(lam_star), ScalarAdaptive(),
               PerConstraintRule(), ALMDualAscent()]
    methods[1].name = f"fixed-grid-tuned(lam={lam_star:g})"
    jobs = [(inst, m, budget, s) for inst in test for m in methods for s in range(a.seeds)]
    t0 = time.perf_counter()
    with Pool(a.workers) as p:
        rows = p.map(_job, jobs)
    df = pd.DataFrame(rows)
    df["n"] = [j[0].n for j in jobs]
    df.to_csv(os.path.join(a.out, "baselines_raw.csv"), index=False)

    df["gap_f"] = df.gap_pct.replace(np.inf, np.nan)
    df["first_f"] = df.first_feasible_round.replace(-1, np.nan)   # failed runs excluded
    summ = (df.groupby(["n", "method"])
              .agg(runs=("gap_pct", "size"),
                   feasible_rate=("found_feasible", "mean"),
                   mean_gap=("gap_f", "mean"),
                   std_gap=("gap_f", "std"),
                   optimal_hits=("gap_f", lambda g: float((g < 1e-9).mean())),
                   mean_first_feas=("first_f", "mean"),
                   mean_feas_frac=("mean_feas_frac", "mean"),
                   mean_secs=("seconds", "mean"))
              .round(3).reset_index())
    summ.to_csv(os.path.join(a.out, "baselines_summary.csv"), index=False)
    pd.set_option("display.width", 200)
    print(summ.to_string(index=False))
    print(f"evaluation took {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
