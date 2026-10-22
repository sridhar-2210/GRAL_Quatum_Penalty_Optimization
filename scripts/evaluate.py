"""Main benchmark: baselines + trained controllers under an identical budget.

    python scripts/evaluate.py --models models/gral_best.pt models/scalar_best.pt
    python scripts/evaluate.py --solver qaoa --sizes 4 5 --rounds 8 --reads 128 --burma-seeds 0 ...
    python scripts/evaluate.py --split val --no-baselines --models a.pt=GRAL-v2 b.pt=GRAL-v3 ...

* Models: `path` or `path=Name` (the name shown in tables).
* Fixed λ is grid-tuned on TRAIN instances with the same solver (tuning cost printed).
* --split test (default) uses seeds never used in training or validation;
  --split val is for model selection only.
* Sizes above 14 were never seen in training -> generalisation test.
* Paired statistics per (instance, seed): win/tie/loss and Wilcoxon signed-rank
  p-value, for the optimality gap and for the valid-sample rate.
Outputs: results/<tag>_raw.csv, <tag>_summary.csv, <tag>_paired.csv
"""
import argparse
import os
import sys
import time
from multiprocessing import get_context

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gral import (Budget, run_episode, make_split, burma14, FixedPenalty,  # noqa: E402
                  ScalarAdaptive, PerConstraintRule, ALMDualAscent)

TRAIN_SEED0, VAL_SEED0, TEST_SEED0 = 1_000_000, 5_000_000, 9_000_000
INF_GAP = 100.0      # gap charged to a run that never finds a valid tour (paired stats)


def _make_solver(spec):
    if spec[0] == "qaoa":
        from gral.qaoa import QAOASolver
        return QAOASolver(p=spec[1], maxiter=spec[2])
    from gral.runner import sa_solver
    return sa_solver


def _job(args):
    inst, spec, budget, seed, solver_spec = args
    kind, val = spec
    if kind == "model":
        import torch
        torch.set_num_threads(1)
        from gral.rl.controller import LearnedController
        path, name = val
        ctrl = LearnedController(path, name=name)
    else:
        ctrl = val
    r = run_episode(inst, ctrl, budget, seed=seed, solver=_make_solver(solver_spec)).row()
    r["group"] = f"rand{inst.n}" if inst.name.startswith("rand") else inst.name
    r["n"] = inst.n
    r["seed"] = seed
    return r


def _parse_model(m):
    path, _, name = m.partition("=")
    return path, (name or None)


def paired_table(df, learned, order, value_col, better="lower", fill=None):
    from scipy.stats import wilcoxon
    d = df.copy()
    d["v"] = d[value_col].replace(np.inf, fill) if fill is not None else d[value_col]
    sign = 1.0 if better == "lower" else -1.0
    key = ["instance", "seed"]
    rows = []
    for L in learned:
        mine = d[d.method == L].set_index(key)
        for other in d.method.unique():
            if other == L:
                continue
            theirs = d[d.method == other].set_index(key)
            for grp in order + ["ALL"]:
                sel = mine if grp == "ALL" else mine[mine.group == grp]
                j = sel[["v"]].join(theirs[["v"]], rsuffix="_o", how="inner")
                dlt = sign * (j.v - j.v_o)                       # negative = learned better
                try:
                    p = wilcoxon(dlt[dlt.abs() > 1e-12]).pvalue if (dlt.abs() > 1e-12).sum() >= 6 else np.nan
                except ValueError:
                    p = np.nan
                rows.append({"model": L, "vs": other, "group": grp, "pairs": len(j),
                             "win": int((dlt < -1e-9).sum()), "tie": int((dlt.abs() <= 1e-9).sum()),
                             "loss": int((dlt > 1e-9).sum()),
                             "mean_diff": round(float(sign * dlt.mean()), 4), "wilcoxon_p": p})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=[])
    ap.add_argument("--no-baselines", action="store_true")
    ap.add_argument("--split", choices=["test", "val"], default="test")
    ap.add_argument("--solver", choices=["sa", "qaoa"], default="sa")
    ap.add_argument("--p", type=int, default=3, help="QAOA depth")
    ap.add_argument("--maxiter", type=int, default=120, help="QAOA angle-optimiser iterations")
    ap.add_argument("--sizes", type=int, nargs="+", default=[10, 12, 14, 16])
    ap.add_argument("--count", type=int, default=10)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--burma-seeds", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--reads", type=int, default=32)
    ap.add_argument("--sweeps", type=int, default=200)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--out", default="results")
    ap.add_argument("--tag", default="eval")
    ap.add_argument("--tsplib", action="store_true",
                    help="test on the official TSPLIB benchmark set instead of random instances")
    ap.add_argument("--tune-sizes", type=int, nargs="+", default=None,
                    help="sizes of the random TRAIN instances used to grid-tune lambda")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    budget = Budget(a.rounds, a.reads, a.sweeps)
    solver_spec = ("qaoa", a.p, a.maxiter) if a.solver == "qaoa" else ("sa",)
    ctx = get_context("spawn")
    print(f"[{a.tag}] solver={a.solver} split={a.split} sizes={a.sizes} budget={budget}")

    with ctx.Pool(a.workers) as pool:
        specs = []
        if not a.no_baselines:
            # grid-tune fixed lambda on TRAIN instances with the same solver
            t0 = time.perf_counter()
            train = make_split(a.tune_sizes or a.sizes, count=2, seed0=TRAIN_SEED0, solve=False)
            grid = (0.25, 0.5, 1, 2, 4)
            jobs = [(inst, ("ctrl", FixedPenalty(l)), budget, 0, solver_spec) for l in grid for inst in train]
            tune = pd.DataFrame(pool.map(_job, jobs))
            tune["lam"] = [j[1][1].lam for j in jobs]
            # score = best length relative to nearest-neighbour (no exact optimum needed)
            from gral.tsp import nearest_neighbour_length
            nn = {inst.name: nearest_neighbour_length(inst.dist) for inst in train}
            tune["score"] = np.where(tune.found_feasible,
                                     100 * tune.best_length / tune.instance.map(nn), 1000.0)
            g = tune.groupby("lam").agg(score=("score", "mean"), valid=("mean_feas_frac", "mean"))
            g = g.assign(neg_valid=-g.valid).sort_values(["score", "neg_valid"])   # ties -> more valid samples
            lam_star = float(g.index[0])
            print(f"grid-tuned lambda = {lam_star} ({time.perf_counter() - t0:.0f}s, {len(jobs)} episodes)")
            fixed_star = FixedPenalty(lam_star)
            fixed_star.name = f"fixed-grid-tuned(lam={lam_star:g})"
            specs = [("ctrl", FixedPenalty(1.0)), ("ctrl", fixed_star), ("ctrl", ScalarAdaptive()),
                     ("ctrl", PerConstraintRule()), ("ctrl", ALMDualAscent())]
        specs += [("model", _parse_model(m)) for m in a.models]

        seed0 = TEST_SEED0 if a.split == "test" else VAL_SEED0
        if a.tsplib:
            from gral.tsplib import benchmark
            insts = benchmark()
        else:
            insts = make_split(a.sizes, count=a.count, seed0=seed0)
        jobs = [(inst, sp, budget, s, solver_spec) for inst in insts for sp in specs for s in range(a.seeds)]
        if a.burma_seeds > 0 and a.split == "test" and not a.tsplib:
            b14 = burma14()
            jobs += [(b14, sp, budget, s, solver_spec) for sp in specs for s in range(a.burma_seeds)]
        t0 = time.perf_counter()
        df = pd.DataFrame(pool.map(_job, jobs))
        print(f"evaluation: {len(jobs)} episodes in {time.perf_counter() - t0:.0f}s")

    df.to_csv(os.path.join(a.out, f"{a.tag}_raw.csv"), index=False)
    df["gap_f"] = df.gap_pct.replace(np.inf, np.nan)
    df["first_f"] = df.first_feasible_round.replace(-1, np.nan)
    nsize = df.groupby("group").n.first()
    order = sorted(df.group.unique(), key=lambda g: (not g.startswith("rand"), int(nsize[g]), g))
    summ = (df.groupby(["group", "method"])
              .agg(runs=("gap_pct", "size"),
                   feasible_rate=("found_feasible", "mean"),
                   mean_gap=("gap_f", "mean"),
                   std_gap=("gap_f", "std"),
                   median_gap=("gap_f", "median"),
                   optimal_hits=("gap_f", lambda g: float((g < 1e-9).mean())),
                   valid_sample_pct=("mean_feas_frac", lambda v: 100 * v.mean()),
                   optimal_sample_pct=("mean_opt_frac", lambda v: 100 * v.mean()),
                   mean_first_feas=("first_f", "mean"),
                   mean_secs=("seconds", "mean"))
              .round(3).reset_index())
    summ["group"] = pd.Categorical(summ.group, order, ordered=True)
    summ = summ.sort_values(["group", "mean_gap"])
    summ.to_csv(os.path.join(a.out, f"{a.tag}_summary.csv"), index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(summ.to_string(index=False))

    learned = [m for m in df.method.unique() if m.startswith(("GRAL", "scalar-PPO"))]
    if learned and not a.no_baselines:
        pg = paired_table(df, learned, order, "gap_pct", "lower", fill=INF_GAP)
        pg.insert(0, "metric", "gap_pct")
        pv = paired_table(df, learned, order, "mean_feas_frac", "higher")
        pv.insert(0, "metric", "valid_sample_rate")
        pr = pd.concat([pg, pv], ignore_index=True)
        pr.to_csv(os.path.join(a.out, f"{a.tag}_paired.csv"), index=False)
        main_metric = "valid_sample_rate" if a.solver == "qaoa" else "gap_pct"
        print(f"\nPaired comparison on {main_metric} (win = learned model better):")
        print(pr[(pr.metric == main_metric) & (pr.group == "ALL")].to_string(index=False))
        if "GRAL" in learned:
            print(f"\nGRAL per group ({main_metric}):")
            print(pr[(pr.metric == main_metric) & (pr.model == "GRAL") & (pr.group != "ALL")].to_string(index=False))


if __name__ == "__main__":
    main()
