"""Overnight run — v3 training, model selection, final classical + quantum evaluation.

    python scripts/run_phase3.py

Steps (stops at the first required failure):
  0. archive the current (v2) models to models/v2/
  1. tests: core, env, QAOA, PyTorch/PPO  (+ optional Qiskit cross-check)
  2. train GRAL v3 and scalar-PPO v3
  3. MODEL SELECTION on VALIDATION instances only: v2 vs v3 for each method;
     the better one is copied to models/final/  (the test set is never used to choose)
  4. FINAL classical evaluation (simulated annealing): all baselines + final models
     + the no-residual ablation (models/v1)
  5. FINAL quantum evaluation (QAOA, 9 and 16 qubits): same methods, controllers
     trained only on the classical solver (zero-shot transfer)
Everything printed is also saved to results/phase3_log.txt — send that file back.
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOG = os.path.join(ROOT, "results", "phase3_log.txt")


def run(cmd, log, required=True):
    line = f"\n===== {' '.join(cmd)}  ({time.strftime('%H:%M:%S')}) ====="
    print(line, flush=True)
    log.write(line + "\n")
    p = subprocess.Popen([sys.executable, "-u"] + cmd, cwd=ROOT, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1)
    for out in p.stdout:
        print(out, end="", flush=True)
        log.write(out)
        log.flush()
    ok = p.wait() == 0
    if not ok:
        msg = (f"\n!!!!! FAILED: {' '.join(cmd)}" +
               (" — stopping. Send results/phase3_log.txt to Claude." if required
                else " — optional step, continuing."))
        print(msg)
        log.write(msg + "\n")
        if required:
            sys.exit(1)
    return ok


def say(msg, log):
    print(msg, flush=True)
    log.write(msg + "\n")
    log.flush()


def exists(rel):
    return os.path.exists(os.path.join(ROOT, rel))


def select(log, candidates, out_name):
    """candidates: list of (path, name); pick lowest mean validation gap."""
    df = pd.read_csv(os.path.join(ROOT, "results", "select_raw.csv"))
    df["g"] = df.gap_pct.replace(np.inf, 100.0)
    means = df.groupby("method").g.mean()
    present = [(p, n) for p, n in candidates if n in means.index]
    best = min(present, key=lambda pn: means[pn[1]])
    for p, n in present:
        say(f"  validation mean gap {n:16s} {means[n]:.3f}%" + ("   <- selected" if (p, n) == best else ""), log)
    os.makedirs(os.path.join(ROOT, "models", "final"), exist_ok=True)
    shutil.copy(os.path.join(ROOT, best[0]), os.path.join(ROOT, "models", "final", out_name))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=150)
    ap.add_argument("--skip-train", action="store_true", help="reuse models/gral_best.pt etc.")
    a = ap.parse_args()
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    t0 = time.time()
    with open(LOG, "a", encoding="utf-8") as log:
        # 0) archive v2
        if not a.skip_train and exists("models/gral_best.pt") and not exists("models/v2"):
            os.makedirs(os.path.join(ROOT, "models", "v2"))
            for f in glob.glob(os.path.join(ROOT, "models", "*.*")):
                shutil.move(f, os.path.join(ROOT, "models", "v2"))
            say("archived v2 models to models/v2", log)

        # 1) tests
        for t in ("tests/test_core.py", "tests/test_env.py", "tests/test_qaoa.py", "tests/test_rl.py"):
            run([t], log)
        run(["tests/test_qiskit_crosscheck.py"], log, required=False)

        # 2) train v3
        if not a.skip_train:
            run(["scripts/train.py", "--mode", "gral", "--iters", str(a.iters)], log)
            run(["scripts/train.py", "--mode", "scalar", "--iters", str(a.iters)], log)

        # 3) model selection on validation instances
        gral_c = [(p, n) for p, n in (("models/v2/gral_best.pt", "GRAL-v2"), ("models/gral_best.pt", "GRAL-v3")) if exists(p)]
        scal_c = [(p, n) for p, n in (("models/v2/scalar_best.pt", "scalar-PPO-v2"), ("models/scalar_best.pt", "scalar-PPO-v3")) if exists(p)]
        run(["scripts/evaluate.py", "--split", "val", "--no-baselines", "--sizes", "10", "12", "14",
             "--count", "8", "--seeds", "2", "--burma-seeds", "0", "--tag", "select", "--models"]
            + [f"{p}={n}" for p, n in gral_c + scal_c], log)
        say("\nMODEL SELECTION (validation instances only):", log)
        g = select(log, gral_c, "gral.pt")
        s = select(log, scal_c, "scalar.pt")
        say(f"final GRAL = {g[1]}, final scalar-PPO = {s[1]}", log)

        models = ["models/final/gral.pt=GRAL", "models/final/scalar.pt=scalar-PPO"]
        models += [f"{p}={n}" for p, n in (("models/v1/gral_best.pt", "GRAL-noresidual"),
                                           ("models/v1/scalar_best.pt", "scalar-PPO-noresidual")) if exists(p)]

        # 4) final classical evaluation
        run(["scripts/evaluate.py", "--tag", "final_sa", "--models"] + models, log)

        # 5) final quantum evaluation (QAOA)
        run(["scripts/evaluate.py", "--tag", "final_qaoa", "--solver", "qaoa", "--p", "3",
             "--maxiter", "120", "--sizes", "4", "5", "--count", "10", "--seeds", "3",
             "--rounds", "8", "--reads", "128", "--burma-seeds", "0", "--models"] + models, log)

        say(f"\nALL DONE in {(time.time() - t0) / 60:.1f} min. Send results/phase3_log.txt to Claude.", log)


if __name__ == "__main__":
    main()
