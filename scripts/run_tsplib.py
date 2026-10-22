"""Official TSPLIB benchmark:  python scripts/run_tsplib.py

Evaluates every baseline and the final trained controllers (no retraining) on
9 official TSPLIB instances (burma14 ... bays29, 14-29 cities, 169-784 binary
variables), 10 independent runs each, at two solver budgets:
  A. the standard budget used in training  (10 rounds x 32 samples x 200 sweeps)
  B. a 5x larger budget                   (10 rounds x 32 samples x 1000 sweeps)
Fixed lambda is grid-tuned on random TRAIN instances of 14, 21 and 29 cities.
Everything printed is saved to results/tsplib_log.txt — send that file back.
Options: --only A | B
"""
import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOG = os.path.join(ROOT, "results", "tsplib_log.txt")
ENV = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
           NUMBA_NUM_THREADS="1")       # one thread per worker process: no CPU oversubscription


def run(cmd, log):
    line = f"\n===== {' '.join(cmd)}  ({time.strftime('%H:%M:%S')}) ====="
    print(line, flush=True)
    log.write(line + "\n")
    p = subprocess.Popen([sys.executable, "-u"] + cmd, cwd=ROOT, env=ENV, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1)
    for out in p.stdout:
        print(out, end="", flush=True)
        log.write(out)
        log.flush()
    if p.wait() != 0:
        msg = f"\n!!!!! FAILED: {' '.join(cmd)} — send results/tsplib_log.txt to Claude"
        print(msg)
        log.write(msg + "\n")
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["A", "B"])
    a = ap.parse_args()
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    models = ["models/final/gral.pt=GRAL", "models/final/scalar.pt=scalar-PPO"]
    models += [f"{p}={n}" for p, n in (("models/v1/gral_best.pt", "GRAL-noresidual"),
                                       ("models/v1/scalar_best.pt", "scalar-PPO-noresidual"))
               if os.path.exists(os.path.join(ROOT, p))]
    common = ["scripts/evaluate.py", "--tsplib", "--seeds", "10", "--tune-sizes", "14", "21", "29",
              "--rounds", "10", "--reads", "32", "--models"] + models
    t0 = time.time()
    with open(LOG, "a", encoding="utf-8") as log:
        run(["tests/test_core.py"], log)
        run(["tests/test_tsplib.py"], log)
        if a.only in (None, "A"):
            run(common + ["--sweeps", "200", "--tag", "tsplib_b200"], log)
        if a.only in (None, "B"):
            run(common + ["--sweeps", "1000", "--tag", "tsplib_b1000"], log)
        msg = f"\nALL DONE in {(time.time() - t0) / 60:.1f} min. Send results/tsplib_log.txt to Claude."
        print(msg)
        log.write(msg + "\n")


if __name__ == "__main__":
    main()
