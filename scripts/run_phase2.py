"""Phase 2 + 3 in one command:  python scripts/run_phase2.py

Runs, in order, and stops at the first failure:
  1. tests (core, env, PyTorch/PPO)
  2. train GRAL (proposed method)
  3. train scalar PPO (learned baseline / ablation)
  4. full evaluation against every baseline, with paired statistics
Everything printed is also saved to results/phase2_log.txt — send that file back.
Options:  --iters N  (default 150)   --skip-tests   --only-eval
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def run(cmd, log):
    line = f"\n===== {' '.join(cmd)}  ({time.strftime('%H:%M:%S')}) ====="
    print(line, flush=True)
    log.write(line + "\n")
    p = subprocess.Popen([sys.executable, "-u"] + cmd, cwd=ROOT, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1)
    for out in p.stdout:
        print(out, end="", flush=True)
        log.write(out)
        log.flush()
    if p.wait() != 0:
        msg = f"\n!!!!! FAILED: {' '.join(cmd)} — send results/phase2_log.txt to Claude"
        print(msg)
        log.write(msg + "\n")
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=150)
    ap.add_argument("--skip-tests", action="store_true")
    ap.add_argument("--only-eval", action="store_true")
    a = ap.parse_args()
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    mdir, v1dir = os.path.join(ROOT, "models"), os.path.join(ROOT, "models", "v1")
    if not a.only_eval and os.path.exists(os.path.join(mdir, "gral_best.pt")) and not os.path.exists(v1dir):
        os.makedirs(v1dir)                       # keep the first (no-residual) models as ablation
        for f in glob.glob(os.path.join(mdir, "*.*")):
            shutil.move(f, v1dir)
        print("archived first-version models to models/v1 (used as the no-residual ablation)")
    t0 = time.time()
    with open(os.path.join(ROOT, "results", "phase2_log.txt"), "a", encoding="utf-8") as log:
        if not a.skip_tests and not a.only_eval:
            run(["tests/test_core.py"], log)
            run(["tests/test_env.py"], log)
            run(["tests/test_rl.py"], log)
        if not a.only_eval:
            run(["scripts/train.py", "--mode", "gral", "--iters", str(a.iters)], log)
            run(["scripts/train.py", "--mode", "scalar", "--iters", str(a.iters)], log)
        models = ["models/gral_best.pt", "models/scalar_best.pt"]
        models += [m for m in ("models/v1/gral_best.pt", "models/v1/scalar_best.pt")
                   if os.path.exists(os.path.join(ROOT, m))]
        run(["scripts/evaluate.py", "--models"] + models, log)
        msg = f"\nALL DONE in {(time.time() - t0) / 60:.1f} min. Send results/phase2_log.txt to Claude."
        print(msg)
        log.write(msg + "\n")


if __name__ == "__main__":
    main()
