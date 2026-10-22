"""Builds notebooks/01_foundation.ipynb (no nbformat dependency)."""
import json
import os

cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": s.strip("\n")})


md(r"""
# GRAL — Phase 1: Foundation and baselines

**Project:** Design and Analysis of Hybrid Classical–Quantum Algorithms for Combinatorial Optimization
**Method:** GRAL — learned per-constraint penalty control (Graph neural network + PPO + augmented Lagrangian QUBO)

This notebook builds and verifies everything the learned controller will sit on top of:
1. TSP instances (random + Burma14) with **exact optima** (Held-Karp).
2. The **per-constraint augmented-Lagrangian QUBO**: one weight $w_c$ and multiplier $\mu_c$ for every constraint.
3. A **simulated-annealing** solver.
4. Every **baseline** GRAL must beat, under a fair, identical budget.

Run cells top to bottom. Runtime → *Run all* is fine.
""")

md("## 1. Setup\nUpload `gral_project.zip` when asked (only the first time in a session).")
code(r"""
import os, sys, zipfile
if not any(os.path.exists(p) for p in ('gral_project', 'gral', '../gral')):
    try:
        from google.colab import files
        up = files.upload()                       # choose gral_project.zip
        zipfile.ZipFile(next(iter(up))).extractall('.')
    except ImportError:
        raise SystemExit('Not in Colab: run this notebook from inside the gral_project folder.')
if os.path.exists('gral_project'):
    os.chdir('gral_project')
elif os.path.exists('../gral'):
    os.chdir('..')
sys.path.insert(0, os.getcwd())
try:
    import numba
except ImportError:
    !pip -q install numba
import gral
print('gral', gral.__version__, '| numba fast path:', gral.HAVE_NUMBA)
""")

md("## 2. Correctness checks\nEvery line must say `PASS`. These prove the encoding is right before any experiment.")
code("!python tests/test_core.py")

md(r"""
## 3. The encoding in one picture

City 0 is fixed at position 0, so an $n$-city tour needs $(n-1)^2$ binaries
(4 cities → 9 qubits, 5 → 16, Burma14 → 169).
There are $2(n-1)$ one-hot constraints: each city once, each position once, with residual $g_c(x)=\sum x - 1$.

$$E(x;w,\mu)=\hat D(x)+\sum_c\big[w_c\,g_c(x)^2+\mu_c\,g_c(x)\big]$$

A single scalar λ is the special case $w_c=\lambda,\ \mu_c=0$. GRAL controls the whole vectors $w, \mu$.
""")
code(r"""
import numpy as np
from gral import *
inst = random_euclidean(6, seed=1)
enc = TSPEncoding(inst)
print(f'{inst.n} cities -> {enc.N} binary variables, {enc.n_constraints} constraints')
print('exact optimum:', inst.optimum, 'tour', inst.meta['opt_tour'])
W, h, off = enc.build(w=2.0)
x = enc.encode(inst.meta['opt_tour'])
print('QUBO energy of optimal tour x scale =', enc.energy(x, W, h, off)[0] * enc.scale)
""")

md("## 4. One episode per baseline — how each controller behaves round by round")
code(r"""
import matplotlib.pyplot as plt
inst = random_euclidean(12, seed=5)
budget = Budget(rounds=10, reads=32, sweeps=200)
scale = TSPEncoding(inst).scale
methods = [FixedPenalty(1.0), ScalarAdaptive(), PerConstraintRule(), ALMDualAscent()]
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for m in methods:
    r = run_episode(inst, m, budget, seed=0, keep_history=True)
    H = r.history
    ks = [s.k + 1 for s in H]
    ax[0].plot(ks, [s.feas_frac for s in H], marker='o', label=r.method)
    ax[1].plot(ks, [100*(s.best_len*scale - inst.optimum)/inst.optimum
                    if np.isfinite(s.best_len) else np.nan for s in H], marker='o')
    ax[2].plot(ks, [s.w.mean() for s in H], marker='o')
    print(f'{r.method:28s} gap {r.gap_pct:6.2f}%  first feasible round {r.first_feasible_round}')
ax[0].set(title='fraction of valid tours per round', xlabel='round')
ax[1].set(title='best gap so far (%)', xlabel='round')
ax[2].set(title='mean penalty weight w', xlabel='round', yscale='log')
ax[0].legend(fontsize=8); plt.tight_layout(); plt.show()
""")

md(r"""
**What to notice:** a large fixed λ makes almost every sample a valid tour but the tours are worse
(the penalty dominates the distance). Adaptive rules trade some validity for better tours.
That trade-off — learned instead of hand-written, per constraint instead of global — is what GRAL targets.
""")

md("## 5. Fair benchmark of all baselines\n"
   "Tunes the fixed λ on a TRAIN split, evaluates everything on a separate TEST split. "
   "With numba this takes roughly 10–20 minutes; reduce `--count` for a quick run.")
code("!python scripts/benchmark_baselines.py --sizes 10 12 14 --count 10 --seeds 3 --burma")

code(r"""
import pandas as pd
s = pd.read_csv('results/baselines_summary.csv')
display(s)
fig, ax = plt.subplots(figsize=(9, 4))
piv = s.pivot(index='n', columns='method', values='mean_gap')
piv.plot.bar(ax=ax, rot=0)
ax.set(ylabel='mean optimality gap (%)  — lower is better', xlabel='cities',
       title='Baselines under identical budget (TEST split)')
plt.tight_layout(); plt.show()
""")

md(r"""
## 6. Phase 1 checklist
- [ ] All tests PASS
- [ ] `results/baselines_summary.csv` saved (download it — these are the numbers GRAL must beat)
- [ ] Show the summary table to your supervisor together with the proposal

**Next — Phase 2:** wrap `run_episode` as a Gymnasium environment and train scalar PPO.
""")

nb = {"cells": cells,
      "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                   "language_info": {"name": "python"},
                   "colab": {"provenance": []}},
      "nbformat": 4, "nbformat_minor": 5}
os.makedirs("notebooks", exist_ok=True)
with open("notebooks/01_foundation.ipynb", "w") as f:
    json.dump(nb, f, indent=1)
print("wrote notebooks/01_foundation.ipynb")
