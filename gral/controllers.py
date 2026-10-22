"""Penalty controllers.  Each one decides (w, mu) before every solver round.

Interface:
    w, mu = ctrl.reset(encoding)          # start of an episode
    w, mu = ctrl.update(round_stats)      # after each round

Baselines implemented here (the methods GRAL must beat):
    FixedPenalty          one scalar lambda for every constraint, never changes
    ScalarAdaptive        one scalar lambda, raised/lowered by a feasibility rule
                          (the kind of hand-written rule used in most projects)
    PerConstraintRule     one weight per constraint, multiplicative rule-based update
    ALMDualAscent         classical augmented Lagrangian: per-constraint multipliers
                          mu_c <- mu_c + rho * g_c, with rho increased on stagnation
The learned controllers (scalar PPO, GRAL) live in gral/rl/ (phase 2).
"""
from __future__ import annotations

import numpy as np

W_MIN, W_MAX = 0.05, 50.0      # safe bounds for penalty weights (normalised units)


class FixedPenalty:
    def __init__(self, lam: float):
        self.lam = float(lam)
        self.name = f"fixed(lam={lam:g})"

    def reset(self, enc):
        self.nc = enc.n_constraints
        return np.full(self.nc, self.lam), np.zeros(self.nc)

    def update(self, s):
        return s.w, s.mu


class ScalarAdaptive:
    """lam *= up if too few feasible samples, *= down if plenty (bounded)."""
    def __init__(self, lam0=1.0, target=0.3, up=1.5, down=0.85):
        self.lam0, self.target, self.up, self.down = lam0, target, up, down
        self.name = "scalar-adaptive"

    def reset(self, enc):
        self.nc = enc.n_constraints
        self.lam = self.lam0
        return np.full(self.nc, self.lam), np.zeros(self.nc)

    def update(self, s):
        self.lam *= self.up if s.feas_frac < self.target else self.down
        self.lam = float(np.clip(self.lam, W_MIN, W_MAX))
        return np.full(self.nc, self.lam), np.zeros(self.nc)


class PerConstraintRule:
    """w_c grows for constraints that are frequently violated, decays otherwise."""
    def __init__(self, w0=1.0, tau=0.2, up=1.5, down=0.9):
        self.w0, self.tau, self.up, self.down = w0, tau, up, down
        self.name = "per-constraint-rule"

    def reset(self, enc):
        self.nc = enc.n_constraints
        self.w = np.full(self.nc, self.w0)
        return self.w.copy(), np.zeros(self.nc)

    def update(self, s):
        self.w = np.where(s.viol_freq > self.tau, self.w * self.up, self.w * self.down)
        self.w = np.clip(self.w, W_MIN, W_MAX)
        return self.w.copy(), np.zeros(self.nc)


class ALMDualAscent:
    """Augmented Lagrangian method.  E = D + sum mu_c g_c + (rho/2) sum g_c^2.
    Dual ascent on mu using the mean residual of the sample batch; rho is
    multiplied by `grow` whenever total violation fails to drop by `shrink`."""
    def __init__(self, rho0=1.0, grow=1.5, shrink=0.75, use="mean"):
        self.rho0, self.grow, self.shrink, self.use = rho0, grow, shrink, use
        self.name = "ALM-dual-ascent"

    def reset(self, enc):
        self.nc = enc.n_constraints
        self.rho = self.rho0
        self.mu = np.zeros(self.nc)
        self.prev_v = np.inf
        return np.full(self.nc, self.rho / 2), self.mu.copy()

    def update(self, s):
        g = s.mean_g if self.use == "mean" else s.best_g
        self.mu = np.clip(self.mu + self.rho * g, -W_MAX, W_MAX)
        if s.mean_violation > self.shrink * self.prev_v:
            self.rho = min(self.rho * self.grow, 2 * W_MAX)
        self.prev_v = s.mean_violation
        return np.full(self.nc, self.rho / 2), self.mu.copy()


def tune_fixed(instances, budget, grid=(0.25, 0.5, 1, 2, 4, 8), seeds=(0, 1, 2)):
    """Pick the fixed lambda with the best mean score on *training/validation*
    instances (score = gap if feasible, large penalty otherwise).  Returns
    (best_lambda, table).  The cost of this search must be reported."""
    from .runner import run_episode
    table = {}
    for lam in grid:
        scores = []
        for inst in instances:
            for sd in seeds:
                r = run_episode(inst, FixedPenalty(lam), budget, seed=sd)
                scores.append(r.gap_pct if r.found_feasible else 1000.0)
        table[lam] = float(np.mean(scores))
    best = min(table, key=table.get)
    return best, table
