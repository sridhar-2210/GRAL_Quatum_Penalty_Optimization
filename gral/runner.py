"""The outer optimisation loop shared by every penalty controller.

One *episode* solves one instance under a fixed budget:
    for round k = 1..K:
        build QUBO from current (w, mu)      # controller's decision
        run the solver: R reads x S sweeps   # identical for every method
        measure feedback -> controller.update(...)
The best feasible tour seen in any round is the episode's answer.

Because every method gets the same K, R, S and seeds, comparisons are fair.
`RoundStats` is exactly the information the learned controllers observe.
The solver is pluggable: simulated annealing by default, QAOA in phase 4.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .qubo import TSPEncoding
from .sa import anneal
from .tsp import TSPInstance, tour_length


@dataclass
class Budget:
    rounds: int = 10          # K
    reads: int = 32           # R samples per round
    sweeps: int = 200         # S SA sweeps per read (ignored by solvers without sweeps)


@dataclass
class RoundStats:
    k: int
    K: int
    w: np.ndarray
    mu: np.ndarray
    viol_freq: np.ndarray       # per constraint: fraction of samples with g_c != 0
    mean_g: np.ndarray          # per constraint: mean residual g_c (signed)
    mean_abs_g: np.ndarray      # per constraint: mean |g_c|
    best_g: np.ndarray          # per constraint: residual of the lowest-energy sample
    feas_frac: float            # fraction of samples that are valid tours
    round_best_len: float       # best feasible normalised length this round (inf if none)
    best_len: float             # best feasible normalised length so far (inf if none)
    mean_violation: float       # mean total sum_c g_c^2


@dataclass
class EpisodeResult:
    instance: str
    method: str
    best_length: float          # original units, inf if never feasible
    optimum: float | None
    gap_pct: float              # 100 (L - L*) / L*, inf if infeasible
    found_feasible: bool
    first_feasible_round: int   # 1-based, -1 if never
    mean_feas_frac: float
    seconds: float
    best_tour: list | None
    mean_opt_frac: float = float("nan")   # fraction of samples that are an optimal tour
    history: list = field(default_factory=list)

    def row(self):
        return {k: v for k, v in self.__dict__.items() if k not in ("history", "best_tour")}


def sa_solver(W, h, off, budget: Budget, seed: int):
    return anneal(W, h, off, num_reads=budget.reads, num_sweeps=budget.sweeps, seed=seed)


class RoundTracker:
    """Runs solver rounds for one instance and keeps the best feasible tour.
    Used by run_episode (baselines + learned controllers) and by the RL env,
    so training and evaluation see exactly the same mechanics."""

    def __init__(self, enc: TSPEncoding, budget: Budget, seed: int, solver=sa_solver):
        self.enc, self.budget, self.solver = enc, budget, solver
        if hasattr(solver, "reset"):          # e.g. QAOA warm-start angles: fresh per episode
            solver.reset()
        self.rng = np.random.default_rng(seed)
        self.best_len, self.best_tour, self.first = np.inf, None, -1
        self.k = 0
        self.feas_fracs, self.opt_fracs = [], []

    def round(self, w, mu) -> RoundStats:
        enc = self.enc
        W, h, off = enc.build(w, mu)
        X, E = self.solver(W, h, off, self.budget, int(self.rng.integers(2**31)))
        G = enc.g(X)
        feas = np.all(G == 0, axis=1)
        lens = enc.distance_hat(X)
        round_best = float(lens[feas].min()) if feas.any() else np.inf
        if feas.any():
            j = np.flatnonzero(feas)[np.argmin(lens[feas])]
            if round_best < self.best_len - 1e-12:
                self.best_len, self.best_tour = round_best, enc.decode(X[j])
            if self.first < 0:
                self.first = self.k + 1
        self.feas_fracs.append(float(feas.mean()))
        opt = enc.inst.optimum
        if opt:
            hit = feas & np.isclose(lens * enc.scale, opt, rtol=1e-9, atol=1e-6)
            self.opt_fracs.append(float(hit.mean()))
        stats = RoundStats(
            k=self.k, K=self.budget.rounds, w=np.array(w, float), mu=np.array(mu, float),
            viol_freq=(G != 0).mean(0), mean_g=G.mean(0), mean_abs_g=np.abs(G).mean(0),
            best_g=G[np.argmin(E)], feas_frac=float(feas.mean()),
            round_best_len=round_best, best_len=self.best_len,
            mean_violation=float((G ** 2).sum(1).mean()))
        self.k += 1
        return stats

    def result(self, inst: TSPInstance, method: str, secs: float, history=None):
        L = tour_length(inst.dist, self.best_tour) if self.best_tour is not None else np.inf
        opt = inst.optimum
        gap = 100.0 * (L - opt) / opt if (opt and np.isfinite(L)) else np.inf
        return EpisodeResult(inst.name, method, L, opt, gap, bool(np.isfinite(L)), self.first,
                             float(np.mean(self.feas_fracs)), secs, self.best_tour,
                             history=history or [],
                             mean_opt_frac=float(np.mean(self.opt_fracs)) if self.opt_fracs else float("nan"))


def run_episode(inst: TSPInstance, controller, budget: Budget = Budget(),
                seed: int = 0, keep_history: bool = False, solver=sa_solver) -> EpisodeResult:
    enc = TSPEncoding(inst)
    tr = RoundTracker(enc, budget, seed, solver)
    w, mu = controller.reset(enc)
    hist = []
    t0 = time.perf_counter()
    for _ in range(budget.rounds):
        stats = tr.round(w, mu)
        if keep_history:
            hist.append(stats)
        w, mu = controller.update(stats)
    secs = time.perf_counter() - t0
    return tr.result(inst, getattr(controller, "name", type(controller).__name__), secs, hist)
