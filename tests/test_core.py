"""Correctness checks for phase 1.  Run:  python -m pytest -q tests   (or python tests/test_core.py)"""
import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gral import (TSPEncoding, random_euclidean, burma14, held_karp, tour_length,  # noqa: E402
                  anneal, Budget, run_episode, FixedPenalty, ALMDualAscent)
from gral.sa import _anneal_kernel  # noqa: E402


def brute_force_opt(dist):
    n = dist.shape[0]
    return min(tour_length(dist, (0,) + p) for p in itertools.permutations(range(1, n)))


def test_held_karp_matches_brute_force():
    for seed in range(5):
        inst = random_euclidean(7, seed, solve=False)
        assert abs(held_karp(inst.dist)[0] - brute_force_opt(inst.dist)) < 1e-9


def test_burma14_optimum_is_3323():
    assert burma14().optimum == 3323


def test_feasible_tours_energy_equals_length():
    inst = random_euclidean(5, 3)
    enc = TSPEncoding(inst)
    rng = np.random.default_rng(0)
    w, mu = rng.uniform(0.1, 5, enc.n_constraints), rng.normal(0, 2, enc.n_constraints)
    W, h, off = enc.build(w, mu)
    for p in itertools.permutations(range(1, 5)):
        tour = (0,) + p
        x = enc.encode(tour)
        assert enc.decode(x) == list(tour)
        assert np.allclose(enc.g(x), 0)
        assert np.isclose(enc.energy(x, W, h, off)[0], tour_length(inst.dist, tour) / enc.scale)


def test_energy_matches_definition_on_random_states():
    inst = random_euclidean(5, 4)
    enc = TSPEncoding(inst)
    rng = np.random.default_rng(1)
    w, mu = rng.uniform(0.1, 5, enc.n_constraints), rng.normal(0, 2, enc.n_constraints)
    W, h, off = enc.build(w, mu)
    X = rng.integers(0, 2, (200, enc.N))
    G = enc.g(X)
    direct = enc.distance_hat(X) + (G ** 2) @ w + G @ mu
    assert np.allclose(enc.energy(X, W, h, off), direct)


def test_qubo_ground_state_is_optimal_tour():
    inst = random_euclidean(4, 7)                 # 9 binaries -> 512 states
    enc = TSPEncoding(inst)
    W, h, off = enc.build(2.0)                    # w > max normalised edge -> feasible GS
    X = np.array(list(itertools.product([0, 1], repeat=enc.N)))
    E = enc.energy(X, W, h, off)
    best = X[np.argmin(E)]
    tour = enc.decode(best)
    assert tour is not None
    assert np.isclose(tour_length(inst.dist, tour), inst.optimum)


def test_numba_kernel_logic_matches_numpy_backend_quality():
    # runs the compiled-path kernel in pure Python on a tiny problem
    inst = random_euclidean(4, 2)
    enc = TSPEncoding(inst)
    W, h, off = enc.build(2.0)
    X, E = anneal(W, h, off, num_reads=8, num_sweeps=60, seed=0, backend="numba")
    tours = [enc.decode(x) for x in X]
    assert any(t is not None and np.isclose(tour_length(inst.dist, t), inst.optimum) for t in tours)
    Xn, En = anneal(W, h, off, num_reads=8, num_sweeps=60, seed=0, backend="numpy")
    assert np.isclose(En.min(), E.min())


def test_sa_solves_small_instance():
    inst = random_euclidean(6, 11)
    r = run_episode(inst, FixedPenalty(2.0), Budget(rounds=3, reads=32, sweeps=200), seed=0)
    assert r.found_feasible and r.gap_pct < 5


def test_alm_runs():
    inst = random_euclidean(6, 12)
    r = run_episode(inst, ALMDualAscent(), Budget(rounds=4, reads=16, sweeps=150), seed=0,
                    keep_history=True)
    assert len(r.history) == 4


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
