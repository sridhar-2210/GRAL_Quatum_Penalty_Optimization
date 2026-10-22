"""QAOA solver checks (NumPy only).  Run: python tests/test_qaoa.py"""
import os
import sys

import numpy as np
from scipy.linalg import expm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gral import Budget, TSPEncoding, random_euclidean, run_episode, FixedPenalty  # noqa: E402
from gral.qaoa import QAOASolver, apply_mixer, qaoa_state, qubo_spectrum, basis_bits  # noqa: E402

X = np.array([[0, 1], [1, 0]], dtype=complex)
I2 = np.eye(2, dtype=complex)


def _op_on(q, N, op):
    """Dense operator acting on qubit q (little-endian: qubit 0 is the rightmost kron factor)."""
    out = np.array([[1.0 + 0j]])
    for k in reversed(range(N)):
        out = np.kron(out, op if k == q else I2)
    return out


def test_mixer_matches_dense_expm():
    N, beta = 4, 0.37
    HM = sum(_op_on(q, N, X) for q in range(N))
    rng = np.random.default_rng(0)
    psi = rng.normal(size=2 ** N) + 1j * rng.normal(size=2 ** N)
    ref = expm(-1j * beta * HM) @ psi
    assert np.allclose(apply_mixer(psi.copy(), beta, N), ref)


def test_full_circuit_matches_dense_expm():
    N = 4
    rng = np.random.default_rng(1)
    C = rng.uniform(0, 1, 2 ** N)
    theta = np.array([0.9, 2.1, 0.4, 0.2])            # p = 2
    HM = sum(_op_on(q, N, X) for q in range(N))
    psi = np.full(2 ** N, 2 ** (-N / 2), dtype=complex)
    for l in range(2):
        psi = np.exp(-1j * theta[l] * C) * psi
        psi = expm(-1j * theta[2 + l] * HM) @ psi
    assert np.allclose(qaoa_state(theta, C, N), psi)


def test_spectrum_matches_encoding_energy():
    enc = TSPEncoding(random_euclidean(4, 2))
    W, h, off = enc.build(1.3, np.linspace(-1, 1, enc.n_constraints))
    E = qubo_spectrum(W, h, off)
    B = basis_bits(enc.N)
    assert np.allclose(E, enc.energy(B, W, h, off))
    # bit order: basis index i has variable k = bit k of i
    assert B[5].tolist()[:3] == [1, 0, 1]


def test_qaoa_concentrates_on_valid_tours():
    """9 qubits: only 6 of 512 bitstrings are valid tours (1.2%).  Optimised QAOA
    must put clearly more probability on them than uniform sampling."""
    inst = random_euclidean(4, 7)
    enc = TSPEncoding(inst)
    W, h, off = enc.build(1.0)
    solver = QAOASolver(p=3, maxiter=150)
    Xs, Es = solver(W, h, off, Budget(reads=2000), seed=0)
    feas = np.all(enc.g(Xs) == 0, axis=1)
    assert Xs.shape == (2000, enc.N)
    assert np.allclose(Es, enc.energy(Xs, W, h, off))
    assert feas.mean() > 3 * 6 / 512, feas.mean()      # >= 3x uniform


def test_qaoa_in_outer_loop_and_reset():
    inst = random_euclidean(4, 3)
    solver = QAOASolver(p=2, maxiter=60)
    r1 = run_episode(inst, FixedPenalty(1.0), Budget(rounds=3, reads=128), seed=0, solver=solver)
    r2 = run_episode(inst, FixedPenalty(1.0), Budget(rounds=3, reads=128), seed=0, solver=solver)
    assert r1.best_length == r2.best_length          # reset() makes episodes independent
    assert len(solver.history) == 3


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
