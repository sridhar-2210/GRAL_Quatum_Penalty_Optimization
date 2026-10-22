"""QAOA solver — the quantum half of the hybrid classical–quantum loop.

Exact statevector simulation in NumPy (no quantum SDK needed; cross-checked
against Qiskit in tests/test_qiskit_crosscheck.py).

One call = one solver round of the outer penalty loop:
  1. build the diagonal cost Hamiltonian C(x) from the current QUBO (w, mu),
  2. classical optimiser (COBYLA) tunes the 2p circuit angles (gamma, beta) to
     minimise the expected QUBO energy  <psi(gamma,beta)| C |psi(gamma,beta)>,
     warm-started from the previous round's angles,
  3. measures `budget.reads` shots and returns them as samples.
So there are two nested loops: inner = QAOA angles (classical optimiser on a
quantum circuit), outer = constraint penalties (rule / learned controller).

Conventions: qubit k <-> binary variable k <-> bit k of the basis-state index
(little-endian, same as Qiskit).  Mixer = standard transverse field,
exp(-i beta sum_k X_k) = prod_k RX_k(2 beta).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

_BITS = {}


def basis_bits(N: int) -> np.ndarray:
    if N not in _BITS:
        idx = np.arange(2 ** N, dtype=np.int64)
        _BITS[N] = ((idx[:, None] >> np.arange(N)) & 1).astype(np.int8)
    return _BITS[N]


def qubo_spectrum(W, h, off):
    """Energy of every basis state (vector of length 2^N)."""
    B = basis_bits(len(h)).astype(float)
    return 0.5 * np.einsum("ij,ij->i", B @ W, B) + B @ h + off


def apply_mixer(psi: np.ndarray, beta: float, N: int) -> np.ndarray:
    c, s = np.cos(beta), -1j * np.sin(beta)
    for q in range(N):
        v = psi.reshape(2 ** (N - q - 1), 2, 2 ** q)
        a, b = v[:, 0, :].copy(), v[:, 1, :].copy()
        v[:, 0, :] = c * a + s * b
        v[:, 1, :] = s * a + c * b
    return psi


def qaoa_state(theta, C, N: int) -> np.ndarray:
    """|psi(gamma, beta)> for p = len(theta)//2 layers; theta = [g1..gp, b1..bp]."""
    p = len(theta) // 2
    psi = np.full(2 ** N, 2 ** (-N / 2), dtype=complex)
    for l in range(p):
        psi *= np.exp(-1j * theta[l] * C)
        psi = apply_mixer(psi, theta[p + l], N)
    return psi


class QAOASolver:
    """Callable solver: (W, h, off, budget, seed) -> (samples (R,N) int8, energies (R,))."""

    def __init__(self, p: int = 3, maxiter: int = 120, warm_start: bool = True, max_qubits: int = 18):
        self.p, self.maxiter, self.warm_start, self.max_qubits = p, maxiter, warm_start, max_qubits
        self.reset()

    def reset(self):
        """Called at the start of every episode so no information leaks between runs."""
        self.theta = None
        self.history = []

    def _ramp(self):
        p = self.p
        t = (np.arange(p) + 0.5) / p
        return np.concatenate([np.pi * t, (np.pi / 4) * (1 - t)])   # linear-ramp schedule

    def __call__(self, W, h, off, budget, seed):
        N = len(h)
        if N > self.max_qubits:
            raise ValueError(f"{N} qubits is too many for statevector QAOA (max {self.max_qubits})")
        rng = np.random.default_rng(seed)
        E = qubo_spectrum(W, h, off)
        span = E.max() - E.min()
        C = (E - E.min()) / (span if span > 0 else 1.0)       # phases use a [0,1]-scaled cost

        def expected_energy(theta):
            psi = qaoa_state(theta, C, N)
            return float(np.dot(np.abs(psi) ** 2, E))

        starts = [self._ramp()]
        if self.warm_start and self.theta is not None:
            starts.insert(0, self.theta)
        best = None
        for x0 in starts:
            r = minimize(expected_energy, x0, method="COBYLA",
                         options={"maxiter": self.maxiter, "rhobeg": 0.3})
            if best is None or r.fun < best.fun:
                best = r
        self.theta = np.array(best.x)
        probs = np.abs(qaoa_state(self.theta, C, N)) ** 2
        probs /= probs.sum()
        idx = rng.choice(len(probs), size=budget.reads, p=probs)
        self.history.append({"expected_energy": best.fun, "evals": int(best.nfev)})
        X = basis_bits(N)[idx].copy()
        return X, E[idx]
