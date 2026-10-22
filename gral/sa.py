"""Simulated annealing for QUBOs written as  E = 0.5 x^T W x + h.x + offset.

Two back-ends with identical semantics:
  * numba (compiled; used automatically when numba is installed, e.g. Colab)
  * numpy (vectorised over reads; slower, no dependencies)

Single-spin-flip Metropolis, geometric inverse-temperature schedule.
Default beta range follows the usual heuristic: at the hottest point the
largest possible energy change is accepted with prob 1/2, at the coldest the
smallest non-zero coefficient change is accepted with prob 1/100.
"""
from __future__ import annotations

import numpy as np

try:
    from numba import njit
    HAVE_NUMBA = True
except Exception:          # pragma: no cover
    HAVE_NUMBA = False

    def njit(*a, **k):
        def deco(f):
            return f
        return deco(a[0]) if a and callable(a[0]) else deco


def default_beta_range(W, h):
    max_delta = np.max(np.abs(h) + np.abs(W).sum(1))
    coeffs = np.concatenate([np.abs(h), np.abs(W[np.triu_indices_from(W, 1)])])
    coeffs = coeffs[coeffs > 1e-12]
    min_delta = coeffs.min() if coeffs.size else 1.0
    return np.log(2.0) / max_delta, np.log(100.0) / min_delta


@njit(cache=True)
def _anneal_kernel(W, h, betas, X, rand_order, rand_u):
    R, N = X.shape
    S = betas.shape[0]
    for r in range(R):
        x = X[r]
        f = h.copy()
        for i in range(N):
            if x[i] == 1:
                for j in range(N):
                    f[j] += W[i, j]
        for s in range(S):
            beta = betas[s]
            for q in range(N):
                i = rand_order[r, s, q]
                dE = f[i] if x[i] == 0 else -f[i]
                if dE <= 0.0 or rand_u[r, s, q] < np.exp(-beta * dE):
                    delta = 1.0 if x[i] == 0 else -1.0
                    x[i] = 1 - x[i]
                    for j in range(N):
                        f[j] += delta * W[i, j]
    return X


def _anneal_numpy(W, h, betas, X, rng):
    R, N = X.shape
    Xf = X.astype(float)
    F = Xf @ W + h
    rows = np.arange(R)
    for beta in betas:
        for i in rng.permutation(N):
            xi = Xf[:, i]
            dE = np.where(xi == 0, F[:, i], -F[:, i])
            acc = (dE <= 0) | (rng.random(R) < np.exp(-beta * np.clip(dE, 0, None)))
            if not acc.any():
                continue
            r = rows[acc]
            delta = 1.0 - 2.0 * xi[r]
            Xf[r, i] = 1.0 - xi[r]
            F[r] += delta[:, None] * W[i]
    return Xf.astype(np.int8)


def anneal(W, h, offset=0.0, num_reads=32, num_sweeps=200, beta_range=None,
           seed=None, init=None, backend="auto"):
    """Return (samples int8 (R,N), energies (R,))."""
    rng = np.random.default_rng(seed)
    W = np.ascontiguousarray(W, dtype=np.float64)
    h = np.ascontiguousarray(h, dtype=np.float64)
    N = h.shape[0]
    b0, b1 = beta_range if beta_range is not None else default_beta_range(W, h)
    betas = np.geomspace(b0, b1, num_sweeps)
    if init is None:
        X = rng.integers(0, 2, size=(num_reads, N)).astype(np.int8)
    else:
        X = np.array(init, dtype=np.int8, copy=True).reshape(num_reads, N)

    use_numba = (backend == "numba") or (backend == "auto" and HAVE_NUMBA)
    if use_numba:
        order = np.argsort(rng.random((num_reads, num_sweeps, N)), axis=2).astype(np.int64)
        u = rng.random((num_reads, num_sweeps, N))
        X = _anneal_kernel(W, h, betas, X, order, u)
    else:
        X = _anneal_numpy(W, h, betas, X, rng)
    Xf = X.astype(float)
    E = 0.5 * np.einsum("ri,ij,rj->r", Xf, W, Xf) + Xf @ h + offset
    return X, E
