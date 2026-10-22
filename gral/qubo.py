"""Per-constraint augmented-Lagrangian QUBO encoding of the TSP.

Variables: x[i,t] = 1 if city i is visited at tour position t.
City 0 is fixed at position 0 (removes the rotational symmetry), so only
cities 1..n-1 and positions 1..n-1 are encoded: m = n-1, N = m*m binaries.
For n = 4 that is 9 qubits; n = 5 -> 16; burma14 -> 169.

Constraints (2m one-hot equalities), g_c(x) = sum(x over the group) - 1:
    c = 0..m-1      : city   c+1 appears exactly once
    c = m..2m-1     : position c-m+1 holds exactly one city

Energy minimised by the solver:
    E(x; w, mu) = D_hat(x) + sum_c [ w_c * g_c(x)^2 + mu_c * g_c(x) ]

D_hat is the tour length divided by `scale` (default: max distance), so all
coefficients are O(1) regardless of the instance's units.  w_c >= 0 is the
per-constraint quadratic penalty, mu_c is the per-constraint Lagrange
multiplier.  A single scalar lambda is the special case w_c = lambda, mu_c = 0.
For any feasible tour every g_c = 0, so E equals the normalised tour length.
"""
from __future__ import annotations

import numpy as np

from .tsp import TSPInstance


class TSPEncoding:
    def __init__(self, inst: TSPInstance, scale: float | None = None):
        self.inst = inst
        self.n = inst.n
        self.m = self.n - 1
        self.N = self.m * self.m
        self.n_constraints = 2 * self.m
        self.scale = float(scale if scale is not None else inst.dist.max())
        self.d = inst.dist / self.scale

        m, N = self.m, self.N
        # membership matrix A (n_constraints x N): A[c, k] = 1 if var k in group c
        A = np.zeros((self.n_constraints, N))
        for i in range(1, self.n):
            for t in range(1, self.n):
                k = self.idx(i, t)
                A[i - 1, k] = 1.0
                A[m + t - 1, k] = 1.0
        self.A = A

        # distance part as an upper-triangular QUBO + linear vector
        Qd = np.zeros((N, N))
        hd = np.zeros(N)
        for i in range(1, self.n):
            hd[self.idx(i, 1)] += self.d[0, i]          # 0 -> first city
            hd[self.idx(i, m)] += self.d[i, 0]          # last city -> 0
        for t in range(1, m):
            for i in range(1, self.n):
                for j in range(1, self.n):
                    if i == j:
                        continue
                    a, b = self.idx(i, t), self.idx(j, t + 1)
                    lo, hi = min(a, b), max(a, b)
                    Qd[lo, hi] += self.d[i, j]
        self.Qd, self.hd = Qd, hd

        # pairs inside each constraint group (for the g^2 cross terms)
        self.groups = [np.flatnonzero(A[c]) for c in range(self.n_constraints)]

    # ------------------------------------------------------------------ #
    def idx(self, i: int, t: int) -> int:
        return (i - 1) * self.m + (t - 1)

    def build(self, w, mu=None):
        """Return (W, h, offset): symmetric zero-diagonal coupling matrix W,
        linear field h and constant, so that  E(x) = x^T Q x + offset  with
        Q = triu(W) + diag(h), i.e. E = 0.5 x^T W x + h.x + offset."""
        w = np.broadcast_to(np.asarray(w, float), (self.n_constraints,))
        mu = np.zeros(self.n_constraints) if mu is None else \
            np.broadcast_to(np.asarray(mu, float), (self.n_constraints,))
        Q = self.Qd.copy()
        h = self.hd.copy()
        offset = 0.0
        for c, grp in enumerate(self.groups):
            # w*g^2 = w*( -sum x_k + 2 sum_{k<l} x_k x_l + 1 )
            # mu*g  = mu*(  sum x_k - 1 )
            h[grp] += -w[c] + mu[c]
            offset += w[c] - mu[c]
            for a in range(len(grp)):
                for b in range(a + 1, len(grp)):
                    Q[grp[a], grp[b]] += 2.0 * w[c]
        W = Q + Q.T
        np.fill_diagonal(W, 0.0)
        return W, h, offset

    # ------------------------------------------------------------------ #
    def energy(self, X, W, h, offset):
        X = np.atleast_2d(X).astype(float)
        return 0.5 * np.einsum("ri,ij,rj->r", X, W, X) + X @ h + offset

    def g(self, X):
        """Constraint residuals g_c(x) for every sample: shape (R, n_constraints)."""
        return np.atleast_2d(X) @ self.A.T - 1.0

    def distance_hat(self, X):
        X = np.atleast_2d(X).astype(float)
        return np.einsum("ri,ij,rj->r", X, self.Qd, X) + X @ self.hd

    def decode(self, x):
        """Tour (list of cities starting at 0) if x is feasible, else None."""
        M = np.asarray(x).reshape(self.m, self.m)       # rows: city-1, cols: pos-1
        if not (np.all(M.sum(0) == 1) and np.all(M.sum(1) == 1)):
            return None
        return [0] + [int(np.argmax(M[:, t])) + 1 for t in range(self.m)]

    def encode(self, tour):
        tour = list(tour)
        k0 = tour.index(0)
        tour = tour[k0:] + tour[:k0]
        x = np.zeros(self.N)
        for t in range(1, self.n):
            x[self.idx(tour[t], t)] = 1.0
        return x

    def constraint_graph(self):
        """Edges between constraints that share a variable (city c <-> position p
        share exactly x[c, p]).  Bipartite K_{m,m}.  Used by the GNN policy."""
        m = self.m
        src, dst = [], []
        for ci in range(m):
            for pj in range(m):
                src += [ci, m + pj]
                dst += [m + pj, ci]
        return np.array([src, dst])
