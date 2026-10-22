"""TSP instances, distances and exact reference solutions.

Everything here is classical and solver-independent:
  * random Euclidean instances (for training / validation / test splits)
  * TSPLIB burma14 (GEO distances, known optimum 3323)
  * Held-Karp dynamic programming for exact optima (n <= ~15)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np


@dataclass
class TSPInstance:
    name: str
    dist: np.ndarray                 # (n, n) symmetric, zero diagonal, original units
    coords: np.ndarray | None = None
    optimum: float | None = None     # exact optimal tour length if known
    meta: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return self.dist.shape[0]


# --------------------------------------------------------------------------- #
# Instance generators
# --------------------------------------------------------------------------- #
def random_euclidean(n: int, seed: int, scale: float = 100.0,
                     integer: bool = True, solve: bool = True) -> TSPInstance:
    """Uniform random cities in a [0, scale]^2 square (TSPLIB EUC_2D rounding)."""
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0, scale, size=(n, 2))
    d = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))
    if integer:
        d = np.floor(d + 0.5)
    np.fill_diagonal(d, 0.0)
    inst = TSPInstance(name=f"rand{n}_s{seed}", dist=d, coords=xy, meta={"seed": seed})
    if solve and n <= 16:
        inst.optimum, inst.meta["opt_tour"] = held_karp(d)
    return inst


def make_split(sizes, count: int, seed0: int, solve: bool = True):
    """Deterministic list of instances. Use disjoint seed ranges per split."""
    out = []
    for n in sizes:
        for k in range(count):
            out.append(random_euclidean(n, seed=seed0 + 1000 * n + k, solve=solve))
    return out


# TSPLIB burma14 (type GEO). Known optimal tour length: 3323.
_BURMA14 = [
    (16.47, 96.10), (16.47, 94.44), (20.09, 92.54), (22.39, 93.37),
    (25.23, 97.24), (22.00, 96.05), (20.47, 97.02), (17.20, 96.29),
    (16.30, 97.38), (14.05, 98.12), (16.53, 97.38), (21.52, 95.59),
    (19.41, 97.13), (20.09, 94.55),
]


def _geo_rad(v: float) -> float:
    deg = int(v)
    minutes = v - deg
    return 3.141592 * (deg + 5.0 * minutes / 3.0) / 180.0


def burma14(solve: bool = True) -> TSPInstance:
    pts = [(_geo_rad(a), _geo_rad(b)) for a, b in _BURMA14]
    n = len(pts)
    d = np.zeros((n, n))
    rrr = 6378.388
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            q1 = math.cos(pts[i][1] - pts[j][1])
            q2 = math.cos(pts[i][0] - pts[j][0])
            q3 = math.cos(pts[i][0] + pts[j][0])
            d[i, j] = int(rrr * math.acos(0.5 * ((1 + q1) * q2 - (1 - q1) * q3)) + 1.0)
    inst = TSPInstance(name="burma14", dist=d, coords=np.array(_BURMA14),
                       meta={"known_optimum": 3323})
    if solve:
        inst.optimum, inst.meta["opt_tour"] = held_karp(d)
    return inst


# --------------------------------------------------------------------------- #
# Tours
# --------------------------------------------------------------------------- #
def tour_length(dist: np.ndarray, tour) -> float:
    tour = list(tour)
    return float(sum(dist[tour[k], tour[(k + 1) % len(tour)]] for k in range(len(tour))))


def nearest_neighbour_length(dist: np.ndarray) -> float:
    """Greedy nearest-neighbour tour from city 0: a cheap reference length that
    needs no exact solver (used to normalise RL rewards and features)."""
    n = dist.shape[0]
    tour, left = [0], set(range(1, n))
    while left:
        j = min(left, key=lambda c: dist[tour[-1], c])
        tour.append(j)
        left.remove(j)
    return tour_length(dist, tour)


def held_karp(dist: np.ndarray):
    """Exact TSP by Held-Karp DP. O(n^2 2^n): fine up to n ~ 15 in Python."""
    n = dist.shape[0]
    if n <= 3:
        tour = list(range(n))
        return tour_length(dist, tour), tour
    m = n - 1                                    # cities 1..n-1, start fixed at 0
    full = 1 << m
    INF = float("inf")
    C = np.full((full, m), INF)
    P = np.full((full, m), -1, dtype=np.int64)
    for j in range(m):
        C[1 << j, j] = dist[0, j + 1]
    for size in range(2, m + 1):
        for subset in combinations(range(m), size):
            S = 0
            for b in subset:
                S |= 1 << b
            for j in subset:
                prev = S & ~(1 << j)
                best, arg = INF, -1
                for k in subset:
                    if k == j:
                        continue
                    v = C[prev, k] + dist[k + 1, j + 1]
                    if v < best:
                        best, arg = v, k
                C[S, j] = best
                P[S, j] = arg
    S = full - 1
    costs = [C[S, j] + dist[j + 1, 0] for j in range(m)]
    j = int(np.argmin(costs))
    best = float(costs[j])
    order = []
    while j != -1:
        order.append(j + 1)
        S, j = S & ~(1 << j), int(P[S, j])
    tour = [0] + order[::-1]
    return best, tour
