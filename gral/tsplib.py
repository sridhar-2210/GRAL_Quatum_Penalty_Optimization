"""Official TSPLIB benchmark instances (Reinelt 1991), read from the original
.tsp files shipped in gral/data/tsplib/.

Supported: EDGE_WEIGHT_TYPE EUC_2D, CEIL_2D, ATT, GEO, EXPLICIT with
EDGE_WEIGHT_FORMAT FULL_MATRIX, UPPER_ROW, LOWER_ROW, UPPER_DIAG_ROW,
LOWER_DIAG_ROW.  Distances follow the TSPLIB95 definitions exactly (nint
rounding, GEO with PI = 3.141592 and RRR = 6378.388), so published optima apply.
"""
from __future__ import annotations

import math
import os

import numpy as np

from .tsp import TSPInstance, held_karp

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "tsplib")

# Published optimal tour lengths (TSPLIB95 solutions file).
OPTIMA = {"burma14": 3323, "ulysses16": 6859, "gr17": 2085, "gr21": 2707,
          "ulysses22": 7013, "gr24": 1272, "fri26": 937, "bayg29": 1610,
          "bays29": 2020, "dantzig42": 699, "swiss42": 1273}

# Benchmark set used in the paper: every instance whose QUBO has <= 784 binaries.
BENCHMARK = ["burma14", "ulysses16", "gr17", "gr21", "ulysses22", "gr24",
             "fri26", "bayg29", "bays29"]


def _nint(x):
    return int(x + 0.5)


def _geo_rad(v):
    deg = int(v)
    return 3.141592 * (deg + 5.0 * (v - deg) / 3.0) / 180.0


def parse(path: str):
    with open(path) as f:
        lines = f.read().splitlines()
    spec, i = {}, 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line == "EOF":
            i += 1
            continue
        if line.startswith(("NODE_COORD_SECTION", "EDGE_WEIGHT_SECTION", "DISPLAY_DATA_SECTION")):
            break
        if ":" in line:
            k, v = line.split(":", 1)
            spec[k.strip()] = v.strip()
        i += 1
    n = int(spec["DIMENSION"])
    sections, cur = {}, None
    for line in lines[i:]:
        s = line.strip()
        if not s or s == "EOF":
            continue
        if s.endswith("SECTION") or s.split()[0].endswith("_SECTION"):
            cur = s.split()[0]
            sections[cur] = []
            continue
        sections[cur].extend(s.split())
    return spec, n, sections


def _distance_matrix(spec, n, sec):
    kind = spec["EDGE_WEIGHT_TYPE"]
    d = np.zeros((n, n))
    if kind == "EXPLICIT":
        vals = [float(v) for v in sec["EDGE_WEIGHT_SECTION"]]
        fmt = spec["EDGE_WEIGHT_FORMAT"]
        it = iter(vals)
        if fmt == "FULL_MATRIX":
            d = np.array(vals[: n * n]).reshape(n, n)
        elif fmt in ("UPPER_ROW", "LOWER_ROW", "UPPER_DIAG_ROW", "LOWER_DIAG_ROW"):
            diag = "DIAG" in fmt
            for a in range(n):
                if fmt.startswith("UPPER"):
                    rng = range(a if diag else a + 1, n)
                else:
                    rng = range(0, a + 1 if diag else a)
                for b in rng:
                    d[a, b] = d[b, a] = next(it)
        else:
            raise ValueError(f"unsupported EDGE_WEIGHT_FORMAT {fmt}")
        np.fill_diagonal(d, 0.0)
        return d
    xy = np.array([float(v) for v in sec["NODE_COORD_SECTION"]]).reshape(n, 3)[:, 1:]
    for a in range(n):
        for b in range(a + 1, n):
            dx, dy = xy[a] - xy[b]
            if kind == "EUC_2D":
                v = _nint(math.sqrt(dx * dx + dy * dy))
            elif kind == "CEIL_2D":
                v = math.ceil(math.sqrt(dx * dx + dy * dy))
            elif kind == "ATT":
                r = math.sqrt((dx * dx + dy * dy) / 10.0)
                t = _nint(r)
                v = t + 1 if t < r else t
            elif kind == "GEO":
                la, lo = _geo_rad(xy[a][0]), _geo_rad(xy[a][1])
                lb, lob = _geo_rad(xy[b][0]), _geo_rad(xy[b][1])
                q1, q2, q3 = math.cos(lo - lob), math.cos(la - lb), math.cos(la + lb)
                v = int(6378.388 * math.acos(0.5 * ((1 + q1) * q2 - (1 - q1) * q3)) + 1.0)
            else:
                raise ValueError(f"unsupported EDGE_WEIGHT_TYPE {kind}")
            d[a, b] = d[b, a] = v
    return d


def load(name: str, exact_check: bool = False) -> TSPInstance:
    """Load an official TSPLIB instance with its published optimum.
    exact_check=True also re-derives the optimum by Held-Karp (n <= 17)."""
    spec, n, sec = parse(os.path.join(DATA_DIR, f"{name}.tsp"))
    d = _distance_matrix(spec, n, sec)
    inst = TSPInstance(name=name, dist=d, optimum=float(OPTIMA[name]),
                       meta={"source": "TSPLIB", "type": spec["EDGE_WEIGHT_TYPE"],
                             "format": spec.get("EDGE_WEIGHT_FORMAT", "")})
    if exact_check:
        inst.meta["held_karp"] = held_karp(d)[0]
    return inst


def benchmark():
    return [load(n) for n in BENCHMARK]
