"""TSPLIB reader checks.  Run: python tests/test_tsplib.py"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gral.tsp import burma14  # noqa: E402
from gral.tsplib import BENCHMARK, OPTIMA, load  # noqa: E402


def test_burma14_file_matches_builtin():
    assert np.array_equal(load("burma14").dist, burma14(solve=False).dist)


def test_exact_optimum_equals_published():
    # GEO (burma14, ulysses16) and EXPLICIT LOWER_DIAG_ROW (gr17) re-derived by Held-Karp
    for name in ("burma14", "ulysses16", "gr17"):
        assert load(name, exact_check=True).meta["held_karp"] == OPTIMA[name], name


def test_all_benchmark_instances_valid():
    for name in BENCHMARK:
        d = load(name).dist
        assert np.allclose(d, d.T) and np.allclose(np.diag(d), 0) and (d[~np.eye(len(d), dtype=bool)] > 0).all(), name


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
