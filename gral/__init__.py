"""GRAL: Graph-RL Augmented-Lagrangian penalty control for hybrid
classical-quantum combinatorial optimisation."""
from .tsp import TSPInstance, random_euclidean, make_split, burma14, held_karp, tour_length
from .qubo import TSPEncoding
from .sa import anneal, HAVE_NUMBA
from .runner import Budget, RoundStats, EpisodeResult, run_episode
from .controllers import (FixedPenalty, ScalarAdaptive, PerConstraintRule,
                          ALMDualAscent, tune_fixed)

__version__ = "0.1.0"
