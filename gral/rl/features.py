"""Observation features, action mapping and reward for the learned controllers.
Pure NumPy, shared by the training environment and the evaluation controller,
so a trained policy sees exactly the same inputs at test time.

Observation for an instance with C = 2(n-1) constraints:
    node  (C, F_NODE)  one row per constraint
    glob  (F_GLOB,)    global solver-progress features
    adj   (C, C)       row-normalised constraint graph (constraints sharing a variable)
Action:
    "gral"   : (C, 2)  per constraint  [dlog w_c, dmu_c]   (each in [-1, 1])
    "scalar" : (1,)    one global dlog lambda, mu fixed at 0
"""
from __future__ import annotations

import numpy as np

from ..controllers import W_MAX, W_MIN
from ..qubo import TSPEncoding
from ..tsp import nearest_neighbour_length

F_NODE = 7
F_GLOB = 8
W0 = 1.0                 # same starting penalty as every baseline
DLOGW = 0.5              # max change of log w per round
DMU = 0.5                # max change of mu per round (normalised units)
INFEAS_RATIO = 1.5       # score assigned while no feasible tour exists
REWARD_SCALE = 10.0


def reference_length(enc: TSPEncoding) -> float:
    """Nearest-neighbour tour length in normalised units (no exact solver needed)."""
    return nearest_neighbour_length(enc.inst.dist) / enc.scale


def adjacency(enc: TSPEncoding) -> np.ndarray:
    C = enc.n_constraints
    A = (enc.A @ enc.A.T > 0).astype(float)
    np.fill_diagonal(A, 0.0)
    deg = A.sum(1, keepdims=True)
    return A / np.maximum(deg, 1.0)


def ratio(best_len: float, ref: float) -> float:
    return best_len / ref if np.isfinite(best_len) else INFEAS_RATIO


def observe(enc: TSPEncoding, s, ref: float, prev_ratio: float, adj=None):
    m = enc.m
    is_city = np.r_[np.ones(m), np.zeros(m)]
    node = np.stack([np.log(s.w), s.mu / 10.0, s.viol_freq, s.mean_g, s.mean_abs_g,
                     s.best_g, is_city], axis=1).astype(np.float32)
    r = ratio(s.best_len, ref)
    rr = ratio(s.round_best_len, ref)
    glob = np.array([s.feas_frac, float(np.isfinite(s.best_len)), r, rr,
                     (s.k + 1) / s.K, s.mean_violation / enc.n_constraints,
                     np.log(enc.n) / np.log(20.0), r - prev_ratio], dtype=np.float32)
    return {"node": node, "glob": glob,
            "adj": (adjacency(enc) if adj is None else adj).astype(np.float32)}


def initial_penalties(enc: TSPEncoding):
    C = enc.n_constraints
    return np.full(C, W0), np.zeros(C)


# Residual base rules — must stay identical to gral.controllers
#   gral   -> PerConstraintRule(w0=1, tau=0.2, up=1.5, down=0.9)
#   scalar -> ScalarAdaptive(lam0=1, target=0.3, up=1.5, down=0.85)
RULE_TAU, RULE_UP, RULE_DOWN = 0.2, 1.5, 0.9
SCALAR_TARGET, SCALAR_UP, SCALAR_DOWN = 0.3, 1.5, 0.85


def rule_factor(s, mode: str):
    """Multiplicative update the hand-designed rule would apply after round s."""
    if mode == "scalar":
        return SCALAR_UP if s.feas_frac < SCALAR_TARGET else SCALAR_DOWN
    return np.where(s.viol_freq > RULE_TAU, RULE_UP, RULE_DOWN)


def apply_action(w, mu, action, mode: str, s=None, residual: bool = True):
    """Map a policy action to the next (w, mu).
    residual=True  (GRAL v2): learned correction on top of the rule-based update,
                   so action = 0 reproduces the rule exactly;
    residual=False (v1 / ablation): action changes (w, mu) directly from W0."""
    a = np.clip(np.asarray(action, float), -1.0, 1.0)
    if mode == "scalar":
        base = rule_factor(s, "scalar") if residual else 1.0
        lam = float(np.clip(w.mean() * base * np.exp(DLOGW * a.reshape(-1)[0]), W_MIN, W_MAX))
        return np.full_like(w, lam), np.zeros_like(mu)
    a = a.reshape(len(w), 2)
    base = rule_factor(s, "gral") if residual else 1.0
    w = np.clip(w * base * np.exp(DLOGW * a[:, 0]), W_MIN, W_MAX)
    mu = np.clip(mu + DMU * a[:, 1], -W_MAX, W_MAX)
    return w, mu


def reward(prev_ratio: float, new_ratio: float) -> float:
    """Telescoping: the episode return equals REWARD_SCALE * (initial - final
    ratio), i.e. how much the best tour improved relative to the
    nearest-neighbour reference, with infeasibility worst of all."""
    return REWARD_SCALE * (prev_ratio - new_ratio)
