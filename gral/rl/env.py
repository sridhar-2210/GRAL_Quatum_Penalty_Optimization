"""Penalty-control environment (Gymnasium-style API, no gym dependency needed).

One episode = one instance, K solver rounds (same Budget as the baselines):
    reset()  -> runs round 1 at the initial penalties, returns obs
    step(a)  -> applies the action to (w, mu), runs the next round
    done after K rounds in total.

Paired ("shadow") reward, used for training when `paired=True`:
the hand-designed rule is run on the SAME instance with the SAME solver seed
sequence, and the agent is rewarded only for doing better than it:
    r_k = SCALE * [ (prev_k - ratio_k) - (base_prev_k - base_ratio_k) ]
Summed over the episode this is SCALE * (base_final - agent_final): the agent's
improvement over the rule.  Pairing removes instance difficulty and most of
the solver noise from the learning signal (common random numbers).
"""
from __future__ import annotations

import numpy as np

from ..controllers import PerConstraintRule, ScalarAdaptive
from ..qubo import TSPEncoding
from ..runner import Budget, RoundTracker, sa_solver
from ..tsp import random_euclidean
from . import features as F

TRAIN_SEED_LO, TRAIN_SEED_HI = 100_000_000, 2_000_000_000   # disjoint from val/test seeds


class PenaltyEnv:
    def __init__(self, mode="gral", sizes=(6, 7, 8, 9, 10, 11, 12), budget=Budget(),
                 seed=0, solver=sa_solver, residual=True, paired=True):
        assert mode in ("gral", "scalar")
        self.mode, self.sizes, self.budget, self.solver = mode, tuple(sizes), budget, solver
        self.residual, self.paired = residual, paired
        self.rng = np.random.default_rng(seed)

    def _base_ctrl(self):
        return ScalarAdaptive() if self.mode == "scalar" else PerConstraintRule()

    def reset(self, inst=None, seed=None):
        if inst is None:
            n = int(self.rng.choice(self.sizes))
            inst = random_euclidean(n, seed=int(self.rng.integers(TRAIN_SEED_LO, TRAIN_SEED_HI)),
                                    solve=False)
        self.inst = inst
        self.enc = TSPEncoding(inst)
        self.ref = F.reference_length(self.enc)
        self.adj = F.adjacency(self.enc)
        tseed = int(self.rng.integers(2**31)) if seed is None else seed
        self.base_ratios = None
        if self.paired:                       # shadow run of the rule, same solver seeds
            tr = RoundTracker(self.enc, self.budget, tseed, self.solver)
            ctrl = self._base_ctrl()
            w, mu = ctrl.reset(self.enc)
            self.base_ratios = []
            for _ in range(self.budget.rounds):
                s = tr.round(w, mu)
                self.base_ratios.append(F.ratio(s.best_len, self.ref))
                w, mu = ctrl.update(s)
        # agent tracker created after the shadow run (resets any stateful solver)
        self.tracker = RoundTracker(self.enc, self.budget, tseed, self.solver)
        self.w, self.mu = F.initial_penalties(self.enc)
        self.prev_ratio = F.INFEAS_RATIO
        s = self.tracker.round(self.w, self.mu)
        return self._obs(s)

    def _obs(self, s):
        obs = F.observe(self.enc, s, self.ref, self.prev_ratio, self.adj)
        self.last_stats = s
        self._ratio = F.ratio(s.best_len, self.ref)
        return obs

    def step(self, action):
        prev = self._ratio
        self.prev_ratio = prev
        self.w, self.mu = F.apply_action(self.w, self.mu, action, self.mode,
                                         self.last_stats, self.residual)
        s = self.tracker.round(self.w, self.mu)
        obs = self._obs(s)
        k = self.tracker.k - 1                                   # index of the round just run
        r = F.reward(prev, self._ratio)
        if self.paired:
            r -= F.reward(self.base_ratios[k - 1], self.base_ratios[k])
        done = self.tracker.k >= self.budget.rounds
        info = {"ratio": self._ratio}
        if self.paired:
            info["base_ratio"] = self.base_ratios[k]
        return obs, r, done, info

    def first_reward(self):
        """Reward for round 1 (before the first action; identical for agent and
        shadow because both start from the same penalties and seeds)."""
        r = F.reward(F.INFEAS_RATIO, self._ratio)
        if self.paired:
            r -= F.reward(F.INFEAS_RATIO, self.base_ratios[0])
        return r
