"""Wraps a trained policy as a controller with the same reset/update interface
as the baselines, so it is evaluated by exactly the same run_episode code."""
from __future__ import annotations

import numpy as np
import torch

from . import features as F
from .policy import load


class LearnedController:
    def __init__(self, policy, name=None, deterministic=True):
        if isinstance(policy, str):
            policy = load(policy)
        self.policy = policy.eval()
        self.mode = policy.mode
        self.residual = bool(getattr(policy, "residual", True))
        self.deterministic = deterministic
        base = "GRAL" if self.mode == "gral" else "scalar-PPO"
        self.name = name or (base if self.residual else base + "-noresidual")

    def reset(self, enc):
        self.enc = enc
        self.ref = F.reference_length(enc)
        self.adj = F.adjacency(enc)
        self.prev = F.INFEAS_RATIO
        self.w, self.mu = F.initial_penalties(enc)
        return self.w.copy(), self.mu.copy()

    @torch.no_grad()
    def act(self, obs):
        t = {k: torch.as_tensor(v)[None] for k, v in obs.items()}
        d, _ = self.policy.dist(t["node"], t["glob"], t["adj"])
        a = d.mean if self.deterministic else d.sample()
        return a[0].numpy()

    def update(self, s):
        obs = F.observe(self.enc, s, self.ref, self.prev, self.adj)
        self.prev = F.ratio(s.best_len, self.ref)
        self.w, self.mu = F.apply_action(self.w, self.mu, self.act(obs), self.mode, s, self.residual)
        return self.w.copy(), self.mu.copy()
