"""Policies (PyTorch).

GRALPolicy  — the proposed method.  A message-passing GNN over the constraint
              graph.  Every constraint is a node; all nodes share weights, so
              one trained network works for any number of cities.  Outputs a
              per-constraint action [dlog w_c, dmu_c] and a state value.
ScalarPolicy — ablation / learned baseline.  Pools the same features into one
              vector and outputs a single dlog lambda (classic "RL tunes λ").

Both start with near-zero action means.  With residual=True (v2, default)
action 0 means "apply the hand-designed rule", so a fresh policy behaves exactly
like the best rule-based baseline and training learns corrections to it.
`residual` is stored in the checkpoint (old v1 checkpoints load as False).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from .features import F_GLOB, F_NODE


def mlp(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, o))


def _small_last(m: nn.Sequential, scale=0.01):
    with torch.no_grad():
        m[-1].weight.mul_(scale)
        m[-1].bias.zero_()


class GRALPolicy(nn.Module):
    mode = "gral"

    def __init__(self, hidden: int = 64, n_layers: int = 3, log_std0: float = -2.0):
        super().__init__()
        self.kwargs = dict(hidden=hidden, n_layers=n_layers, log_std0=log_std0)
        self.enc = mlp(F_NODE + F_GLOB, hidden, hidden)
        self.layers = nn.ModuleList([mlp(3 * hidden, hidden, hidden) for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(n_layers)])
        self.head = mlp(hidden, hidden, 2)
        self.value = mlp(2 * hidden + F_GLOB, hidden, 1)
        self.log_std = nn.Parameter(torch.full((2,), log_std0))
        _small_last(self.head)

    def forward(self, node, glob, adj):
        B, C, _ = node.shape
        x = torch.cat([node, glob[:, None, :].expand(B, C, glob.shape[-1])], -1)
        h = self.enc(x)
        for layer, norm in zip(self.layers, self.norms):
            msg = torch.bmm(adj, h)                              # neighbours' mean
            pool = h.mean(1, keepdim=True).expand_as(h)          # whole-graph context
            h = norm(h + layer(torch.cat([h, msg, pool], -1)))
        mean = self.head(h)                                      # (B, C, 2)
        v = self.value(torch.cat([h.mean(1), h.max(1).values, glob], -1)).squeeze(-1)
        return mean, v

    def dist(self, node, glob, adj):
        mean, v = self(node, glob, adj)
        return Normal(mean, self.log_std.exp().expand_as(mean)), v


class ScalarPolicy(nn.Module):
    mode = "scalar"

    def __init__(self, hidden: int = 64, log_std0: float = -2.0, **_):
        super().__init__()
        self.kwargs = dict(hidden=hidden, log_std0=log_std0)
        i = 3 * F_NODE + F_GLOB
        self.pi = mlp(i, hidden, 1)
        self.value = mlp(i, hidden, 1)
        self.log_std = nn.Parameter(torch.full((1,), log_std0))
        _small_last(self.pi)

    def _pool(self, node, glob):
        return torch.cat([node.mean(1), node.max(1).values, node.min(1).values, glob], -1)

    def forward(self, node, glob, adj=None):
        z = self._pool(node, glob)
        return self.pi(z), self.value(z).squeeze(-1)             # (B, 1), (B,)

    def dist(self, node, glob, adj=None):
        mean, v = self(node, glob, adj)
        return Normal(mean, self.log_std.exp().expand_as(mean)), v


def log_prob(d: Normal, a: torch.Tensor) -> torch.Tensor:
    """Joint log-probability per sample (sum over constraints and action dims)."""
    return d.log_prob(a).flatten(1).sum(1)


def entropy(d: Normal) -> torch.Tensor:
    return d.entropy().flatten(1).mean(1)


def make_policy(mode: str, residual: bool = True, **kwargs) -> nn.Module:
    p = {"gral": GRALPolicy, "scalar": ScalarPolicy}[mode](**kwargs)
    p.residual = residual
    return p


def save(policy: nn.Module, path: str, extra=None):
    torch.save({"mode": policy.mode, "kwargs": policy.kwargs,
                "residual": bool(getattr(policy, "residual", True)),
                "state_dict": policy.state_dict(), "extra": extra or {}}, path)


def load(path: str) -> nn.Module:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    p = make_policy(ck["mode"], **ck["kwargs"])
    p.load_state_dict(ck["state_dict"])
    p.residual = bool(ck.get("residual", False))          # v1 files have no flag
    p.eval()
    return p
