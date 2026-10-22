"""PPO training with parallel rollouts (one CPU process per worker).

Each iteration:
  1. workers play `episodes` episodes on fresh random TRAIN instances with the
     current (stochastic) policy;
  2. GAE advantages (gamma = 1, finite horizon K-1 decisions);
  3. clipped-surrogate PPO updates, batching transitions of equal graph size;
  4. every `val_every` iterations the deterministic policy is scored on fixed
     VALIDATION instances; the best checkpoint is kept.
"""
from __future__ import annotations

import csv
import os
import time
from dataclasses import asdict, dataclass
from multiprocessing import get_context

import numpy as np
import torch

from ..runner import Budget, run_episode
from ..tsp import random_euclidean
from . import features as F
from .controller import LearnedController
from .env import PenaltyEnv
from .policy import entropy, log_prob, make_policy, save

VAL_SEED0 = 5_000_000


@dataclass
class PPOConfig:
    mode: str = "gral"
    residual: bool = True        # v2: learn corrections on top of the rule-based update
    paired: bool = True          # v2: reward = improvement over the rule on the same seeds
    sizes: tuple = (8, 9, 10, 11, 12, 13, 14)     # v3: small sizes are solved by every method
    rounds: int = 10
    reads: int = 32
    sweeps: int = 200
    iters: int = 150
    episodes: int = 64
    workers: int = 8
    lr: float = 3e-4
    gamma: float = 1.0
    lam: float = 0.95
    clip: float = 0.2
    epochs: int = 4
    minibatch: int = 64
    vf_coef: float = 0.5
    ent_coef: float = 0.0
    max_grad_norm: float = 0.5
    hidden: int = 64
    n_layers: int = 3
    val_every: int = 10
    val_sizes: tuple = (10, 12, 14)
    val_count: int = 5
    val_seeds: int = 2
    seed: int = 0
    out: str = "models"


# --------------------------------------------------------------------------- #
# rollout worker (top-level so it can be pickled on Windows)
# --------------------------------------------------------------------------- #
def _rollout(args):
    mode, residual, paired, pkw, state, sizes, budget, n_eps, seed = args
    torch.set_num_threads(1)
    pol = make_policy(mode, residual=residual, **pkw)
    pol.load_state_dict(state)
    pol.eval()
    env = PenaltyEnv(mode, sizes=sizes, budget=budget, seed=seed, residual=residual, paired=paired)
    trajs = []
    for _ in range(n_eps):
        obs = env.reset()
        tr = {"obs": [], "act": [], "logp": [], "val": [], "rew": []}
        done = False
        while not done:
            t = {k: torch.as_tensor(v)[None] for k, v in obs.items()}
            with torch.no_grad():
                d, v = pol.dist(t["node"], t["glob"], t["adj"])
                a = d.sample()
                lp = log_prob(d, a)
            nobs, r, done, info = env.step(a[0].numpy())
            tr["obs"].append(obs)
            tr["act"].append(a[0].numpy())
            tr["logp"].append(float(lp))
            tr["val"].append(float(v))
            tr["rew"].append(float(r))
            obs = nobs
        tr["final_ratio"] = info["ratio"]
        tr["base_final_ratio"] = info.get("base_ratio", np.nan)
        tr["n"] = env.inst.n
        trajs.append(tr)
    return trajs


def _val_job(args):
    mode, residual, pkw, state, inst, budget, seed = args
    torch.set_num_threads(1)
    pol = make_policy(mode, residual=residual, **pkw)
    pol.load_state_dict(state)
    ctrl = LearnedController(pol)
    r = run_episode(inst, ctrl, budget, seed=seed)
    ref = F.reference_length(ctrl.enc) * ctrl.enc.scale
    return r.best_length / ref if r.found_feasible else F.INFEAS_RATIO


# --------------------------------------------------------------------------- #
def gae(rew, val, gamma, lam):
    T = len(rew)
    adv = np.zeros(T)
    last = 0.0
    for t in reversed(range(T)):
        nv = val[t + 1] if t + 1 < T else 0.0
        delta = rew[t] + gamma * nv - val[t]
        last = delta + gamma * lam * last
        adv[t] = last
    return adv, adv + np.array(val)


def _stack(obs_list):
    return {k: torch.as_tensor(np.stack([o[k] for o in obs_list])) for k in obs_list[0]}


def train(cfg: PPOConfig, log=print):
    os.makedirs(cfg.out, exist_ok=True)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    pol = make_policy(cfg.mode, residual=cfg.residual, hidden=cfg.hidden, n_layers=cfg.n_layers)
    tag = cfg.mode if cfg.residual else f"{cfg.mode}-noresidual"
    opt = torch.optim.Adam(pol.parameters(), lr=cfg.lr)
    budget = Budget(cfg.rounds, cfg.reads, cfg.sweeps)
    val_insts = [random_euclidean(n, VAL_SEED0 + 1000 * n + k, solve=False)
                 for n in cfg.val_sizes for k in range(cfg.val_count)]
    ctx = get_context("spawn")
    pool = ctx.Pool(cfg.workers)
    best_val, rows = np.inf, []
    path_best = os.path.join(cfg.out, f"{tag}_best.pt")
    path_last = os.path.join(cfg.out, f"{tag}_last.pt")
    logf = open(os.path.join(cfg.out, f"{tag}_train_log.csv"), "w", newline="")
    writer = None
    t_start = time.perf_counter()
    try:
        for it in range(1, cfg.iters + 1):
            t0 = time.perf_counter()
            state = {k: v.detach().clone() for k, v in pol.state_dict().items()}
            per = [cfg.episodes // cfg.workers + (i < cfg.episodes % cfg.workers)
                   for i in range(cfg.workers)]
            jobs = [(cfg.mode, cfg.residual, cfg.paired, pol.kwargs, state, cfg.sizes, budget, n,
                     cfg.seed * 100003 + it * 997 + i)
                    for i, n in enumerate(per) if n > 0]
            trajs = [t for ts in pool.map(_rollout, jobs) for t in ts]

            # ---- assemble batch -------------------------------------------------
            obs, act, oldlp, advs, rets = [], [], [], [], []
            for tr in trajs:
                a, r = gae(tr["rew"], tr["val"], cfg.gamma, cfg.lam)
                obs += tr["obs"]
                act += tr["act"]
                oldlp += tr["logp"]
                advs += list(a)
                rets += list(r)
            advs = np.array(advs)
            advs = (advs - advs.mean()) / (advs.std() + 1e-8)
            groups = {}
            for i, o in enumerate(obs):
                groups.setdefault(o["node"].shape[0], []).append(i)

            # ---- PPO update -----------------------------------------------------
            pol.train()
            stats = {"pi": 0.0, "v": 0.0, "kl": 0.0, "clipfrac": 0.0, "n": 0}
            for _ in range(cfg.epochs):
                batches = []
                for idx in groups.values():
                    idx = np.random.permutation(idx)
                    batches += [idx[s:s + cfg.minibatch] for s in range(0, len(idx), cfg.minibatch)]
                np.random.shuffle(batches)
                for b in batches:
                    o = _stack([obs[i] for i in b])
                    A = torch.as_tensor(np.stack([act[i] for i in b]))
                    lp0 = torch.as_tensor(np.array([oldlp[i] for i in b]), dtype=torch.float32)
                    adv = torch.as_tensor(advs[b], dtype=torch.float32)
                    ret = torch.as_tensor(np.array([rets[i] for i in b]), dtype=torch.float32)
                    d, v = pol.dist(o["node"], o["glob"], o["adj"])
                    lp = log_prob(d, A)
                    ratio = torch.exp(lp - lp0)
                    s1 = ratio * adv
                    s2 = torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * adv
                    pi_loss = -torch.min(s1, s2).mean()
                    v_loss = ((v - ret) ** 2).mean()
                    loss = pi_loss + cfg.vf_coef * v_loss - cfg.ent_coef * entropy(d).mean()
                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(pol.parameters(), cfg.max_grad_norm)
                    opt.step()
                    with torch.no_grad():
                        stats["pi"] += float(pi_loss)
                        stats["v"] += float(v_loss)
                        stats["kl"] += float((lp0 - lp).mean())
                        stats["clipfrac"] += float(((ratio - 1).abs() > cfg.clip).float().mean())
                        stats["n"] += 1
            pol.eval()
            k = max(stats.pop("n"), 1)
            stats = {key: val / k for key, val in stats.items()}

            fr = np.array([t["final_ratio"] for t in trajs])
            br = np.array([t["base_final_ratio"] for t in trajs])
            row = {"iter": it, "train_final_ratio": float(fr.mean()),
                   "train_feasible": float((fr < F.INFEAS_RATIO).mean()),
                   # % of the nearest-neighbour length by which the agent beats the rule
                   "vs_rule_pct": float(np.nanmean(br - fr) * 100) if cfg.paired else float("nan"),
                   **{f"loss_{k_}": v_ for k_, v_ in stats.items()},
                   "std": float(pol.log_std.detach().exp().mean()), "secs": time.perf_counter() - t0}

            # ---- validation -----------------------------------------------------
            if it % cfg.val_every == 0 or it == cfg.iters:
                state = {k_: v_.detach().clone() for k_, v_ in pol.state_dict().items()}
                vj = [(cfg.mode, cfg.residual, pol.kwargs, state, inst, budget, sd)
                      for inst in val_insts for sd in range(cfg.val_seeds)]
                vr = float(np.mean(pool.map(_val_job, vj)))
                row["val_ratio"] = vr
                if vr < best_val:
                    best_val = vr
                    save(pol, path_best, {"iter": it, "val_ratio": vr, "config": asdict(cfg)})
            save(pol, path_last, {"iter": it, "config": asdict(cfg)})
            if writer is None:
                writer = csv.DictWriter(logf, fieldnames=list(row.keys()) + (["val_ratio"] if "val_ratio" not in row else []))
                writer.writeheader()
            writer.writerow(row)
            logf.flush()
            rows.append(row)
            msg = (f"[{tag}] it {it:4d}/{cfg.iters}  train ratio {row['train_final_ratio']:.4f}  "
                   f"vs rule {row['vs_rule_pct']:+.2f}%  "
                   f"feasible {row['train_feasible']:.2f}  std {row['std']:.3f}  "
                   f"kl {stats['kl']:+.4f}  {row['secs']:.1f}s")
            if "val_ratio" in row:
                msg += f"  | VAL ratio {row['val_ratio']:.4f} (best {best_val:.4f})"
            log(msg)
    finally:
        pool.close()
        pool.join()
        logf.close()
    log(f"done in {(time.perf_counter() - t_start) / 60:.1f} min; best checkpoint: {path_best}")
    return pol, rows
