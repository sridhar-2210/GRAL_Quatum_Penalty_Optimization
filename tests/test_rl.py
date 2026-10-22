"""PyTorch checks for the learned controllers.  Run on your laptop:
    python tests/test_rl.py
Every line must print PASS.  The last test runs 2 tiny PPO iterations (~1 min)."""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch  # noqa: E402
from gral import Budget, random_euclidean, run_episode, FixedPenalty  # noqa: E402
from gral.rl import PenaltyEnv  # noqa: E402
from gral.rl.policy import make_policy, log_prob, save, load  # noqa: E402
from gral.rl.controller import LearnedController  # noqa: E402


def _t(obs):
    return {k: torch.as_tensor(v)[None] for k, v in obs.items()}


def test_policy_shapes_any_size():
    for mode in ("gral", "scalar"):
        pol = make_policy(mode)
        for n in (5, 9, 14):
            env = PenaltyEnv(mode, sizes=(n,), budget=Budget(2, 4, 20))
            o = _t(env.reset())
            d, v = pol.dist(o["node"], o["glob"], o["adj"])
            C = env.enc.n_constraints
            assert tuple(d.mean.shape) == ((1, C, 2) if mode == "gral" else (1, 1))
            assert tuple(v.shape) == (1,)
            a = d.sample()
            assert log_prob(d, a).shape == (1,)
            env.step(a[0].numpy())


def test_zero_output_policy_equals_rule():
    """Residual design: a policy whose output is exactly 0 IS the rule-based method."""
    from gral import PerConstraintRule, ScalarAdaptive
    inst = random_euclidean(8, 3)
    b = Budget(5, 16, 100)
    for mode, rule in (("gral", PerConstraintRule()), ("scalar", ScalarAdaptive())):
        pol = make_policy(mode)
        last = (pol.head if mode == "gral" else pol.pi)[-1]
        with torch.no_grad():
            last.weight.zero_()
            last.bias.zero_()
        r1 = run_episode(inst, LearnedController(pol), b, seed=5)
        r2 = run_episode(inst, rule, b, seed=5)
        assert np.isclose(r1.best_length, r2.best_length)


def test_v1_checkpoint_loads_without_residual():
    pol = make_policy("gral")
    ck = {"mode": "gral", "kwargs": pol.kwargs, "state_dict": pol.state_dict(), "extra": {}}
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "v1.pt")
        torch.save(ck, p)                       # old format: no "residual" key
        c = LearnedController(p)
    assert c.residual is False and c.name == "GRAL-noresidual"


def test_save_load_roundtrip():
    pol = make_policy("gral")
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "m.pt")
        save(pol, p)
        q = load(p)
    env = PenaltyEnv("gral", sizes=(6,), budget=Budget(2, 4, 20))
    o = _t(env.reset())
    assert torch.allclose(pol(o["node"], o["glob"], o["adj"])[0], q(o["node"], o["glob"], o["adj"])[0])


def test_ppo_two_iterations():
    from gral.rl.ppo import PPOConfig, train
    with tempfile.TemporaryDirectory() as d:
        for mode in ("gral", "scalar"):
            cfg = PPOConfig(mode=mode, iters=2, episodes=4, workers=2, sizes=(5, 6),
                            rounds=4, reads=8, sweeps=50, val_every=2, val_sizes=(6,),
                            val_count=1, out=d)
            pol, rows = train(cfg, log=lambda *_: None)
            assert len(rows) == 2 and "val_ratio" in rows[-1]
            ctrl = LearnedController(os.path.join(d, f"{mode}_best.pt"))
            r = run_episode(random_euclidean(8, 1), ctrl, Budget(3, 8, 50))
            assert r.method in ("GRAL", "scalar-PPO")


if __name__ == "__main__":
    print("torch", torch.__version__)
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
