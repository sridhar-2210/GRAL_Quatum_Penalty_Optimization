"""NumPy-only checks for the RL environment.  Run: python tests/test_env.py"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gral import (Budget, random_euclidean, run_episode, FixedPenalty,  # noqa: E402
                  PerConstraintRule, ScalarAdaptive)
from gral.rl import PenaltyEnv, features as F  # noqa: E402


def _zero(env, mode):
    return np.zeros((env.enc.n_constraints, 2)) if mode == "gral" else np.zeros(1)


def test_shapes_and_episode_length():
    for mode in ("gral", "scalar"):
        env = PenaltyEnv(mode, sizes=(5, 7), budget=Budget(4, 8, 50), seed=1)
        obs = env.reset()
        C = env.enc.n_constraints
        assert obs["node"].shape == (C, F.F_NODE)
        assert obs["glob"].shape == (F.F_GLOB,)
        assert obs["adj"].shape == (C, C)
        assert np.allclose(obs["adj"].sum(1), 1.0)
        steps, done = 0, False
        while not done:
            obs, r, done, info = env.step(_zero(env, mode))
            steps += 1
            assert np.isfinite(r)
        assert steps == 3          # K rounds total = reset round + K-1 steps


def test_return_telescopes_to_improvement_over_rule():
    env = PenaltyEnv("gral", sizes=(6,), budget=Budget(5, 16, 100), seed=2)
    env.reset()
    total = env.first_reward()
    done, rng = False, np.random.default_rng(0)
    while not done:
        _, r, done, info = env.step(rng.uniform(-1, 1, (env.enc.n_constraints, 2)))
        total += r
    assert np.isclose(total, F.REWARD_SCALE * (info["base_ratio"] - info["ratio"]))


def test_zero_action_reproduces_rule_and_zero_reward():
    """Residual design: action 0 == the hand-designed rule, so paired reward is 0."""
    inst = random_euclidean(8, 3)
    b = Budget(5, 16, 100)
    for mode, rule in (("gral", PerConstraintRule()), ("scalar", ScalarAdaptive())):
        env = PenaltyEnv(mode, budget=b)
        env.reset(inst=inst, seed=123)
        rewards, done = [env.first_reward()], False
        while not done:
            _, r, done, _ = env.step(_zero(env, mode))
            rewards.append(r)
        ref = run_episode(inst, rule, b, seed=123)
        assert np.isclose(env.tracker.best_len * env.enc.scale, ref.best_length)
        assert np.allclose(rewards, 0.0)


def test_no_residual_zero_action_is_fixed_baseline():
    inst = random_euclidean(7, 3)
    b = Budget(4, 16, 100)
    env = PenaltyEnv("gral", budget=b, residual=False, paired=False)
    env.reset(inst=inst, seed=123)
    done = False
    while not done:
        _, _, done, _ = env.step(np.zeros((env.enc.n_constraints, 2)))
    r = run_episode(inst, FixedPenalty(1.0), b, seed=123)
    assert np.isclose(env.tracker.best_len * env.enc.scale, r.best_length)


def test_action_bounds():
    class S:
        viol_freq = np.ones(4)
        feas_frac = 0.0
    w, mu = np.ones(4), np.zeros(4)
    for _ in range(50):
        w, mu = F.apply_action(w, mu, np.full((4, 2), 5.0), "gral", S)
    assert w.max() <= 50 + 1e-9 and mu.max() <= 50 + 1e-9
    w2, mu2 = F.apply_action(np.ones(4), np.zeros(4), [-1.0], "scalar", S, residual=False)
    assert np.allclose(w2, np.exp(-F.DLOGW)) and np.allclose(mu2, 0)


def test_training_seeds_disjoint_from_test_and_validation():
    from gral.rl.env import TRAIN_SEED_LO
    from gral.rl.ppo import VAL_SEED0
    assert TRAIN_SEED_LO > 9_000_000 + 1000 * 20 and TRAIN_SEED_LO > VAL_SEED0 + 1000 * 20


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
