"""Train a learned penalty controller with PPO.

    python scripts/train.py --mode gral              # proposed method (GNN, per constraint)
    python scripts/train.py --mode scalar            # learned scalar-λ baseline / ablation
    python scripts/train.py --mode gral --quick      # 3-iteration smoke test

Outputs in models/: <mode>_best.pt (best validation), <mode>_last.pt, <mode>_train_log.csv
Training uses only TRAIN-split random instances (8–14 cities by default);
validation uses a separate fixed set; evaluation (scripts/evaluate.py) uses TEST
instances and larger sizes never seen in training.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def main():
    from gral.rl.ppo import PPOConfig, train
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["gral", "scalar"], default="gral")
    ap.add_argument("--iters", type=int, default=150)
    ap.add_argument("--episodes", type=int, default=64)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--sizes", type=int, nargs="+", default=[8, 9, 10, 11, 12, 13, 14])
    ap.add_argument("--no-residual", action="store_true", help="v1 ablation: no residual, no pairing")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="models")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    cfg = PPOConfig(mode=a.mode, iters=a.iters, episodes=a.episodes, workers=a.workers,
                    sizes=tuple(a.sizes), lr=a.lr, seed=a.seed, out=a.out,
                    residual=not a.no_residual, paired=not a.no_residual)
    if a.quick:
        cfg.iters, cfg.episodes, cfg.val_every, cfg.val_count = 3, max(4, a.workers), 3, 1
        cfg.out = os.path.join(a.out, "quick")
    print(f"training {cfg.mode} with {cfg.workers} workers, {cfg.iters} iterations x "
          f"{cfg.episodes} episodes, sizes {cfg.sizes}")
    train(cfg)


if __name__ == "__main__":
    main()
