"""
Train a MaskablePPO agent on sapai's 1v1 matches.

    python -m sapai.rl.train --policy token --timesteps 2000000 --out runs/token
    python -m sapai.rl.train --resume models/token_22.5M.zip --out runs/token1v1

Every match is against one opponent for the whole match: the greedy bot, or
(self-play) one of the agent's own earlier versions. Every --pool-every steps
the current model is saved into the run's pool/ folder; each match picks the
greedy bot with probability --greedy-fraction, or otherwise a model from the
pool, newer ones more often. --pool-models adds other saved models to the pool
at the start. Only the newest --pool-size models are kept.

Every --eval-every steps the agent plays games against the greedy bot; the
latest model is saved as latest_model.zip, the best one against the greedy
bot as best_model.zip, and progress is written to progress.csv. Use
sapai.rl.benchmark to rate models against each other.

A model trained before matches were 1v1 is upgraded automatically when it is
resumed (see sapai.rl.grow --upgrade).
"""

import argparse
import csv
import json
import os
import shutil
import time

import numpy as np
import torch as th
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

from sapai.rl.env import SAPEnv
from sapai.rl.evaluate import format_results, play_games
from sapai.rl.opponents import PoolOpponent
from sapai.rl.policy import TokenMaskablePolicy


def make_env(pool_dir, greedy_fraction, seed, env_kwargs):
    def init():
        env = SAPEnv(PoolOpponent(pool_dir, greedy_fraction), **env_kwargs)
        env.reset(seed=seed)
        return env

    return init


class EvalCallback(BaseCallback):
    """
    Evaluates against the greedy bot, saves the best and latest models, logs
    progress, and saves the model into the self-play pool every pool_every
    steps (keeping the newest pool_size models).
    """

    def __init__(self, out, eval_every, eval_games, pool_every, pool_size, verbose=1):
        super().__init__(verbose)
        self.out = out
        self.pool_dir = os.path.join(out, "pool")
        self.eval_every = eval_every
        self.eval_games = eval_games
        self.pool_every = pool_every
        self.pool_size = pool_size
        self.best = -np.inf
        self.start = time.time()
        self.csv_path = os.path.join(out, "progress.csv")
        self.episode_rewards = []

    def _on_training_start(self):
        ### Count from the current step, which is not 0 when resuming
        self.next_eval = self.num_timesteps + self.eval_every
        self.next_pool = self.num_timesteps + self.pool_every
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="") as f:
                csv.writer(f).writerow(
                    [
                        "timesteps",
                        "minutes",
                        "train_mean_reward",
                        "game_win_rate",
                        "battle_win_rate",
                        "battle_loss_rate",
                        "mean_wins",
                        "mean_turns",
                        "pool_size",
                    ]
                )

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
        if self.pool_every and self.num_timesteps >= self.next_pool:
            self.next_pool = self.num_timesteps + self.pool_every
            self.add_to_pool()
        if self.num_timesteps >= self.next_eval:
            self.next_eval += self.eval_every
            self.evaluate()
        return True

    def pool(self):
        return sorted(
            (
                os.path.join(self.pool_dir, x)
                for x in os.listdir(self.pool_dir)
                if x.endswith(".zip")
            ),
            key=os.path.getmtime,
        )

    def add_to_pool(self):
        """Save the current model into the pool; drop the oldest beyond pool_size"""
        os.makedirs(self.pool_dir, exist_ok=True)
        ### Written under another name first so opponents never load half a file
        tmp = os.path.join(self.out, "pool_saving.zip")
        self.model.save(tmp)
        os.replace(tmp, os.path.join(self.pool_dir, f"step_{self.num_timesteps}.zip"))
        pool = self.pool()
        for old in pool[: max(0, len(pool) - self.pool_size)]:
            os.remove(old)
        if self.verbose:
            print(
                f"[{self.num_timesteps} steps] saved to the self-play pool"
                + f" ({min(len(pool), self.pool_size)} models)",
                flush=True,
            )

    def evaluate(self):
        r = play_games(self.model, self.eval_games, seed=1_000_000)
        train_reward = (
            float(np.mean(self.episode_rewards[-200:]))
            if self.episode_rewards
            else float("nan")
        )
        minutes = (time.time() - self.start) / 60
        pool_size = len(self.pool()) if os.path.isdir(self.pool_dir) else 0
        with open(self.csv_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [
                    self.num_timesteps,
                    round(minutes, 1),
                    round(train_reward, 3),
                    r["game_win_rate"],
                    round(r["battle_win_rate"], 4),
                    round(r["battle_loss_rate"], 4),
                    r["mean_wins"],
                    r["mean_turns"],
                    pool_size,
                ]
            )
        self.model.save(os.path.join(self.out, "latest_model"))
        ### Score against the greedy bot: battle win rate minus loss rate
        score = r["battle_win_rate"] - r["battle_loss_rate"]
        if score > self.best:
            self.best = score
            self.model.save(os.path.join(self.out, "best_model"))
        if self.verbose:
            print(
                f"[{self.num_timesteps} steps, {minutes:.0f} min,"
                + f" train reward {train_reward:.2f}] "
                + format_results("vs greedy", r),
                flush=True,
            )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    parser.add_argument("--policy", choices=["token", "mlp"], default="token")
    parser.add_argument("--timesteps", type=int, default=2_000_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--resume", help="continue training a saved model")
    parser.add_argument("--out", default="runs/token")
    parser.add_argument("--eval-every", type=int, default=100_000)
    parser.add_argument("--eval-games", type=int, default=200)
    parser.add_argument(
        "--pool-every",
        type=int,
        default=200_000,
        help="save the model into the self-play pool every N steps (0: never)",
    )
    parser.add_argument(
        "--pool-size", type=int, default=20, help="newest pool models to keep"
    )
    parser.add_argument(
        "--pool-models",
        nargs="*",
        default=[],
        help="saved models to put in the self-play pool at the start",
    )
    parser.add_argument(
        "--greedy-fraction",
        type=float,
        default=0.2,
        help="share of matches against the greedy bot (all of them while the"
        + " pool is empty)",
    )
    parser.add_argument(
        "--wins-needed",
        type=int,
        default=None,
        help="wins that end the match (default: none, only lives)",
    )
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="default 2e-4 for token, 3e-4 for mlp",
    )
    parser.add_argument("--target-kl", type=float, default=None)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args(argv)

    if args.learning_rate is None:
        args.learning_rate = 2e-4 if args.policy == "token" else 3e-4
    os.makedirs(args.out, exist_ok=True)
    th.set_num_threads(args.threads)
    ### Settings of the run, used by sapai.rl.resume_all to continue it
    with open(os.path.join(args.out, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    pool_dir = os.path.join(args.out, "pool")
    os.makedirs(pool_dir, exist_ok=True)
    for path in args.pool_models:
        ### copy2 keeps the modification time, which orders the pool by age
        shutil.copy2(path, os.path.join(pool_dir, os.path.basename(path)))
    print(f"Self-play pool: {len(os.listdir(pool_dir))} models", flush=True)

    env_kwargs = {"wins_needed": args.wins_needed}
    env_fns = [
        make_env(pool_dir, args.greedy_fraction, args.seed * 1000 + i, env_kwargs)
        for i in range(args.n_envs)
    ]
    vec_cls = SubprocVecEnv if args.n_envs > 1 else DummyVecEnv
    vec_env = VecMonitor(vec_cls(env_fns))

    common = dict(
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=5,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=0.95,
        ent_coef=args.ent_coef,
        clip_range=0.2,
        target_kl=args.target_kl,
        seed=args.seed,
        device="cpu",
        verbose=0,
    )
    if args.resume:
        resume = args.resume
        saved = MaskablePPO.load(resume, device="cpu")
        if saved.observation_space.shape != vec_env.observation_space.shape:
            from sapai.rl.grow import upgrade_model

            print("Upgrading the model to see the opponent", flush=True)
            resume = os.path.join(args.out, "upgraded_start.zip")
            upgrade_model(saved).save(resume)
        model = MaskablePPO.load(resume, env=vec_env, device="cpu")
        model.learning_rate = args.learning_rate
        model._setup_lr_schedule()
        model.target_kl = args.target_kl
    elif args.policy == "token":
        model = MaskablePPO(
            TokenMaskablePolicy,
            vec_env,
            policy_kwargs={"d_model": 128, "n_heads": 4, "n_layers": 2},
            **common,
        )
    else:
        model = MaskablePPO(
            "MlpPolicy",
            vec_env,
            policy_kwargs={"net_arch": [256, 256]},
            **common,
        )
    n_params = sum(p.numel() for p in model.policy.parameters())
    print(f"{args.policy} policy with {n_params} parameters", flush=True)

    callback = EvalCallback(
        args.out, args.eval_every, args.eval_games, args.pool_every, args.pool_size
    )
    model.learn(
        total_timesteps=args.timesteps,
        callback=callback,
        reset_num_timesteps=not args.resume,
    )
    model.save(os.path.join(args.out, "final_model"))
    callback.evaluate()
    vec_env.close()


if __name__ == "__main__":
    main()
