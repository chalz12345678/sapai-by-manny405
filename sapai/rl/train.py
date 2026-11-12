"""
Train a MaskablePPO agent on sapai.

    python -m sapai.rl.train --policy token --timesteps 2000000 --out runs/token
    python -m sapai.rl.train --policy mlp --timesteps 2000000 --out runs/mlp

Training opponents are teams recorded from greedy and random bot games, sampled
for each turn. Every --eval-every steps the agent plays games against the live
greedy bot; the best model so far is saved as best_model.zip and progress is
written to progress.csv. With --selfplay-every, teams built by the current agent
are added to the opponent library during training; with --selfplay-on-plateau K
as well, self-play only starts once K evaluations in a row bring no new best
against the greedy bot:

    python -m sapai.rl.train --policy mlp --resume runs/mlp/best_model.zip \
        --library-bots greedy --selfplay-every 200000 --selfplay-on-plateau 5
"""

import argparse
import csv
import os
import time

import numpy as np
import torch as th
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

from sapai.rl.env import SAPEnv
from sapai.rl.evaluate import format_results, play_games
from sapai.rl.opponents import LibraryOpponent, TeamLibrary, harvest_teams
from sapai.rl.policy import TokenMaskablePolicy


def make_env(library, seed, env_kwargs):
    def init():
        env = SAPEnv(LibraryOpponent(library), **env_kwargs)
        env.reset(seed=seed)
        return env

    return init


class EvalCallback(BaseCallback):
    """
    Evaluates against the live greedy bot, saves the best and latest models
    and logs progress. Also runs self-play: every selfplay_every steps, teams
    built by the current model are added to the opponent library. With
    selfplay_on_plateau=K, self-play only starts once K evaluations in a row
    have not beaten the best result against the greedy bot.
    """

    def __init__(
        self,
        out,
        eval_every,
        eval_games,
        selfplay_every=0,
        selfplay_on_plateau=0,
        selfplay_games=50,
        selfplay_initial_games=None,
        bot_teams=0,
        self_teams=0,
        prune_bots_at=None,
        keep_bot_fraction=0.2,
        verbose=1,
    ):
        super().__init__(verbose)
        self.out = out
        self.eval_every = eval_every
        self.eval_games = eval_games
        self.selfplay_every = selfplay_every
        self.selfplay_on_plateau = selfplay_on_plateau
        self.selfplay_games = selfplay_games
        self.selfplay_initial_games = selfplay_initial_games or selfplay_games
        self.harvests = 0
        ### Self-play teams harvested so far (also saved to disk) and when to
        ###   thin out the bot teams in the opponent library
        self.selfplay_library = TeamLibrary()
        self.bot_teams = bot_teams
        self.self_teams = self_teams
        self.prune_bots_at = bot_teams if prune_bots_at is None else prune_bots_at
        self.keep_bot_fraction = keep_bot_fraction
        self.bots_pruned = False
        self.selfplay_active = bool(selfplay_every) and not selfplay_on_plateau
        self.next_eval = eval_every
        self.next_selfplay = selfplay_every
        self.best = -np.inf
        self.evals_since_best = 0
        self.start = time.time()
        self.csv_path = os.path.join(out, "progress.csv")
        self.episode_rewards = []

    def _on_training_start(self):
        ### Count from the current step, which is not 0 when resuming
        self.next_eval = self.num_timesteps + self.eval_every
        self.next_selfplay = self.num_timesteps + self.selfplay_every
        if self.selfplay_active:
            ### Self-play from the start: add the model's teams right away
            self.next_selfplay = self.num_timesteps
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
                    "selfplay",
                ]
            )
        ### A self-play pool loaded from an earlier run may already be big
        ###   enough to thin out the bot teams
        self.maybe_prune_bots()

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
        if self.selfplay_active and self.num_timesteps >= self.next_selfplay:
            self.next_selfplay = self.num_timesteps + self.selfplay_every
            self.add_selfplay_teams()
        if self.num_timesteps >= self.next_eval:
            self.next_eval += self.eval_every
            self.evaluate()
        return True

    def add_selfplay_teams(self):
        n_games = self.selfplay_initial_games if self.harvests == 0 else None
        n_games = n_games or self.selfplay_games
        self.harvests += 1
        teams = harvest_teams(self.model, SAPEnv(), n_games=n_games)
        self.training_env.env_method("add_opponent_teams", teams, "self")
        n = sum(len(x) for x in teams.values())
        self.self_teams += n
        self.selfplay_library.add_all(teams, "self")
        self.selfplay_library.save(os.path.join(self.out, "selfplay_teams.pkl"))
        if self.verbose:
            print(
                f"[{self.num_timesteps} steps] self-play: added {n} teams"
                + f" ({self.self_teams} self-play teams in total)",
                flush=True,
            )
        self.maybe_prune_bots()

    def maybe_prune_bots(self):
        """Once there are enough self-play teams, remove most bot teams"""
        if self.bots_pruned or not self.bot_teams or self.keep_bot_fraction >= 1:
            return
        if self.self_teams < self.prune_bots_at:
            return
        self.bots_pruned = True
        left = self.training_env.env_method(
            "prune_opponent_teams", "bot", self.keep_bot_fraction
        )
        if self.verbose:
            print(
                f"[{self.num_timesteps} steps] self-play pool reached"
                + f" {self.self_teams} teams: kept {self.keep_bot_fraction:.0%}"
                + f" of the {self.bot_teams} bot teams ({left[0]} left)",
                flush=True,
            )

    def start_selfplay(self):
        self.selfplay_active = True
        ### Frozen copy of the model at the start of self-play, useful as a
        ###   fixed benchmark later
        self.model.save(os.path.join(self.out, "selfplay_start"))
        if self.verbose:
            print(
                f"[{self.num_timesteps} steps] no new best against the greedy bot"
                + f" in {self.evals_since_best} evaluations: starting self-play",
                flush=True,
            )
        self.add_selfplay_teams()
        self.next_selfplay = self.num_timesteps + self.selfplay_every

    def evaluate(self):
        r = play_games(self.model, self.eval_games, seed=1_000_000)
        train_reward = (
            float(np.mean(self.episode_rewards[-200:]))
            if self.episode_rewards
            else float("nan")
        )
        minutes = (time.time() - self.start) / 60
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
                    int(self.selfplay_active),
                ]
            )
        self.model.save(os.path.join(self.out, "latest_model"))
        ### Score: mean wins against the greedy bot, the main goal
        if r["mean_wins"] > self.best:
            self.best = r["mean_wins"]
            self.evals_since_best = 0
            self.model.save(os.path.join(self.out, "best_model"))
        else:
            self.evals_since_best += 1
        if self.verbose:
            print(
                f"[{self.num_timesteps} steps, {minutes:.0f} min,"
                + f" train reward {train_reward:.2f}] "
                + format_results("vs greedy", r),
                flush=True,
            )
        if (
            self.selfplay_every
            and self.selfplay_on_plateau
            and not self.selfplay_active
            and self.evals_since_best >= self.selfplay_on_plateau
        ):
            self.start_selfplay()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    parser.add_argument("--policy", choices=["token", "mlp"], default="token")
    parser.add_argument("--timesteps", type=int, default=2_000_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--library-games", type=int, default=400)
    parser.add_argument(
        "--library-bots",
        default="greedy,random",
        help="bots whose teams make up the opponent library",
    )
    parser.add_argument("--resume", help="continue training a saved model")
    parser.add_argument("--out", default="runs/token")
    parser.add_argument("--eval-every", type=int, default=100_000)
    parser.add_argument("--eval-games", type=int, default=200)
    parser.add_argument(
        "--selfplay-every",
        type=int,
        default=0,
        help="add teams from the current model to the library every N steps",
    )
    parser.add_argument(
        "--selfplay-on-plateau",
        type=int,
        default=0,
        help="only start self-play after K evaluations without a new best",
    )
    parser.add_argument("--selfplay-games", type=int, default=50)
    parser.add_argument(
        "--selfplay-initial-games",
        type=int,
        default=None,
        help="games for the first harvest, to build a self-play pool quickly",
    )
    parser.add_argument(
        "--selfplay-teams",
        help="selfplay_teams.pkl from an earlier run to start the pool with",
    )
    parser.add_argument(
        "--prune-bots-at",
        type=int,
        default=None,
        help="self-play teams needed before most bot teams are removed"
        + " (default: as many as there are bot teams)",
    )
    parser.add_argument(
        "--keep-bot-fraction",
        type=float,
        default=0.2,
        help="fraction of bot teams kept after pruning (1 keeps all)",
    )
    parser.add_argument("--wins-needed", type=int, default=10)
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

    library_path = os.path.join(args.out, "library.pkl")
    library = TeamLibrary.from_bots(
        args.library_games, bots=args.library_bots.split(","), seed=args.seed
    )
    library.save(library_path)
    bot_teams = len(library)
    self_teams = 0
    if args.selfplay_teams:
        earlier = TeamLibrary.load(args.selfplay_teams)
        library.add_all(earlier.teams, "self")
        self_teams = len(earlier)
    print(
        f"Opponent library: {bot_teams} bot teams, {self_teams} self-play teams",
        flush=True,
    )

    env_kwargs = {"wins_needed": args.wins_needed}
    env_fns = [
        make_env(library, args.seed * 1000 + i, env_kwargs) for i in range(args.n_envs)
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
        model = MaskablePPO.load(args.resume, env=vec_env, device="cpu")
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
        args.out,
        args.eval_every,
        args.eval_games,
        args.selfplay_every,
        args.selfplay_on_plateau,
        args.selfplay_games,
        selfplay_initial_games=args.selfplay_initial_games,
        bot_teams=bot_teams,
        self_teams=self_teams,
        prune_bots_at=args.prune_bots_at,
        keep_bot_fraction=args.keep_bot_fraction,
    )
    if args.selfplay_teams:
        callback.selfplay_library.add_all(earlier.teams, "self")
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
