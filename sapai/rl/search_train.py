"""
Train a SearchAgent (value network plus search, see sapai.rl.search) with
expert iteration.

    python -m sapai.rl.search_train --policy models/token_11.8M.zip \\
        --opponent-pool runs/token8/selfplay_teams.pkl --out runs/search1

1. Bootstrap: the starting policy plays games on its own and a value network
   is fitted to the reward each end of turn position went on to get.
2. Each iteration the search agent plays games against the opponent library
   (recent self-play teams and some greedy bot teams, plus the agent's own
   teams), the value network is refitted on a replay buffer, and every
   --distill-every iterations the policy is trained to pick the actions the
   search picked, so the policy proposes better candidates.
3. Every --eval-every games the agent plays the live greedy bot. The agent is
   saved to latest_agent/ (and best_agent/), with progress.csv, so the
   benchmark watcher and resume_all can follow the run.

The value target is the undiscounted reward for the rest of the game: +1 per
battle won, -1 per battle lost and +5 for winning the game.
"""

import argparse
import collections
import csv
import json
import os
import random
import shutil
import time

import numpy as np
import torch as th

from sapai.battle import Battle
from sapai.rl.env import END_TURN, SAPEnv
from sapai.rl.evaluate import play_games
from sapai.rl.opponents import LibraryOpponent, ModelOpponent, TeamLibrary
from sapai.teams import Team
from sapai.rl.search import VALUE_SCALE, Context, SearchAgent, ValueNet


def expected_battle(team_state, turn, library, rng, k):
    """Mean battle reward of a team against k opponents from the library"""
    total = 0
    for _ in range(k):
        result = Battle(Team.from_state(team_state), library.sample(turn, rng)).battle()
        total += {0: 1, 1: -1, 2: 0}[result]
    return total / k


def play_training_game(
    env,
    agent=None,
    policy=None,
    record_decisions=None,
    library=None,
    rng=None,
    battle_samples=0,
    game_win_bonus=5.0,
):
    """
    Play one game. Returns [(end of turn obs, reward for the rest of the
    game)], the teams the player battled with each turn, and the final info.

    With battle_samples, each turn's battle reward is replaced by the mean
    over that many battles against opponents from the library, which takes
    most of the luck out of the value targets.
    """
    obs, _ = env.reset()
    end_obs, rewards, teams = [], [], {}
    done = False
    while not done:
        if agent is not None:
            agent.shop_env(env, record_decisions)
            action = END_TURN
        elif env.actions_this_turn >= env.max_actions_per_turn - 1:
            ### End the turn here, so its battle reward is counted for it
            action = END_TURN
        else:
            action, _ = policy.predict(
                obs, action_masks=env.action_masks(), deterministic=False
            )
        if int(action) == END_TURN:
            ctx = Context(
                env.actions_this_turn, env.last_result, env.wins, env.max_turns
            )
            end_obs.append(ctx.encode(env.player))
            turn = env.player.turn
            obs, reward, terminated, truncated, info = env.step(END_TURN)
            ### The team that battled this turn
            teams.setdefault(turn, []).append(env.last_team_state)
            if battle_samples:
                reward = expected_battle(
                    env.last_team_state, turn, library, rng, battle_samples
                )
            else:
                ### Leave the game win bonus to the end, as with battle_samples
                reward = {0: 1.0, 1: -1.0, 2: 0.0}[info["result"]]
            rewards.append(reward)
        else:
            obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    returns = np.cumsum(rewards[::-1])[::-1]
    if info.get("game_won"):
        returns = returns + game_win_bonus
    return list(zip(end_obs, returns)), teams, info


class Trainer:
    def __init__(self, args):
        self.args = args
        self.out = args.out
        os.makedirs(self.out, exist_ok=True)
        self.rng = random.Random(args.seed)
        np.random.seed(args.seed)
        th.manual_seed(args.seed)
        self.value_buffer = collections.deque(maxlen=args.buffer)
        self.decision_buffer = collections.deque(maxlen=args.buffer)
        self.games = 0
        self.best = -np.inf
        self.distills = 0
        self.last_score = 0.0
        self.start = time.time()

        ### Opponents: greedy bot teams thinned to --keep-bot-teams, recent
        ###   self-play teams and the agent's own teams, capped to
        ###   --max-opponent-teams by removing the oldest self-play teams
        library = TeamLibrary.from_bots(args.library_games, bots=("greedy",))
        library.prune("bot", args.keep_bot_teams / max(len(library), 1), self.rng)
        for path in args.opponent_pool:
            if os.path.exists(path):
                library.add_library(TeamLibrary.load(path), "self")
        self.own_teams = TeamLibrary()
        own_path = os.path.join(self.out, "own_teams.pkl")
        if args.resume and os.path.exists(os.path.join(args.resume, "own_teams.pkl")):
            self.own_teams = TeamLibrary.load(
                os.path.join(args.resume, "own_teams.pkl")
            )
            library.add_library(self.own_teams, "self")
        self.own_path = own_path
        library.cap(args.max_opponent_teams)
        self.library = library
        print(
            f"Opponent library: {library.count('bot')} bot teams,"
            + f" {library.count('self')} self-play teams",
            flush=True,
        )
        self.env = SAPEnv(LibraryOpponent(library))
        self.env.record_last_team = True

        if args.resume:
            self.agent = SearchAgent.load(os.path.join(args.resume, "latest_agent"))
            progress = os.path.join(args.resume, "progress.csv")
            if os.path.exists(progress):
                with open(progress) as f:
                    rows = list(csv.DictReader(f))
                if rows:
                    self.games = int(rows[-1]["timesteps"])
                    self.best = max(float(r["score_vs_start"]) for r in rows)
                    self.distills = int(rows[-1]["distills"])
                    self.last_score = float(rows[-1]["score_vs_start"])
        else:
            from sb3_contrib import MaskablePPO

            policy = MaskablePPO.load(args.policy, device="cpu")
            value_net = ValueNet(args.d_model, args.n_heads, args.n_layers)
            self.agent = SearchAgent(
                policy,
                value_net,
                n_candidates=args.n_candidates,
                rollout_depth=args.rollout_depth,
                roll_samples=args.roll_samples,
                prior_weight=args.prior_weight,
            )
        self.value_opt = th.optim.AdamW(
            self.agent.value_net.parameters(),
            lr=args.value_lr,
            weight_decay=args.value_weight_decay,
        )
        self.policy_opt = th.optim.Adam(
            self.agent.policy_model.policy.parameters(), lr=args.distill_lr
        )
        from sb3_contrib import MaskablePPO

        ### Frozen copy of the starting policy, the yardstick for evaluation
        self.base_policy = MaskablePPO.load(args.policy, device="cpu")
        self.csv_path = os.path.join(self.out, "progress.csv")
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="") as f:
                csv.writer(f).writerow(
                    [
                        "timesteps",
                        "minutes",
                        "value_loss",
                        "distill_loss",
                        "score_vs_start",
                        "distills",
                    ]
                )
        self.value_loss = float("nan")
        self.distill_loss = float("nan")

    def target_kwargs(self):
        return dict(
            library=self.library,
            rng=self.rng,
            battle_samples=self.args.battle_samples,
        )

    ############################################################################
    #### Training steps
    ############################################################################

    def train_value(self, steps):
        if len(self.value_buffer) < 64:
            return
        net = self.agent.value_net
        net.train()
        losses = []
        for _ in range(steps):
            batch = self.rng.sample(
                self.value_buffer, min(self.args.batch_size, len(self.value_buffer))
            )
            obs = th.as_tensor(np.stack([b[0] for b in batch]))
            target = th.as_tensor(
                np.array([b[1] for b in batch], dtype=np.float32) / VALUE_SCALE
            )
            loss = th.nn.functional.mse_loss(net(obs), target)
            self.value_opt.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            self.value_opt.step()
            losses.append(loss.item())
        net.eval()
        self.value_loss = float(np.mean(losses))

    def distill(self, steps):
        """Train the policy to pick the actions the search picked"""
        if len(self.decision_buffer) < 256:
            return
        policy = self.agent.policy_model.policy
        policy.set_training_mode(True)
        losses = []
        for _ in range(steps):
            batch = self.rng.sample(
                self.decision_buffer,
                min(self.args.batch_size, len(self.decision_buffer)),
            )
            obs = th.as_tensor(np.stack([b[0] for b in batch]))
            masks = np.stack([b[1] for b in batch])
            actions = th.as_tensor(np.array([b[2] for b in batch]))
            _, log_prob, _ = policy.evaluate_actions(obs, actions, action_masks=masks)
            loss = -log_prob.mean()
            self.policy_opt.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            self.policy_opt.step()
            losses.append(loss.item())
        policy.set_training_mode(False)
        self.distill_loss = float(np.mean(losses))

    def add_game(self, samples, teams):
        self.value_buffer.extend((o.astype(np.float32), float(r)) for o, r in samples)
        self.own_teams.add_all(teams, "self", age=float(self.games))
        self.library.add_all(teams, "self", age=float(self.games))
        self.library.cap(self.args.max_opponent_teams)
        self.own_teams.cap(self.args.max_opponent_teams - self.library.count("bot"))
        self.games += 1

    ############################################################################
    #### Loop
    ############################################################################

    def bootstrap(self):
        """Fit a first value network on games of the policy on its own"""
        print(f"Bootstrapping on {self.args.bootstrap_games} policy games", flush=True)
        for _ in range(self.args.bootstrap_games):
            samples, teams, _ = play_training_game(
                self.env, policy=self.agent.policy_model, **self.target_kwargs()
            )
            self.value_buffer.extend(
                (o.astype(np.float32), float(r)) for o, r in samples
            )
        self.train_value(self.args.bootstrap_steps)
        print(f"Bootstrap value loss {self.value_loss:.4f}", flush=True)

    def iteration(self, i):
        self.agent.temperature = self.args.temperature
        for _ in range(self.args.games_per_iter):
            decisions = []
            samples, teams, _ = play_training_game(
                self.env,
                agent=self.agent,
                record_decisions=decisions,
                **self.target_kwargs(),
            )
            self.add_game(samples, teams)
            ### Only decisions where the search had a real choice teach the policy
            self.decision_buffer.extend(
                (o.astype(np.float32), m, a) for o, m, a in decisions if m.sum() > 1
            )
        self.agent.temperature = 0.0
        self.train_value(self.args.value_steps)
        ### Only teach the policy the search's choices once the search is
        ###   measurably stronger than the policy it started from
        if (
            self.args.distill_every
            and (i + 1) % self.args.distill_every == 0
            and self.last_score >= self.args.distill_min_score
        ):
            self.distill(self.args.distill_steps)
            self.distills += 1
        self.own_teams.save(self.own_path)

    def evaluate(self):
        """
        Head-to-head score against the frozen starting policy over play-out
        games (battles won plus half of draws). Above 0.5 means the search
        agent is stronger than the policy it started from.
        """
        r = play_games(
            self.agent,
            self.args.eval_games,
            seed=1_000_000,
            env_kwargs={"play_out": True, "max_turns": self.args.eval_turns},
            opponent=ModelOpponent(self.base_policy),
        )
        score = r["battle_win_rate"] + 0.5 * r["battle_draw_rate"]
        self.last_score = score
        minutes = (time.time() - self.start) / 60
        with open(self.csv_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [
                    self.games,
                    round(minutes, 1),
                    round(self.value_loss, 4),
                    round(self.distill_loss, 4),
                    round(score, 4),
                    self.distills,
                ]
            )
        ### Write to a temporary folder then swap, so the benchmark watcher
        ###   never copies a half written agent
        tmp = os.path.join(self.out, "latest_agent.tmp")
        self.agent.save(tmp)
        latest = os.path.join(self.out, "latest_agent")
        if os.path.exists(latest):
            shutil.rmtree(latest)
        os.rename(tmp, latest)
        if score > self.best:
            self.best = score
            self.agent.save(os.path.join(self.out, "best_agent"))
        print(
            f"[{self.games} games, {minutes:.0f} min, value loss"
            + f" {self.value_loss:.4f}, distill loss {self.distill_loss:.3f},"
            + f" {self.distills} distills] score vs starting policy {score:.3f}",
            flush=True,
        )

    def run(self):
        if not self.args.resume and self.args.bootstrap_games:
            self.bootstrap()
            self.evaluate()
        i = 0
        next_eval = self.games + self.args.eval_every
        while self.games < self.args.total_games:
            self.iteration(i)
            i += 1
            if self.games >= next_eval:
                next_eval = self.games + self.args.eval_every
                self.evaluate()
        self.evaluate()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    parser.add_argument("--policy", default="models/token_11.8M.zip")
    parser.add_argument("--resume", help="run folder to continue")
    parser.add_argument("--out", default="runs/search1")
    parser.add_argument(
        "--opponent-pool",
        nargs="*",
        default=["runs/token8/selfplay_teams.pkl"],
        help="self-play team pools to use as opponents",
    )
    parser.add_argument("--library-games", type=int, default=400)
    parser.add_argument("--keep-bot-teams", type=int, default=2000)
    parser.add_argument("--max-opponent-teams", type=int, default=15000)
    parser.add_argument("--total-games", type=int, default=10**9)
    parser.add_argument("--bootstrap-games", type=int, default=1500)
    parser.add_argument("--bootstrap-steps", type=int, default=3000)
    parser.add_argument("--games-per-iter", type=int, default=20)
    parser.add_argument("--value-steps", type=int, default=200)
    parser.add_argument("--distill-every", type=int, default=5)
    parser.add_argument("--distill-steps", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=300)
    parser.add_argument("--eval-games", type=int, default=60)
    parser.add_argument("--buffer", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--value-lr", type=float, default=3e-4)
    parser.add_argument("--value-weight-decay", type=float, default=1e-2)
    parser.add_argument(
        "--battle-samples",
        type=int,
        default=8,
        help="battles per turn averaged into the value targets (0 uses the"
        + " actual battle)",
    )
    parser.add_argument("--eval-turns", type=int, default=15)
    parser.add_argument(
        "--distill-min-score",
        type=float,
        default=0.52,
        help="only distill when the score against the starting policy is"
        + " at least this",
    )
    parser.add_argument("--distill-lr", type=float, default=3e-5)
    parser.add_argument("--temperature", type=float, default=0.03)
    parser.add_argument("--n-candidates", type=int, default=6)
    parser.add_argument("--rollout-depth", type=int, default=6)
    parser.add_argument("--roll-samples", type=int, default=2)
    parser.add_argument(
        "--prior-weight",
        type=float,
        default=3.0,
        help="score = value + prior_weight * log(policy probability)",
    )
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args(argv)

    th.set_num_threads(args.threads)
    os.makedirs(args.out, exist_ok=True)
    saved = dict(vars(args), _module="sapai.rl.search_train")
    saved.pop("resume")
    with open(os.path.join(args.out, "args.json"), "w") as f:
        json.dump(saved, f, indent=2)
    Trainer(args).run()


if __name__ == "__main__":
    main()
