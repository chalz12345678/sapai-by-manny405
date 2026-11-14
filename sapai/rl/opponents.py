"""
Opponents for SAPEnv's 1v1 matches.

An opponent is the other side of the match. It shops for its own player
every turn, the same way the agent does:
    reset(rng)            called at the start of every match
    shop(player, view)    take this turn's shop actions for player. view is
                          Match.view() for the opponent's side: its own
                          state, plus the agent's team from the last battle
                          and the agent's lives and wins

GreedyOpponent:  the greedy bot from the terminal game
RandomOpponent:  random legal shop actions
ModelOpponent:   a saved model, for example a frozen earlier version of the
                 agent
PoolOpponent:    for self-play: each match it picks the greedy bot or one of
                 the saved models in a folder (the agent's earlier versions)
"""

import glob
import os
import random

import numpy as np

from sapai.cli import Opponent as _GreedyBot
from sapai.rl.env import END_TURN, action_mask, apply_action, encode_view
from sapai.rl.env import observation_for


class GreedyBot:
    """Greedy shopping bot from the terminal game"""

    def __init__(self, rng=None):
        self.bot = _GreedyBot(rng or random.Random())

    def shop(self, player):
        self.bot.player = player
        self.bot.shop_turn()


class RandomBot:
    """Takes random legal shop actions, then ends the turn"""

    def __init__(self, rng=None, max_actions=15, end_prob=0.15):
        self.rng = rng or random.Random()
        self.max_actions = max_actions
        self.end_prob = end_prob

    def shop(self, player):
        for _ in range(self.max_actions):
            if self.rng.random() < self.end_prob:
                break
            legal = [a for a in np.flatnonzero(action_mask(player)) if a != END_TURN]
            if not legal:
                break
            try:
                apply_action(player, int(self.rng.choice(legal)))
            except Exception:
                pass
        player.team.move_forward()


def _rng_from(rng):
    """Python random.Random from a numpy Generator, random.Random or None"""
    if isinstance(rng, random.Random):
        return rng
    if rng is None:
        return random.Random()
    return random.Random(int(rng.integers(0, 2**31 - 1)))


class GreedyOpponent:
    """The greedy bot as a match opponent"""

    name = "greedy"

    def __init__(self):
        self.bot = GreedyBot()

    def reset(self, rng=None):
        self.bot = GreedyBot(_rng_from(rng))

    def shop(self, player, view=None):
        self.bot.shop(player)


class RandomOpponent:
    """Random legal shop actions as a match opponent"""

    name = "random"

    def __init__(self):
        self.bot = RandomBot()

    def reset(self, rng=None):
        self.bot = RandomBot(_rng_from(rng))

    def shop(self, player, view=None):
        self.bot.shop(player)


class ModelOpponent:
    """
    A saved model as a match opponent. Models trained before matches were
    1v1 do not see the agent's team; they are given the part of the
    observation they know (sapai.rl.env.observation_for).
    """

    def __init__(
        self, model, deterministic=True, max_actions_per_turn=30, max_turns=25
    ):
        self.model = model
        self.deterministic = deterministic
        self.max_actions_per_turn = max_actions_per_turn
        self.max_turns = max_turns

    def reset(self, rng=None):
        pass

    def shop(self, player, view):
        view = dict(view, player=player)
        for actions in range(self.max_actions_per_turn - 1):
            obs = encode_view(view, actions, self.max_turns)
            action, _ = self.model.predict(
                observation_for(self.model, obs),
                action_masks=action_mask(player),
                deterministic=self.deterministic,
            )
            if int(action) == END_TURN:
                break
            try:
                apply_action(player, int(action))
            except Exception:
                pass
        player.team.move_forward()


class PoolOpponent:
    """
    Self-play opponent. Each match it is the greedy bot (with probability
    greedy_fraction) or a random saved model from pool_dir, for example the
    agent's earlier checkpoints. Newer checkpoints are picked more often
    (recent_weight), and loaded models are cached.
    """

    def __init__(
        self,
        pool_dir,
        greedy_fraction=0.2,
        recent_weight=2.0,
        max_cached=8,
        deterministic=False,
    ):
        self.pool_dir = pool_dir
        self.greedy_fraction = greedy_fraction
        self.recent_weight = recent_weight
        self.max_cached = max_cached
        self.deterministic = deterministic
        self.cache = {}
        self.current = GreedyOpponent()
        self.current_name = "greedy"

    def models(self):
        """Saved models in the pool, oldest first"""
        return sorted(
            glob.glob(os.path.join(self.pool_dir, "*.zip")), key=os.path.getmtime
        )

    def load(self, path):
        from sb3_contrib import MaskablePPO

        if path not in self.cache:
            if len(self.cache) >= self.max_cached:
                self.cache.pop(next(iter(self.cache)))
            self.cache[path] = MaskablePPO.load(path, device="cpu")
        return self.cache[path]

    def reset(self, rng=None):
        rng = _rng_from(rng)
        paths = self.models() if os.path.isdir(self.pool_dir) else []
        if not paths or rng.random() < self.greedy_fraction:
            self.current = GreedyOpponent()
            self.current_name = "greedy"
        else:
            ### Weight k ** recent_weight for the k-th oldest model
            weights = [(k + 1) ** self.recent_weight for k in range(len(paths))]
            path = rng.choices(paths, weights=weights)[0]
            try:
                model = self.load(path)
                self.current = ModelOpponent(model, deterministic=self.deterministic)
                self.current_name = os.path.basename(path)
            except Exception:
                ### A checkpoint being written right now cannot be read yet
                self.current = GreedyOpponent()
                self.current_name = "greedy"
        self.current.reset(rng)

    def shop(self, player, view):
        self.current.shop(player, view)
