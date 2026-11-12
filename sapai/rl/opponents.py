"""
Opponents for SAPEnv.

An opponent provides the team the agent battles each turn through
    reset(rng)              called at the start of every game
    team_for_turn(turn)     returns the opponent Team for this turn's battle
    after_battle(result)    called with the battle result (0 agent won,
                            1 agent lost, 2 draw)

GreedyOpponent:  a live greedy bot that shops every turn alongside the agent
LibraryOpponent: samples a recorded team for the turn from a TeamLibrary,
                 similar to how the real game matches you against other
                 players' recorded teams

"""

import pickle
import random

import numpy as np

from sapai.cli import Opponent as _GreedyBot
from sapai.player import Player
from sapai.rl.env import END_TURN, action_mask, apply_action
from sapai.teams import Team


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
    """A live greedy bot that shops each turn, like the terminal game"""

    def __init__(self):
        self.bot = None

    def reset(self, rng=None):
        self.bot = _GreedyBot(_rng_from(rng))

    def team_for_turn(self, turn):
        self.bot.shop_turn()
        self.bot.player.end_turn()
        return self.bot.player.team

    def after_battle(self, result):
        self.bot.player.start_turn(winner=(result != 0))


class TeamLibrary:
    """Recorded teams for each turn, stored as Team states"""

    def __init__(self, teams=None):
        self.teams = teams or {}

    def add(self, turn, team_state):
        self.teams.setdefault(int(turn), []).append(team_state)

    def add_all(self, teams):
        for turn, states in teams.items():
            for state in states:
                self.add(turn, state)

    def sample(self, turn, rng):
        turns = sorted(self.teams)
        if not turns:
            raise ValueError("TeamLibrary is empty")
        ### Use the closest recorded turn at or before this turn
        avail = [t for t in turns if t <= turn] or turns[:1]
        states = self.teams[avail[-1]]
        state = states[rng.randrange(len(states))]
        return Team.from_state(state)

    def __len__(self):
        return sum(len(x) for x in self.teams.values())

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.teams, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            return cls(pickle.load(f))

    @classmethod
    def from_bots(cls, n_games, bots=("greedy", "random"), turns=25, seed=0):
        """Record the teams that bots build over n_games games each"""
        library = cls()
        rng = random.Random(seed)
        np.random.seed(seed)
        for game in range(n_games):
            for kind in bots:
                bot = GreedyBot(rng) if kind == "greedy" else RandomBot(rng)
                player = Player()
                for turn in range(1, turns + 1):
                    bot.shop(player)
                    player.end_turn()
                    player.team.move_forward()
                    library.add(turn, player.team.state)
                    player.start_turn(winner=rng.random() < 0.5)
        return library


class LibraryOpponent:
    """Samples a recorded team from a TeamLibrary for each turn"""

    def __init__(self, library):
        self.library = library
        self.rng = random.Random()

    def reset(self, rng=None):
        self.rng = _rng_from(rng)

    def team_for_turn(self, turn):
        return self.library.sample(turn, self.rng)

    def after_battle(self, result):
        pass

    def add_teams(self, teams):
        self.library.add_all(teams)


def harvest_teams(model, env, n_games, deterministic=False):
    """
    Play n_games with a trained model and return the teams it built each turn,
    for self-play: add them to a TeamLibrary so later training faces them.
    """
    env.record_teams = True
    env.recorded_teams = {}
    for _ in range(n_games):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(
                obs, action_masks=env.action_masks(), deterministic=deterministic
            )
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
    env.record_teams = False
    return env.recorded_teams
