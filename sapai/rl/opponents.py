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
from sapai.data import data
from sapai.rl.env import END_TURN, action_mask, apply_action, encode
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


class ModelOpponent:
    """
    A live opponent played by a saved model, for example a frozen earlier
    version of the agent. It shops each turn with its own Player.
    """

    def __init__(
        self, model, deterministic=True, max_actions_per_turn=30, max_turns=25
    ):
        self.model = model
        self.deterministic = deterministic
        self.max_actions_per_turn = max_actions_per_turn
        self.max_turns = max_turns
        self.player = None

    def reset(self, rng=None):
        self.player = Player()
        self.wins = 0
        self.last_result = None

    def team_for_turn(self, turn):
        self.shop()
        self.player.end_turn()
        self.player.team.move_forward()
        return self.player.team

    def shop(self):
        """Take the model's shop actions for this turn, up to ending the turn"""
        p = self.player
        for actions in range(self.max_actions_per_turn):
            obs = encode(p, actions, self.last_result, self.max_turns, self.wins)
            action, _ = self.model.predict(
                obs, action_masks=action_mask(p), deterministic=self.deterministic
            )
            if int(action) == END_TURN:
                break
            try:
                apply_action(p, int(action))
            except Exception:
                pass
        p.team.move_forward()

    def after_battle(self, result):
        self.record_result(result)
        self.player.start_turn(winner=(self.last_result != 1))

    def record_result(self, result):
        """
        Update wins, lives and the last result the model sees. result is from
        the other side: 0 means this opponent lost.
        """
        own_result = {0: 1, 1: 0, 2: 2}[result]
        if own_result == 0:
            self.wins += 1
        elif own_result == 1:
            turn = min(self.player.turn, 11)
            self.player.lives = max(
                self.player.lives - data["turns"][f"turn-{turn}"]["livesLost"], 1
            )
        self.last_result = own_result


class TeamLibrary:
    """
    Recorded teams for each turn, stored as Team states. Each team has a
    source, "bot" for teams from bots and "self" for self-play teams, so a
    source can be thinned out later with prune(), and an age (the training
    step it was harvested at) so the oldest teams can be removed with cap().
    """

    def __init__(self, teams=None, sources=None, ages=None):
        self.teams = teams or {}
        if sources is None:
            sources = {t: ["bot"] * len(x) for t, x in self.teams.items()}
        self.sources = sources
        if ages is None:
            ### Teams were always appended in time order, so the position in
            ###   each turn gives the order they were added in
            ages = {
                t: [i / max(len(x), 1) for i in range(len(x))]
                for t, x in self.teams.items()
            }
        self.ages = ages

    def add(self, turn, team_state, source="bot", age=0.0):
        self.teams.setdefault(int(turn), []).append(team_state)
        self.sources.setdefault(int(turn), []).append(source)
        self.ages.setdefault(int(turn), []).append(float(age))

    def add_all(self, teams, source="bot", age=0.0):
        for turn, states in teams.items():
            for state in states:
                self.add(turn, state, source, age)

    def add_library(self, other, source=None):
        """Add all teams of another library, keeping their ages"""
        for turn, states in other.teams.items():
            for state, s, age in zip(states, other.sources[turn], other.ages[turn]):
                self.add(turn, state, source or s, age)

    def count(self, source=None):
        if source is None:
            return len(self)
        return sum(x.count(source) for x in self.sources.values())

    def _keep(self, turn, keep_idx):
        keep_idx = sorted(keep_idx)
        self.teams[turn] = [self.teams[turn][i] for i in keep_idx]
        self.sources[turn] = [self.sources[turn][i] for i in keep_idx]
        self.ages[turn] = [self.ages[turn][i] for i in keep_idx]

    def prune(self, source, keep_fraction, rng):
        """Keep a random keep_fraction of the teams from source in every turn"""
        for turn in list(self.teams):
            from_source = [i for i, s in enumerate(self.sources[turn]) if s == source]
            n_keep = int(round(len(from_source) * keep_fraction))
            if from_source and n_keep == 0:
                n_keep = 1
            kept = set(rng.sample(from_source, n_keep))
            keep_idx = [
                i for i, s in enumerate(self.sources[turn]) if s != source or i in kept
            ]
            self._keep(turn, keep_idx)

    def cap(self, max_total, source="self"):
        """
        Remove the oldest teams from source until the library holds at most
        max_total teams. Returns the number of teams removed.
        """
        n_remove = len(self) - max(max_total, 0)
        if n_remove <= 0:
            return 0
        candidates = sorted(
            (age, turn, i)
            for turn in self.teams
            for i, (s, age) in enumerate(zip(self.sources[turn], self.ages[turn]))
            if s == source
        )
        remove = {}
        for age, turn, i in candidates[:n_remove]:
            remove.setdefault(turn, set()).add(i)
        for turn, idx in remove.items():
            self._keep(turn, [i for i in range(len(self.teams[turn])) if i not in idx])
        return sum(len(x) for x in remove.values())

    def sample(self, turn, rng):
        turns = sorted(t for t in self.teams if self.teams[t])
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
            pickle.dump(
                {"teams": self.teams, "sources": self.sources, "ages": self.ages}, f
            )

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        if "teams" in data and "sources" in data:
            return cls(data["teams"], data["sources"], data.get("ages"))
        ### Libraries saved before sources were recorded only hold bot teams
        return cls(data)

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
                    library.add(turn, player.team.state, "bot")
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

    def add_teams(self, teams, source="self", age=0.0):
        self.library.add_all(teams, source, age)

    def cap_teams(self, max_total):
        self.library.cap(max_total)
        return self.library.count("self")

    def prune_teams(self, source, keep_fraction, rng=None):
        self.library.prune(source, keep_fraction, rng or self.rng)
        return self.library.count(source)


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
