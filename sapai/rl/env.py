"""
Gymnasium environment for training agents on sapai (Turtle Pack).

One episode is one game. One step is one shop action. Ending the turn starts a
battle against an opponent team for that turn, then the next turn begins.

Observation
-----------
A flat float32 vector made of tokens, see the layout constants below:
    5 team tokens:   pet one-hot, perk one-hot, present, attack, health,
                     level one-hot, experience, battle-only buffs, sell bonus
    7 shop tokens:   slot kind one-hot, pet one-hot, food one-hot, attack,
                     health, cost, frozen
    1 global token:  gold, turn, lives, wins, actions this turn, last battle
                     result one-hot, team size, shop size
The team is always kept compact, so team slot j is the j-th pet from the front.

Actions
-------
Discrete(118) with action masking, see ACTIONS below. Illegal actions are
masked out by action_masks(), which sb3-contrib's MaskablePPO uses.

Rewards
-------
+1 for winning a battle, -1 for losing one, 0 for a draw, and a bonus for
winning the game. Nothing else is rewarded.

"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from sapai.battle import Battle
from sapai.data import data, RULES
from sapai.player import Player
from sapai.teams import Team

from sapai.actions import MAX_SHOP, MAX_TEAM

MAX_STAT = 50

PET_IDS = sorted(k for k in data["pets"] if k != "pet-none")
FOOD_IDS = sorted(k for k in data["foods"] if k != "food-none")
STATUS_IDS = ["none"] + sorted(data["statuses"])
PET_INDEX = {k: i for i, k in enumerate(PET_IDS)}
FOOD_INDEX = {k: i for i, k in enumerate(FOOD_IDS)}
STATUS_INDEX = {k: i for i, k in enumerate(STATUS_IDS)}

N_PETS = len(PET_IDS)
N_FOODS = len(FOOD_IDS)
N_STATUSES = len(STATUS_IDS)

### Token layouts
TEAM_NUM = 10  # present, attack, health, level x3, xp, buff attack, buff health, sell
TEAM_TOKEN = N_PETS + N_STATUSES + TEAM_NUM
SHOP_KINDS = ["pet", "levelup", "food"]
SHOP_NUM = 4  # attack, health, cost, frozen
SHOP_TOKEN = len(SHOP_KINDS) + N_PETS + N_FOODS + SHOP_NUM
GLOBAL_TOKEN = 12
OBS_SIZE = MAX_TEAM * TEAM_TOKEN + MAX_SHOP * SHOP_TOKEN + GLOBAL_TOKEN

### The action list and its helpers live in sapai.actions
from sapai.actions import (  # noqa: E402,F401
    ACTION_START,
    ACTIONS,
    END_TURN,
    N_ACTIONS,
    PAIRS,
    action_mask,
    apply_action,
    food_needs_target,
    team_pets,
)


def encode(player, actions_this_turn=0, last_result=None, max_turns=25, wins=0):
    """Encode the player's state as the flat observation vector"""
    obs = np.zeros(OBS_SIZE, dtype=np.float32)

    for j, pet in enumerate(team_pets(player)[:MAX_TEAM]):
        t = obs[j * TEAM_TOKEN : (j + 1) * TEAM_TOKEN]
        t[PET_INDEX[pet.name]] = 1
        t[N_PETS + STATUS_INDEX.get(pet.status, 0)] = 1
        n = N_PETS + N_STATUSES
        t[n] = 1
        t[n + 1] = min(pet.attack, MAX_STAT) / MAX_STAT
        t[n + 2] = min(pet.health, MAX_STAT) / MAX_STAT
        t[n + 2 + pet.level] = 1
        t[n + 6] = pet.experience / 3
        t[n + 7] = min(pet._until_end_of_battle_attack_buff, MAX_STAT) / MAX_STAT
        t[n + 8] = min(pet._until_end_of_battle_health_buff, MAX_STAT) / MAX_STAT
        t[n + 9] = min(pet.sell_value_bonus, 10) / 10

    shop_start = MAX_TEAM * TEAM_TOKEN
    for i, slot in enumerate(player.shop.slots[:MAX_SHOP]):
        if slot.empty:
            continue
        t = obs[shop_start + i * SHOP_TOKEN : shop_start + (i + 1) * SHOP_TOKEN]
        t[SHOP_KINDS.index(slot.slot_type)] = 1
        n = len(SHOP_KINDS)
        obj = slot.obj
        if slot.slot_type == "food":
            t[n + N_PETS + FOOD_INDEX[obj.name]] = 1
        else:
            t[n + PET_INDEX[obj.name]] = 1
        n += N_PETS + N_FOODS
        t[n] = min(max(obj.attack, 0), MAX_STAT) / MAX_STAT
        t[n + 1] = min(max(obj.health, 0), MAX_STAT) / MAX_STAT
        t[n + 2] = slot.cost / 3
        t[n + 3] = float(slot.frozen)

    g = obs[OBS_SIZE - GLOBAL_TOKEN :]
    g[0] = min(player.gold, 40) / 20
    g[1] = min(player.turn, max_turns) / max_turns
    g[2] = max(player.lives, 0) / 10
    g[3] = min(wins, 10) / 10
    g[4] = min(actions_this_turn, 30) / 30
    g[5 + {None: 0, 0: 1, 1: 2, 2: 3}[last_result]] = 1
    g[9] = len(player.team) / MAX_TEAM
    g[10] = len(player.shop.slots) / MAX_SHOP
    g[11] = 1  # constant, marks the global token
    return obs


class SAPEnv(gym.Env):
    """
    Arguments
    ---------
    opponent: object with reset(rng), team_for_turn(turn) and
        after_battle(result). See sapai.rl.opponents. Defaults to a live
        greedy bot.
    wins_needed: wins needed to win the game
    lives: starting lives. Losses cost lives based on the turn, like the game.
    max_turns: games are truncated after this many turns
    max_actions_per_turn: the turn is ended after this many shop actions
    game_win_bonus: extra reward for winning the game
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        opponent=None,
        wins_needed=RULES["wins_to_win"],
        lives=RULES["starting_lives"],
        max_turns=25,
        max_actions_per_turn=30,
        game_win_bonus=5.0,
        play_out=False,
    ):
        super().__init__()
        if opponent is None:
            from sapai.rl.opponents import GreedyOpponent

            opponent = GreedyOpponent()
        self.opponent = opponent
        self.wins_needed = wins_needed
        self.start_lives = lives
        self.max_turns = max_turns
        self.max_actions_per_turn = max_actions_per_turn
        self.game_win_bonus = game_win_bonus
        ### play_out: play all max_turns turns regardless of wins and lives,
        ###   for head-to-head benchmark matches. Lives stay at least 1 so the
        ###   observation stays like a normal game.
        self.play_out = play_out
        self.observation_space = spaces.Box(0.0, 2.0, (OBS_SIZE,), np.float32)
        self.action_space = spaces.Discrete(N_ACTIONS)
        self.engine_errors = 0
        ### Teams the player had at the end of each turn, for self-play
        self.record_teams = False
        self.recorded_teams = {}
        ### Keep the team of the last battle in last_team_state
        self.record_last_team = False
        self.last_team_state = None
        self.player = None

    ############################################################################
    #### Gymnasium API
    ############################################################################

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        ### sapai draws its randomness from numpy's global random state
        np.random.seed(int(self.np_random.integers(0, 2**31 - 1)))
        self.player = Player(lives=self.start_lives)
        self.wins = 0
        self.losses = 0
        self.draws = 0
        self.actions_this_turn = 0
        self.last_result = None
        self.opponent.reset(self.np_random)
        return self._obs(), self._info()

    def step(self, action):
        action = int(action)
        if not self.action_masks()[action]:
            raise ValueError(f"Illegal action {action} {ACTIONS[action]}")

        if action == END_TURN:
            return self._end_turn()

        try:
            apply_action(self.player, action)
        except Exception:
            ### Engine bug: count it and treat the action as doing nothing
            self.engine_errors += 1
        self.actions_this_turn += 1
        if self.actions_this_turn >= self.max_actions_per_turn:
            return self._end_turn()
        return self._obs(), 0.0, False, False, self._info()

    def action_masks(self):
        return action_mask(self.player)

    def add_opponent_teams(self, teams, source="self", age=0.0):
        """Add recorded teams to a library opponent, used for self-play"""
        self.opponent.add_teams(teams, source, age)

    def cap_opponent_teams(self, max_total):
        """Remove the oldest self-play teams beyond max_total teams in total"""
        return self.opponent.cap_teams(max_total)

    def prune_opponent_teams(self, source, keep_fraction):
        """Keep keep_fraction of a library opponent's teams from source"""
        import random

        rng = random.Random(int(self.np_random.integers(0, 2**31 - 1)))
        return self.opponent.prune_teams(source, keep_fraction, rng)

    def render(self):
        return f"{self._info()}\n{self.player.team}\n{self.player.shop}"

    ############################################################################
    #### Internals
    ############################################################################

    def _obs(self):
        return encode(
            self.player,
            self.actions_this_turn,
            self.last_result,
            self.max_turns,
            self.wins,
        )

    def _info(self):
        return {
            "turn": self.player.turn,
            "wins": self.wins,
            "losses": self.losses,
            "draws": self.draws,
            "lives": self.player.lives,
            "engine_errors": self.engine_errors,
        }

    def _end_turn(self):
        p = self.player
        turn = p.turn
        p.end_turn()
        p.team.move_forward()
        if self.record_teams:
            self.recorded_teams.setdefault(turn, []).append(p.team.state)
        if self.record_last_team:
            self.last_team_state = p.team.state

        opponent_team = self.opponent.team_for_turn(turn)
        try:
            result = Battle(p.team, opponent_team).battle()
        except Exception:
            self.engine_errors += 1
            result = 2
        self.opponent.after_battle(result)

        reward = 0.0
        if result == 0:
            self.wins += 1
            reward = 1.0
        elif result == 1:
            self.losses += 1
            reward = -1.0
            p.lives -= data["turns"][f"turn-{min(turn, 11)}"]["livesLost"]
        else:
            self.draws += 1
        self.last_result = result

        terminated = False
        if self.play_out:
            p.lives = max(p.lives, 1)
        elif self.wins >= self.wins_needed:
            reward += self.game_win_bonus
            terminated = True
        elif p.lives <= 0:
            terminated = True

        ### A loss counts as losing the last battle for Snail
        p.start_turn(winner=(result != 1))
        self.actions_this_turn = 0
        truncated = (not terminated) and p.turn > self.max_turns
        info = self._info()
        info["result"] = result
        if terminated or truncated:
            info["game_won"] = self.wins >= self.wins_needed
        return self._obs(), reward, terminated, truncated, info
