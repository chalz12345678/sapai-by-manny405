"""
Gymnasium environment for training agents on sapai (Turtle Pack): a 1v1 match.

One episode is one match against a single opponent (sapai.match.Match). One
step is one shop action. Ending the turn lets the opponent shop, then the two
teams battle, then the next turn begins. The opponent is the same all match.

Observation
-----------
A flat float32 vector made of tokens, see the layout constants below:
    5 team tokens:   pet one-hot, perk one-hot, present, attack, health,
                     level one-hot, experience, battle-only buffs, sell bonus
    7 shop tokens:   slot kind one-hot, pet one-hot, food one-hot, attack,
                     health, cost, frozen
    1 global token:  gold, turn, lives, wins, actions this turn, last battle
                     result one-hot, team size, shop size
    5 opponent team tokens: the opponent's team as it entered the last
                     battle, laid out like the team tokens
    opponent globals: opponent lives, opponent wins, seen (1 after the first
                     battle)
The first part (BASE_OBS_SIZE) is the layout models were trained on before
the opponent was visible; such models are given only that part.
The team is always kept compact, so team slot j is the j-th pet from the front.

Actions
-------
Discrete(118) with action masking, see sapai.actions. Illegal actions are
masked out by action_masks(), which sb3-contrib's MaskablePPO uses.

Rewards
-------
+1 for winning a battle, -1 for losing one, 0 for a draw, and a bonus for
winning the match. Nothing else is rewarded.

"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from sapai.data import data, RULES
from sapai.match import Match

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
BASE_OBS_SIZE = MAX_TEAM * TEAM_TOKEN + MAX_SHOP * SHOP_TOKEN + GLOBAL_TOKEN
OPPONENT_GLOBAL = 3  # lives, wins, seen
OPPONENT_START = BASE_OBS_SIZE
OBS_SIZE = BASE_OBS_SIZE + MAX_TEAM * TEAM_TOKEN + OPPONENT_GLOBAL

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


def encode_pet(t, pet):
    """Fill one team token for a pet"""
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


def encode(player, actions_this_turn=0, last_result=None, max_turns=25, wins=0):
    """
    The player's own state as the BASE_OBS_SIZE observation (no opponent
    information), the layout of models trained before matches were 1v1
    """
    obs = np.zeros(BASE_OBS_SIZE, dtype=np.float32)

    for j, pet in enumerate(team_pets(player)[:MAX_TEAM]):
        encode_pet(obs[j * TEAM_TOKEN : (j + 1) * TEAM_TOKEN], pet)

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

    g = obs[BASE_OBS_SIZE - GLOBAL_TOKEN :]
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


def encode_view(view, actions_this_turn=0, max_turns=25):
    """
    The full OBS_SIZE observation for one side of a match, from
    Match.view(i): the player's own state plus the opponent's last team,
    lives and wins
    """
    obs = np.zeros(OBS_SIZE, dtype=np.float32)
    obs[:BASE_OBS_SIZE] = encode(
        view["player"],
        actions_this_turn,
        view["last_result"],
        max_turns,
        view["wins"],
    )
    team = view["opponent_last_team"]
    if team is not None:
        pets = [slot.pet for slot in team if not slot.empty][:MAX_TEAM]
        for j, pet in enumerate(pets):
            start = OPPONENT_START + j * TEAM_TOKEN
            encode_pet(obs[start : start + TEAM_TOKEN], pet)
    g = obs[OBS_SIZE - OPPONENT_GLOBAL :]
    g[0] = max(view["opponent_lives"], 0) / 10
    g[1] = min(view["opponent_wins"], 10) / 10
    g[2] = float(team is not None)
    return obs


def observation_for(model, obs):
    """
    Cut a full observation down to what a model was trained on: models from
    before matches were 1v1 take only the first BASE_OBS_SIZE values
    """
    size = model.observation_space.shape[0]
    return obs[..., :size]


class SAPEnv(gym.Env):
    """
    Arguments
    ---------
    opponent: the other side of the match, an object with reset(rng) and
        shop(player, view) that shops for its own player each turn. See
        sapai.rl.opponents. Defaults to the greedy bot.
    wins_needed: wins needed to win the match
    lives: starting lives of both sides. Losses cost lives based on the turn.
    max_turns: matches are truncated after this many turns
    max_actions_per_turn: the turn is ended after this many shop actions
    game_win_bonus: extra reward for winning the match
    play_out: play all max_turns turns whatever the wins and lives (lives
        stay at least 1), for head-to-head benchmark matches
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
        self.play_out = play_out
        self.observation_space = spaces.Box(0.0, 2.0, (OBS_SIZE,), np.float32)
        self.action_space = spaces.Discrete(N_ACTIONS)
        self.engine_errors = 0
        ### Keep the team of the last battle in last_team_state
        self.last_team_state = None
        self.match = None
        self.player = None

    ############################################################################
    #### Gymnasium API
    ############################################################################

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        ### sapai draws its randomness from numpy's global random state
        np.random.seed(int(self.np_random.integers(0, 2**31 - 1)))
        self.match = Match(
            lives=self.start_lives, wins_to_win=self.wins_needed, endless=self.play_out
        )
        self.player = self.match.players[0]
        self.actions_this_turn = 0
        self.opponent.reset(self.np_random)
        return self._obs(), self._info()

    @property
    def wins(self):
        return self.match.wins[0]

    @property
    def last_result(self):
        return self.match.view(0)["last_result"]

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

    def render(self):
        return f"{self._info()}\n{self.player.team}\n{self.player.shop}"

    ############################################################################
    #### Internals
    ############################################################################

    def _obs(self):
        return encode_view(self.match.view(0), self.actions_this_turn, self.max_turns)

    def _info(self):
        results = [r["result"] for r in self.match.history]
        return {
            "turn": self.player.turn,
            "wins": self.match.wins[0],
            "losses": results.count(1),
            "draws": results.count(2),
            "lives": self.player.lives,
            "opponent_lives": self.match.players[1].lives,
            "engine_errors": self.engine_errors,
        }

    def _end_turn(self):
        m = self.match
        try:
            self.opponent.shop(m.players[1], m.view(1))
        except Exception:
            self.engine_errors += 1
        try:
            record = m.battle()
            result = record["result"]
        except Exception:
            ### Engine bug: count it, and end the match as a draw
            self.engine_errors += 1
            info = self._info()
            info["result"] = 2
            info["game_won"] = False
            return self._obs(), 0.0, False, True, info
        self.last_team_state = record["teams"][0]

        reward = {0: 1.0, 1: -1.0, 2: 0.0}[result]
        terminated = m.over and not self.play_out
        if terminated and m.winner == 0:
            reward += self.game_win_bonus
        self.actions_this_turn = 0
        truncated = (not terminated) and (m.over or self.player.turn > self.max_turns)
        info = self._info()
        info["result"] = result
        if terminated or truncated:
            info["game_won"] = m.winner == 0
        return self._obs(), reward, terminated, truncated, info
