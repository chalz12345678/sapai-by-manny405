"""
A 1v1 match: two players who meet in battle every turn.

Each turn both players shop at the same time, each without seeing the
other's shop or current team. Then both end their turn and their teams
battle. After the battle each player knows what the other's team was when
it entered that battle (pets, levels, stats and perks), as you would from
watching the battle in the game, plus the other player's lives and wins.

    from sapai.match import Match

    m = Match()
    p0, p1 = m.players
    ... each side shops with p0 / p1 (buy_pet, sell, roll, ...) ...
    record = m.battle()               # ends the turn for both and battles
    m.opponent_last_team(0)           # what player 0 saw player 1 field
    m.over, m.winner

Lives, wins needed and lives lost per turn come from the rules in
data/turtle_pack.yaml; Match takes overrides for each.
"""

import numpy as np

from sapai.battle import Battle
from sapai.data import RULES, data
from sapai.player import Player
from sapai.teams import Team


def lives_lost(turn):
    """Lives a player loses for losing a battle on this turn"""
    last = len(data["turns"])
    return data["turns"][f"turn-{min(turn, last)}"]["livesLost"]


class Match:
    def __init__(
        self,
        lives=None,
        wins_to_win=RULES["wins_to_win"],
        max_turns=None,
        seed=None,
        lives_lost=lives_lost,
        players=None,
        endless=False,
    ):
        """
        lives: starting lives of each player
        wins_to_win: battle wins that win the match, or None (the rules'
            default) for no limit: only running out of lives ends the match
        max_turns: optional turn limit; when it is reached the match ends,
            won by the player with more wins (or drawn)
        seed: seeds numpy's random numbers, for a reproducible match
        lives_lost: function of the turn number giving the lives lost for a
            loss on that turn
        players: two existing Players to use (for example ones a bot
            already shops with); their lives are set to the starting lives
        endless: keep playing whatever the wins and lives (lives stay at
            least 1), until max_turns if given; for head-to-head benchmarks
        """
        self.endless = endless
        if seed is not None:
            np.random.seed(seed)
        self.start_lives = RULES["starting_lives"] if lives is None else lives
        self.wins_to_win = wins_to_win
        self.max_turns = max_turns
        self.lives_lost = lives_lost
        if players is None:
            players = [Player(lives=self.start_lives) for _ in range(2)]
        if len(players) != 2:
            raise ValueError("a match needs exactly two players")
        self.players = list(players)
        for p in self.players:
            p.lives = self.start_lives
        self.wins = [0, 0]
        self.turn = 1
        ### Team states as each player entered the last battle
        self.last_teams = [None, None]
        self.last_result = None
        self.last_battle = None
        self.history = []
        self.winner = None
        self.over = False

    ############################################################################
    #### What each player can see
    ############################################################################

    @staticmethod
    def other(i):
        return 1 - i

    def opponent_last_team(self, i):
        """
        The team player i's opponent battled with last turn, as a new Team
        (None before the first battle). Changing it does not change the match.
        """
        state = self.last_teams[self.other(i)]
        return None if state is None else Team.from_state(state)

    def view(self, i):
        """Everything player i knows about the match"""
        o = self.other(i)
        result = None
        if self.last_result is not None:
            result = (
                self.last_result if i == 0 else {0: 1, 1: 0, 2: 2}[self.last_result]
            )
        return {
            "player": self.players[i],
            "turn": self.turn,
            "lives": self.players[i].lives,
            "wins": self.wins[i],
            "opponent_lives": self.players[o].lives,
            "opponent_wins": self.wins[o],
            "opponent_last_team": self.opponent_last_team(i),
            ### 0 won, 1 lost, 2 draw, from player i's side
            "last_result": result,
        }

    ############################################################################
    #### Playing
    ############################################################################

    def battle(self):
        """
        End the turn for both players, battle their teams, update lives and
        wins, and start the next turn (unless the match is over). Returns a
        record of the turn: the turn number, both teams as they entered the
        battle, the result (0: player 0 won, 1: player 1 won, 2: draw) and
        lives and wins afterwards.
        """
        if self.over:
            raise Exception("The match is over")
        p0, p1 = self.players
        for p in self.players:
            p.end_turn()
            p.team.move_forward()
        self.last_teams = [p0.team.state, p1.team.state]

        b = Battle(p0.team, p1.team)
        result = b.battle()
        self.last_battle = b
        self.last_result = result

        if result in (0, 1):
            winner, loser = (0, 1) if result == 0 else (1, 0)
            self.wins[winner] += 1
            self.players[loser].lives -= self.lives_lost(self.turn)

        record = {
            "turn": self.turn,
            "teams": list(self.last_teams),
            "result": result,
            "lives": [p.lives for p in self.players],
            "wins": list(self.wins),
        }
        self.history.append(record)
        self._check_over()

        if not self.over:
            self.turn += 1
            ### A player who lost counts as losing the last battle (Snail)
            p0.start_turn(winner=(result != 1))
            p1.start_turn(winner=(result != 0))
        return record

    def _check_over(self):
        if self.endless:
            for p in self.players:
                p.lives = max(p.lives, 1)
            if self.max_turns is not None and self.turn >= self.max_turns:
                self.over = True
                if self.wins[0] != self.wins[1]:
                    self.winner = 0 if self.wins[0] > self.wins[1] else 1
            return
        for i in (0, 1):
            if self.wins_to_win is not None and self.wins[i] >= self.wins_to_win:
                self.over, self.winner = True, i
                return
        out = [p.lives <= 0 for p in self.players]
        if out[0] or out[1]:
            self.over = True
            self.winner = None if out[0] and out[1] else (1 if out[0] else 0)
            return
        if self.max_turns is not None and self.turn >= self.max_turns:
            self.over = True
            if self.wins[0] != self.wins[1]:
                self.winner = 0 if self.wins[0] > self.wins[1] else 1
            else:
                self.winner = None
