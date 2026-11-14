import random
import unittest

from sapai.data import RULES
from sapai.match import Match, lives_lost
from sapai.pets import Pet
from sapai.player import Player


def put(player, *names):
    """Put pets straight onto a player's team (front first)"""
    for name in names:
        player.team.append(Pet(name))


class TestMatch(unittest.TestCase):
    def test_starts_with_the_rules(self):
        m = Match()
        self.assertEqual(m.turn, 1)
        self.assertEqual([p.lives for p in m.players], [RULES["starting_lives"]] * 2)
        self.assertEqual(m.wins, [0, 0])
        self.assertIsNone(m.opponent_last_team(0))
        self.assertFalse(m.over)

    def test_winner_gains_a_win_and_loser_loses_lives_for_the_turn(self):
        m = Match(lives=10)
        put(m.players[0], "pig")
        record = m.battle()
        self.assertEqual(record["result"], 0)
        self.assertEqual(m.wins, [1, 0])
        self.assertEqual(m.players[1].lives, 9)
        self.assertEqual(m.players[0].lives, 10)
        self.assertEqual(m.turn, 2)
        self.assertEqual(m.players[0].turn, 2)
        self.assertEqual(m.players[1].turn, 2)

    def test_draw_costs_no_lives(self):
        m = Match(lives=10)
        record = m.battle()
        self.assertEqual(record["result"], 2)
        self.assertEqual([p.lives for p in m.players], [10, 10])
        self.assertEqual(m.wins, [0, 0])

    def test_a_loss_always_costs_one_life(self):
        self.assertEqual(RULES["starting_lives"], 6)
        for turn in (1, 5, 11, 30):
            self.assertEqual(lives_lost(turn), 1)

    def test_each_player_sees_the_other_team_as_it_entered_battle(self):
        m = Match()
        put(m.players[0], "fish")
        put(m.players[1], "pig", "beaver")
        m.battle()
        seen = m.opponent_last_team(0)
        names = [x.pet.name for x in seen if not x.empty]
        self.assertEqual(names, ["pet-pig", "pet-beaver"])
        ### Stats from before the battle, not the survivors after it
        self.assertEqual(seen[0].pet.health, Pet("pig").health)
        self.assertEqual(
            [x.pet.name for x in m.opponent_last_team(1) if not x.empty], ["pet-fish"]
        )
        ### Changing what you saw does not change the match
        seen[0].pet._health = 99
        self.assertNotEqual(m.opponent_last_team(0)[0].pet.health, 99)

    def test_view_is_from_each_players_side(self):
        m = Match()
        put(m.players[0], "pig")
        m.battle()
        v0, v1 = m.view(0), m.view(1)
        self.assertEqual(v0["last_result"], 0)
        self.assertEqual(v1["last_result"], 1)
        self.assertEqual(v0["opponent_lives"], m.players[1].lives)
        self.assertEqual(v1["opponent_wins"], 1)

    def test_ends_at_the_wins_needed(self):
        m = Match(wins_to_win=2, lives=99)
        ### Battles use copies of the teams, so the Pig stays for both
        put(m.players[0], "pig")
        m.battle()
        m.battle()
        self.assertTrue(m.over)
        self.assertEqual(m.winner, 0)
        with self.assertRaises(Exception):
            m.battle()

    def test_ends_when_a_player_runs_out_of_lives(self):
        m = Match(lives=1)
        put(m.players[1], "pig")
        m.battle()
        self.assertTrue(m.over)
        self.assertEqual(m.winner, 1)

    def test_turn_limit(self):
        m = Match(max_turns=2)
        m.battle()
        self.assertFalse(m.over)
        m.battle()
        self.assertTrue(m.over)
        self.assertIsNone(m.winner)

    def test_uses_given_players(self):
        a, b = Player(), Player()
        m = Match(players=[a, b], lives=3)
        self.assertIs(m.players[0], a)
        self.assertEqual(b.lives, 3)
        with self.assertRaises(ValueError):
            Match(players=[a])

    def test_greedy_bots_finish_a_match(self):
        from sapai.rl.opponents import GreedyBot

        for seed in range(3):
            m = Match(seed=seed)
            bots = [GreedyBot(random.Random(seed)), GreedyBot(random.Random(seed + 9))]
            while not m.over:
                for bot, p in zip(bots, m.players):
                    bot.shop(p)
                m.battle()
                self.assertLess(m.turn, 60)
            self.assertTrue(
                max(m.wins) >= RULES["wins_to_win"]
                or min(p.lives for p in m.players) <= 0
            )
            self.assertEqual(len(m.history), m.turn)


if __name__ == "__main__":
    unittest.main()
