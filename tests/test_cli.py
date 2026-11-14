# %%
import unittest

from sapai.cli import Game


class ScriptedInput:
    """Feeds commands to the game, then answers with a strategy function"""

    def __init__(self, commands, strategy=None):
        self.commands = list(commands)
        self.strategy = strategy
        self.game = None

    def __call__(self, prompt):
        if self.commands:
            return self.commands.pop(0)
        if self.strategy is not None:
            return self.strategy(self.game)
        raise EOFError


def simple_strategy(game):
    """Buy the first affordable pet while there is room, then end the turn"""
    p = game.player
    if len(p.team) < 5:
        for i, slot in enumerate(p.shop):
            if slot.slot_type == "pet" and slot.cost <= p.gold:
                return f"buy {i}"
    return "end"


def make_game(commands, strategy=None, **kwargs):
    out = []
    feed = ScriptedInput(commands, strategy)
    game = Game(seed=1, input_fn=feed, print_fn=out.append, **kwargs)
    feed.game = game
    return game, out


class TestCLI(unittest.TestCase):
    def test_commands(self):
        game, out = make_game(["buy 0", "buy 0", "move 1 0", "freeze 0", "quit"])
        self.assertEqual(game.run(), "quit")
        p = game.player
        self.assertEqual(len(p.team), 2)
        self.assertEqual(p.gold, 4)
        ### Seed 1 shop is fish, otter, ant. Fish then otter are bought (the
        ###   shop shifts after a buy), then otter is moved to the front.
        self.assertEqual(p.team[0].pet.name, "pet-otter")
        self.assertEqual(p.team[1].pet.name, "pet-fish")
        self.assertTrue(p.shop[0].frozen)
        ### Reordered pets stay connected to the player
        for slot in p.team:
            if not slot.empty:
                self.assertIs(slot.pet.team, p.team)
        self.assertIs(p.team.player, p)
        self.assertIs(p.team.shop, p.shop)

    def test_invalid_commands_do_not_crash(self):
        game, out = make_game(
            ["buy 9", "sell 0", "foo", "buy", "move 0 3", "combine 0 0", "quit"]
        )
        self.assertEqual(game.run(), "quit")
        errors = [x for x in out if x.startswith("Can't do that")]
        self.assertEqual(len(errors), 6)

    def test_food_and_info(self):
        ### Seed 1 shop: fish, otter, ant, apple
        game, out = make_game(["buy 0", "buy 2 0", "info cow", "info apple", "quit"])
        game.run()
        self.assertEqual(game.player.team[0].attack, 3)
        self.assertEqual(game.player.team[0].health, 4)
        self.assertTrue(any(x.startswith("Cow: tier 5") for x in out))
        self.assertTrue(any(x.startswith("Apple: tier 1") for x in out))

    def test_merge_from_shop(self):
        from sapai import Shop

        game, out = make_game(["buy 0", "quit"])
        game.run()
        p = game.player
        ### Put a fish in the shop and merge it into the team fish
        p.shop = Shop(["fish"])
        p.gold = 3
        game.handle("buy 0 0")
        self.assertEqual(p.team[0].pet.experience, 1)

    def test_battle_and_log(self):
        game, out = make_game(["buy 0", "end", "log", "quit"])
        game.run()
        self.assertEqual(game.wins + game.losses + game.draws, 1)
        self.assertEqual(game.player.turn, 2)
        self.assertTrue(any(x == "  Attack 1" for x in out))
        self.assertTrue(any(x.startswith("  Result: ") for x in out))
        ### Log shows readable pet names instead of internal ids
        self.assertFalse(any("< pet-" in x for x in out))

    def test_log_attack_and_faint_lines(self):
        import numpy as np
        from sapai import Battle, Pet, Team

        np.random.seed(0)
        out = []
        game = Game(print_fn=out.append)
        b = Battle(Team(["ant", "otter"]), Team(["pigeon"]))
        game.last_result = b.battle()
        game.last_battle = b
        game.show_log()
        ### Stats before the attack, damage taken and stats after
        self.assertIn(
            "    Your Ant 2/2 and their Pigeon 3/2 attack each other:"
            + " yours takes 3 (now 2/-1), theirs takes 2 (now 3/0)",
            out,
        )
        ### Pets with a faint ability are also logged as fainting
        self.assertIn("    You: Ant faints", out)
        self.assertIn("    Opponent: Pigeon faints", out)
        self.assertIn("    You: Ant buffs Otter (now 2/5)", out)
        self.assertIn("  Result: You won.", out)

    def test_full_game(self):
        game, out = make_game([], simple_strategy)
        result = game.run()
        self.assertIn(result, ["won", "lost"])
        if result == "won":
            self.assertEqual(game.wins, 10)
        else:
            self.assertLessEqual(game.player.lives, 0)

    def test_full_games_many_seeds(self):
        for seed in range(5):
            out = []
            feed = ScriptedInput([], simple_strategy)
            game = Game(seed=seed, input_fn=feed, print_fn=out.append, lives=5)
            feed.game = game
            self.assertIn(game.run(), ["won", "lost"])


# %%


try:
    import sb3_contrib  # noqa: F401

    HAVE_RL = True
except ImportError:  # pragma: no cover
    HAVE_RL = False


class TestCLIOpponents(unittest.TestCase):
    def test_find_model_picks_most_trained(self):
        import os
        import tempfile

        from sapai.cli import find_model

        with tempfile.TemporaryDirectory() as d:
            for name in ["token_3.5M.zip", "token_10.2M.zip", "mlp_8.2M.zip", "x.zip"]:
                open(os.path.join(d, name), "w").close()
            self.assertEqual(find_model("token", d), os.path.join(d, "token_10.2M.zip"))
            self.assertEqual(find_model("mlp", d), os.path.join(d, "mlp_8.2M.zip"))
            with self.assertRaises(SystemExit):
                find_model("token", os.path.join(d, "missing"))

    def test_unknown_opponent(self):
        with self.assertRaises(SystemExit) as e:
            Game(opponent="dragon", print_fn=lambda x: None)
        self.assertIn("greedy, token, mlp", str(e.exception))

    def test_missing_rl_packages(self):
        import sys
        from unittest import mock

        with mock.patch.dict(sys.modules, {"sb3_contrib": None}):
            with self.assertRaises(SystemExit) as e:
                Game(opponent=__file__, print_fn=lambda x: None)
        self.assertIn("pip install torch gymnasium sb3-contrib", str(e.exception))

    @unittest.skipUnless(HAVE_RL, "sb3-contrib is required")
    def test_play_against_a_model(self):
        import os
        import tempfile

        from sb3_contrib import MaskablePPO

        from sapai.rl.env import SAPEnv

        with tempfile.TemporaryDirectory() as d:
            model = MaskablePPO("MlpPolicy", SAPEnv(), n_steps=64, device="cpu", seed=0)
            model.save(os.path.join(d, "mlp_1.0M.zip"))

            ### By kind, from a models folder, and by path
            out = []
            feed = ScriptedInput([], simple_strategy)
            game = Game(
                seed=1,
                input_fn=feed,
                print_fn=out.append,
                opponent="mlp",
                models_dir=d,
                lives=3,
            )
            feed.game = game
            self.assertIn(game.run(), ["won", "lost"])
            self.assertIn("Your opponent: trained model mlp_1.0M", out)
            ### The model plays the other side of the same match
            self.assertIs(game.match.players[1], game.opponent.player)
            self.assertEqual(game.match.wins[1], game.losses)
            self.assertEqual(game.opponent.player.turn, game.player.turn)

            game = Game(
                opponent=os.path.join(d, "mlp_1.0M.zip"),
                print_fn=lambda x: None,
            )
            self.assertEqual(game.opponent_name, "trained model mlp_1.0M")
