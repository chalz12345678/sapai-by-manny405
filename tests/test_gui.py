import json
import os
import unittest

from gui import game

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, "..", "gui", "models")


class TestGui(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for name in ["token_16.8M", "mlp_8.2M"]:
            game.load_model(name, os.path.join(MODELS, f"{name}.npz"))

    def test_human_actions(self):
        m = game.Match("human", "greedy", seed=0)
        shop = m.state()["seats"][0]["shop"]
        pet = next(x for x in shop if x["kind"] == "pet")
        r = json.loads(m.do("buy_pet", pet["index"]))
        self.assertTrue(r["ok"], r["error"])
        self.assertEqual(len(r["state"]["seats"][0]["team"]), 1)
        r = json.loads(m.do("end_turn"))
        self.assertTrue(r["ok"], r["error"])
        self.assertGreater(len(r["battle"]["frames"]), 1)
        self.assertEqual(r["state"]["turn"], 2)

    def test_illegal_action_reports_error(self):
        m = game.Match("human", "greedy", seed=0)
        r = json.loads(m.do("sell", 0))
        self.assertFalse(r["ok"])

    def test_models_beat_greedy(self):
        """The numpy port should play as well as the README reports."""
        for name in ["token_16.8M", "mlp_8.2M"]:
            won = 0
            for seed in range(6):
                m = game.Match(name, "greedy", seed=seed)
                while not m.over and len(m.history) < 40:
                    m.play_round()
                won += m.result == "seat0"
            self.assertGreaterEqual(won, 4, name)

    def test_custom_agent(self):
        src = "def shop(seat, rng):\n    pass\n"
        game.load_agent_source("idle", src)
        m = game.Match("idle", "greedy", seed=1)
        m.play_round()
        self.assertEqual(len(m.history), 1)


if __name__ == "__main__":
    unittest.main()
