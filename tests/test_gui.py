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

    def test_save_and_load(self):
        m = game.Match("human", "token_16.8M", seed=2)
        pet = next(x for x in m.state()["seats"][0]["shop"] if x["kind"] == "pet")
        m.do("buy_pet", pet["index"])
        m.do("end_turn")
        m2 = game.Match.from_save(json.loads(json.dumps(m.to_save(), default=str)))
        self.assertEqual(m.state(), m2.state())
        self.assertTrue(json.loads(m2.do("end_turn"))["ok"])

    def test_log_records_duck_sell_behind_pig(self):
        """The game log shows the Duck's sell buff (it used to be lost behind a Pig)."""
        for seed in range(400):
            m = game.Match("human", "greedy", seed=seed)
            shop = m.state()["seats"][0]["shop"]
            names = [x["name"] for x in shop]
            if "duck" in names and any(
                x["kind"] == "pet" and x["name"] != "duck" for x in shop
            ):
                break
        duck = names.index("duck")
        other = next(x for x in shop if x["kind"] == "pet" and x["name"] != "duck")
        m.do("buy_pet", shop[duck]["index"])
        m.do("buy_pet", other["index"])
        team = [p["name"] for p in m.state()["seats"][0]["team"]]
        if team[0] == "duck":
            m.do("swap", 0, 1)
        m.do("sell", 1)
        entry = m.events[-1]
        self.assertEqual(entry["title"], "sold duck")
        if any(x["kind"] == "pet" for x in m.state()["seats"][0]["shop"]):
            self.assertTrue(
                any(l.startswith("shop:") and "→" in l for l in entry["lines"])
            )
        self.assertEqual(entry["warn"], [])

    def test_no_rule_problems_in_model_games(self):
        for seed in range(5):
            m = game.Match("token_16.8M", "greedy", seed=seed)
            while not m.over and len(m.history) < 30:
                m.play_round()
            self.assertEqual([e for e in m.events if e["warn"]], [])

    def test_chat_answers_from_model_numbers(self):
        from gui import chat

        m = game.Match("human", "token_16.8M", seed=3)
        m.do("end_turn")
        thoughts = m.seats[1].thoughts
        self.assertTrue(thoughts)
        for t in thoughts:
            self.assertAlmostEqual(sum(p for _, p in t["options"]), 1.0, delta=0.05)
            self.assertIsNotNone(t["value"])
        rolled = next((t for t in thoughts if t["choice"] == "rolled"), None)
        r = chat.reply(m, "why did you roll?", use_claude=False)
        self.assertEqual(r["seat"], 1)
        if rolled:
            self.assertIn(chat.pct(rolled["confidence"]), r["text"])
        r = chat.reply(m, "what would you do in my spot?", use_claude=False)
        self.assertIn("In your seat", r["text"])

    def test_chat_without_model(self):
        from gui import chat

        m = game.Match("human", "greedy", seed=1)
        self.assertIsNone(chat.reply(m, "hi", use_claude=False)["seat"])


if __name__ == "__main__":
    unittest.main()
