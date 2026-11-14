import json
import os
import shutil
import tempfile
import unittest

from gui import game, replays

HERE = os.path.dirname(os.path.abspath(__file__))


class TestReplays(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.dir, self.index = replays.DIR, replays.INDEX
        replays.DIR = os.path.join(self.tmp, "replays")
        replays.INDEX = os.path.join(replays.DIR, "index.json")
        game.load_model("token_16.8M", os.path.join(HERE, "..", "gui", "models", "token_16.8M.npz"))

    def tearDown(self):
        replays.DIR, replays.INDEX = self.dir, self.index
        shutil.rmtree(self.tmp)

    def test_record_and_read_back(self):
        m = game.Match("token_16.8M", "greedy", seed=3)
        rid = "c" * 32
        for _ in range(3):
            m.play_round()
            replays.record(rid, m)
        meta = replays.listing()[0]
        self.assertEqual(meta["turns"], 3)
        self.assertEqual(meta["status"], "unfinished")
        doc = replays.get(rid)
        self.assertEqual(len(doc["rounds"]), 3)
        self.assertGreater(len(doc["rounds"][-1]["frames"]), 1)
        self.assertTrue(doc["rounds"][0]["thoughts"][0])  # the model's reasoning is kept
        self.assertEqual(len(replays.load_rounds(rid)), 3)

    def test_finished_game(self):
        m = game.Match("token_16.8M", "greedy", seed=1)
        rid = "d" * 32
        while not m.over and len(m.history) < 40:
            m.play_round()
        replays.record(rid, m)
        meta = replays.listing()[0]
        self.assertEqual(meta["status"], "finished")
        self.assertIn("won", meta["result"])
        self.assertGreaterEqual(min(s["lives"] for s in meta["score"]), 0)

    def test_rename_and_delete(self):
        m = game.Match("human", "greedy", seed=2)
        rid = "e" * 32
        m.do("end_turn")
        replays.record(rid, m)
        self.assertEqual(replays.rename(rid, "First win", "vs greedy")["title"], "First win")
        self.assertEqual(replays.listing()[0]["note"], "vs greedy")
        replays.delete(rid)
        self.assertEqual(replays.listing(), [])
        with self.assertRaises(Exception):
            replays.get(rid)

    def test_import_old_saves(self):
        saves = os.path.join(self.tmp, "saves")
        os.makedirs(saves)
        m = game.Match("human", "greedy", seed=2)
        m.do("end_turn")
        m.do("end_turn")
        with open(os.path.join(saves, "f" * 32 + ".json"), "w") as f:
            json.dump(m.to_save(), f, default=str)
        self.assertEqual(replays.import_saves(saves, game.Match.from_save), 1)
        doc = replays.get("f" * 32)
        self.assertFalse(doc["meta"]["steps"])
        self.assertEqual(len(doc["rounds"]), 2)
        self.assertTrue(doc["rounds"][0]["lines"])
        self.assertEqual(replays.import_saves(saves, game.Match.from_save), 0)

    def test_bad_ids_refused(self):
        with self.assertRaises(Exception):
            replays.get("../../etc/passwd")


if __name__ == "__main__":
    unittest.main()
