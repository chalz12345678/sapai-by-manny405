"""
Battle Lab tests on synthetic frames: each pet is a distinct picture, stats
are white digits drawn with a small bitmap font on dark circles, and perks
and levels are coloured markers. The real game looks different, but the lab
only relies on what it learns from frames, so the same steps apply.
"""

import os
import tempfile
import unittest
import zlib
from unittest import mock

import numpy as np

from gui import lab

### 3x5 bitmap digits
FONT = {
    "0": ["111", "101", "101", "101", "111"],
    "1": ["010", "110", "010", "010", "111"],
    "2": ["111", "001", "111", "100", "111"],
    "3": ["111", "001", "111", "001", "111"],
    "4": ["101", "101", "111", "001", "001"],
    "5": ["111", "100", "111", "001", "111"],
    "6": ["111", "100", "111", "101", "111"],
    "7": ["111", "001", "010", "010", "010"],
    "8": ["111", "101", "111", "101", "111"],
    "9": ["111", "101", "111", "001", "111"],
}
W, H = 960, 540
SCALE = 3

LAYOUT = {
    ### Player pets on the left, front (index 0) nearest the middle
    "player": [[0.42 - 0.08 * i, 0.5] for i in range(5)],
    "opponent": [[0.58 + 0.08 * i, 0.5] for i in range(5)],
    "slot": [0.06, 0.1],
    "attack": [-0.018, 0.085, 0.028, 0.045],
    "health": [0.018, 0.085, 0.028, 0.045],
    "perk": [0.02, -0.075, 0.02, 0.03],
    "level": [-0.015, -0.075, 0.03, 0.03],
    "mirror": False,
}


def pet_picture(name, h, w):
    ### crc32, not hash(): Python's hash of a string changes every run
    rng = np.random.default_rng(zlib.crc32(name.encode()))
    base = rng.integers(40, 200, size=3)
    pic = np.zeros((h, w, 3)) + base
    for _ in range(4):
        y, x = rng.integers(0, h - 4), rng.integers(0, w - 4)
        pic[y : y + h // 3, x : x + w // 3] = rng.integers(0, 255, size=3)
    return pic


def draw_number(frame, box, value):
    cx, cy, bw, bh = box
    x0, y0 = int((cx - bw / 2) * W), int((cy - bh / 2) * H)
    x1, y1 = int((cx + bw / 2) * W), int((cy + bh / 2) * H)
    frame[y0:y1, x0:x1] = (60, 20, 20)
    text = str(value)
    width = len(text) * 4 * SCALE - SCALE
    x = (x0 + x1) // 2 - width // 2
    y = (y0 + y1) // 2 - 5 * SCALE // 2
    for ch in text:
        for r, row in enumerate(FONT[ch]):
            for c, bit in enumerate(row):
                if bit == "1":
                    frame[
                        y + r * SCALE : y + (r + 1) * SCALE,
                        x + c * SCALE : x + (c + 1) * SCALE,
                    ] = 255
        x += 4 * SCALE


def fill(frame, box, color):
    cx, cy, bw, bh = box
    x0, y0 = int((cx - bw / 2) * W), int((cy - bh / 2) * H)
    x1, y1 = int((cx + bw / 2) * W), int((cy + bh / 2) * H)
    frame[y0:y1, x0:x1] = color


PERK_COLORS = {"none": (30, 30, 30), "Garlic": (230, 230, 200), "Melon": (40, 200, 60)}
LEVEL_COLORS = {"1": (90, 90, 90), "2": (200, 160, 30), "3": (220, 60, 220)}


def make_frame(teams):
    """teams: {"player": [pet or None, ...], "opponent": [...]}, front first"""
    frame = np.full((H, W, 3), (70, 110, 160), dtype=np.float64)
    for side in lab.SIDES:
        for i, pet in enumerate(teams.get(side, [])):
            if pet is None:
                continue
            box = lab.part_box(LAYOUT, side, i, "pet")
            cx, cy, bw, bh = box
            x0, y0 = int((cx - bw / 2) * W), int((cy - bh / 2) * H)
            x1, y1 = int((cx + bw / 2) * W), int((cy + bh / 2) * H)
            frame[y0:y1, x0:x1] = pet_picture(pet["name"], y1 - y0, x1 - x0)
            draw_number(frame, lab.part_box(LAYOUT, side, i, "attack"), pet["attack"])
            draw_number(frame, lab.part_box(LAYOUT, side, i, "health"), pet["health"])
            fill(frame, lab.part_box(LAYOUT, side, i, "perk"), PERK_COLORS[pet["perk"]])
            fill(
                frame,
                lab.part_box(LAYOUT, side, i, "level"),
                LEVEL_COLORS[str(pet["level"])],
            )
    return frame.astype(np.uint8)


def pet(name, attack, health, level=1, perk="none"):
    return {
        "name": name,
        "attack": attack,
        "health": health,
        "level": level,
        "perk": perk,
    }


def labels(teams):
    return {
        side: [p if p is not None else {"empty": True} for p in teams[side]]
        for side in lab.SIDES
    }


class LabCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        patcher = mock.patch.object(lab, "DATA_DIR", self.tmp.name)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.tmp.cleanup)
        lab.save_layout(LAYOUT)


class TestImages(LabCase):
    def test_resize_keeps_the_mean(self):
        a = np.random.default_rng(0).integers(0, 255, (37, 53, 3)).astype(float)
        r = lab.resize(a, 10, 12)
        self.assertEqual(r.shape, (10, 12, 3))
        self.assertAlmostEqual(r.mean(), a.mean(), delta=1.0)

    def test_png_is_valid(self):
        import base64

        data = base64.b64decode(lab.png_b64(np.zeros((4, 5, 3))))
        self.assertTrue(data.startswith(b"\x89PNG"))

    def test_glyphs_split_digits(self):
        frame = np.zeros((60, 80, 3), dtype=np.float64)
        box = (0.5, 0.5, 0.8, 0.6)
        global W, H
        old = (W, H)
        W, H = 80, 60
        try:
            draw_number(frame, box, 47)
        finally:
            W, H = old
        self.assertEqual(len(lab.glyphs(frame.astype(np.uint8))), 2)

    def test_layout_is_checked(self):
        with self.assertRaises(ValueError):
            lab.save_layout({"player": [[0.1, 0.1]]})


class TestLearningToRead(LabCase):
    FIRST = {
        "player": [
            pet("Ant", 2, 1),
            pet("Fish", 3, 4, 2, "Garlic"),
            pet("Pig", 4, 1),
            None,
            None,
        ],
        "opponent": [
            pet("Beaver", 5, 6, perk="Melon"),
            pet("Duck", 7, 8, 3),
            pet("Otter", 9, 10),
            None,
            None,
        ],
    }

    def test_reads_a_new_frame_after_one_lesson(self):
        added = lab.teach(make_frame(self.FIRST), labels(self.FIRST))
        self.assertEqual(added["pet"], 7)  # 6 pets and the empty slot picture
        self.assertGreaterEqual(added["digit"], 10)

        ### Same kinds of pets in different places with different stats
        second = {
            "player": [
                pet("Otter", 10, 3, perk="Garlic"),
                pet("Ant", 1, 9, 3),
                None,
                None,
                None,
            ],
            "opponent": [
                pet("Pig", 27, 18, 2, "Melon"),
                pet("Fish", 6, 5),
                pet("Beaver", 41, 50),
                pet("Duck", 8, 8),
                None,
            ],
        }
        seen = lab.read_frame(make_frame(second))
        for side in lab.SIDES:
            for i, want in enumerate(second[side]):
                got = seen[side][i]
                with self.subTest(side=side, slot=i):
                    if want is None:
                        self.assertTrue(got["empty"])
                        continue
                    self.assertEqual(got["name"], want["name"])
                    self.assertEqual(got["attack"], want["attack"])
                    self.assertEqual(got["health"], want["health"])
                    self.assertEqual(got["perk"], want["perk"])
                    self.assertEqual(got["level"], str(want["level"]))
                    self.assertEqual(got["unsure"], [])

    def test_unknown_pet_is_flagged(self):
        lab.teach(make_frame(self.FIRST), labels(self.FIRST))
        seen = lab.read_frame(
            make_frame({"player": [pet("Mammoth", 3, 10)], "opponent": []})
        )
        self.assertIn("name", seen["player"][0]["unsure"])

    def test_memory_survives_a_restart(self):
        lab.teach(make_frame(self.FIRST), labels(self.FIRST))
        memory = lab.Memory.load()
        self.assertIn("Ant", memory.items["pet"])
        self.assertEqual(set(memory.items["digit"]), set("0123456789") - {"0"} | {"0"})


class TestRecreatingTheBattle(LabCase):
    START = {
        "player": [pet("Fish", 2, 3)],
        "opponent": [pet("Fish", 2, 3)],
    }

    def test_run_replays_and_matches_the_game(self):
        out = lab.run(self.START, {"player": [], "opponent": []}, runs=10)
        self.assertEqual(out["outcome"], "draw")
        self.assertGreater(len(out["frames"]), 1)
        self.assertEqual(out["odds"]["draw"], 1.0)
        self.assertTrue(out["match"]["matches"])
        self.assertEqual(out["game"]["result"], "draw")

    def test_a_different_ending_is_a_discrepancy(self):
        out = lab.run(
            self.START,
            {"player": [pet("Fish", 2, 1)], "opponent": []},
            runs=10,
        )
        self.assertFalse(out["match"]["matches"])
        self.assertEqual(out["game"]["result"], "win")

    def test_save_writes_a_scenario_the_runner_accepts(self):
        from sapai.scenarios import check, load

        with tempfile.TemporaryDirectory() as d:
            with mock.patch.object(lab, "SCENARIO_DIR", d):
                saved = lab.save(self.START, {"player": [], "opponent": []}, runs=10)
                self.assertTrue(saved["file"].endswith("from_game.yaml"))
                bad = lab.save(
                    self.START, {"player": [pet("Fish", 2, 1)], "opponent": []}, runs=10
                )
                self.assertTrue(bad["file"].endswith("game_discrepancies.yaml"))
                good = load([os.path.join(d, "from_game.yaml")])
                wrong = load([os.path.join(d, "game_discrepancies.yaml")])
        self.assertEqual(check(good[0]), [])
        self.assertTrue(wrong[0]["discrepancy"])
        self.assertNotEqual(check(wrong[0]), [])


class TestUndoLesson(LabCase):
    def test_undo_forgets_only_the_last_lesson(self):
        first = TestLearningToRead.FIRST
        lab.checkpoint()
        lab.teach(make_frame(first), labels(first))
        self.assertIn("Ant", lab.Memory.load().items["pet"])
        pigs_before = len(lab.Memory.load().items["pet"]["Pig"])

        ### A lesson with a mistake: the Ant typed as a Pig
        wrong = {"player": [pet("Pig", 2, 1)] + [None] * 4, "opponent": [None] * 5}
        lab.checkpoint()
        lab.teach(make_frame({"player": [pet("Ant", 2, 1)], "opponent": []}), wrong)
        self.assertTrue(lab.can_undo())
        lab.undo()
        self.assertFalse(lab.can_undo())
        ### The first lesson is kept, the mistaken one is gone
        seen = lab.read_frame(
            make_frame({"player": [pet("Ant", 2, 1)], "opponent": []})
        )
        self.assertEqual(seen["player"][0]["name"], "Ant")
        self.assertEqual(len(lab.Memory.load().items["pet"]["Pig"]), pigs_before)

    def test_undo_of_the_first_lesson_forgets_everything(self):
        first = TestLearningToRead.FIRST
        lab.checkpoint()
        lab.teach(make_frame(first), labels(first))
        lab.undo()
        self.assertEqual(lab.Memory.load().counts(), {})

    def test_nothing_to_undo(self):
        with self.assertRaises(ValueError):
            lab.undo()


class TestTeamsFromText(unittest.TestCase):
    TEXT = """
    before you: Ant 2/1, fish 3/4 L2 garlic, -, Pig 4/1
    before them: Beaver 5/6 meat bone; Duck 7/8 L3
    after you: Pig 4/1
    after them: none
    """

    def test_fills_every_slot(self):
        t = lab.parse_teams(self.TEXT)
        you = t["start"]["player"]
        self.assertEqual(you[1], pet("Fish", 3, 4, 2, "Garlic"))
        self.assertEqual(you[2], {"empty": True})
        self.assertEqual(you[3]["name"], "Pig")
        self.assertEqual(you[4], {"empty": True})
        self.assertEqual(t["start"]["opponent"][0]["perk"], "Meat Bone")
        self.assertEqual(t["end"]["opponent"], [{"empty": True}] * 5)

    def test_missing_frame_is_left_alone(self):
        t = lab.parse_teams("after you: Pig 4/1")
        self.assertEqual(set(t), {"end"})
        self.assertEqual(t["end"]["opponent"], [{"empty": True}] * 5)

    def test_rows_teach_like_typed_rows(self):
        first = TestLearningToRead.FIRST
        typed = lab.parse_teams(
            "before you: Ant 2/1, Fish 3/4 L2 Garlic, Pig 4/1\n"
            "before them: Beaver 5/6 Melon, Duck 7/8 L3, Otter 9/10"
        )["start"]
        self.assertEqual(typed, labels(first))

    def test_mistakes_name_the_line(self):
        for text, words in [
            ("before you: Antt 2/1", "line 1: unknown pet"),
            ("before you: Ant 2/1\nbefore us: Pig 1/1", "line 2"),
            ("during you: Ant 2/1", "line 1"),
            ("before you: Ant 2", "line 1: cannot read pet"),
            ("before you: " + ", ".join(["Ant 1/1"] * 6), "at most 5"),
            ("", "Nothing to fill in"),
        ]:
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, words):
                lab.parse_teams(text)


class TestEndLevels(unittest.TestCase):
    def test_survivor_keeps_its_start_level(self):
        start = {"player": [pet("Worm", 6, 11, 3), pet("Cat", 5, 6)], "opponent": []}
        end = {
            "player": [pet("Worm", 6, 14), pet("Zombie Cricket", 1, 1)],
            "opponent": [],
        }
        fixed = lab.with_start_levels(start, end)
        self.assertEqual(fixed["player"][0]["level"], 3)
        self.assertEqual(fixed["player"][1]["level"], 1)
        ### The typed rows are not changed
        self.assertEqual(end["player"][0]["level"], 1)

    def test_two_levels_of_one_pet_are_left_alone(self):
        start = {"player": [pet("Ant", 2, 1), pet("Ant", 4, 3, 2)], "opponent": []}
        end = {"player": [pet("Ant", 4, 3)], "opponent": []}
        self.assertEqual(lab.with_start_levels(start, end)["player"][0]["level"], 1)

    def test_run_matches_with_the_level_left_at_1(self):
        start = {"player": [pet("Fish", 2, 3, 2)], "opponent": [pet("Ant", 1, 1)]}
        end = {"player": [pet("Fish", 2, 2)], "opponent": []}
        out = lab.run(start, end, runs=5)
        self.assertTrue(out["match"]["matches"])
        self.assertEqual(out["game"]["player"], ["Fish 2/2 L2"])


if __name__ == "__main__":
    unittest.main()
