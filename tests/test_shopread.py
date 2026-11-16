"""
Shop Reader tests on synthetic shop screens (the same drawing helpers as the
Battle Lab tests): it learns from one screen, reads another, builds the
engine's player and asks a model for its top moves.
"""

import os
import tempfile
import unittest
from unittest import mock

import numpy as np

from gui import lab, shopread
from tests import test_lab as T

LAYOUT = {
    "team": [[0.50 - 0.08 * i, 0.35] for i in range(5)],
    "shop": [[0.14 + 0.09 * i, 0.70] for i in range(7)],
    "slot": [0.06, 0.1],
    "shop_slot": [0.06, 0.1],
    "attack": [-0.018, 0.085, 0.028, 0.045],
    "health": [0.018, 0.085, 0.028, 0.045],
    "perk": [0.02, -0.075, 0.02, 0.03],
    "level": [-0.015, -0.075, 0.03, 0.03],
    "shop_attack": [-0.018, 0.085, 0.028, 0.045],
    "shop_health": [0.018, 0.085, 0.028, 0.045],
    "gold": [0.05, 0.05, 0.05, 0.06],
    "lives": [0.12, 0.05, 0.05, 0.06],
    "turn": [0.19, 0.05, 0.05, 0.06],
    "wins": [0.26, 0.05, 0.05, 0.06],
}
ICE = np.array([150, 220, 255])


def pixels(box):
    cx, cy, bw, bh = box
    x0, y0 = int((cx - bw / 2) * T.W), int((cy - bh / 2) * T.H)
    x1, y1 = int((cx + bw / 2) * T.W), int((cy + bh / 2) * T.H)
    return x0, y0, x1, y1


def make_screen(team, shop, counters):
    """team: 5 pets or None, front first; shop: slots or None, left to right"""
    frame = np.full((T.H, T.W, 3), (60, 100, 70), dtype=np.float64)
    for i, p in enumerate(team):
        if p is None:
            continue
        x0, y0, x1, y1 = pixels(shopread.box(LAYOUT, "team", i, "pet"))
        frame[y0:y1, x0:x1] = T.pet_picture(p["name"], y1 - y0, x1 - x0)
        T.draw_number(frame, shopread.box(LAYOUT, "team", i, "attack"), p["attack"])
        T.draw_number(frame, shopread.box(LAYOUT, "team", i, "health"), p["health"])
        T.fill(frame, shopread.box(LAYOUT, "team", i, "perk"), T.PERK_COLORS[p["perk"]])
        T.fill(
            frame,
            shopread.box(LAYOUT, "team", i, "level"),
            T.LEVEL_COLORS[str(p["level"])],
        )
    for i, s in enumerate(shop):
        if s is None:
            continue
        x0, y0, x1, y1 = pixels(shopread.box(LAYOUT, "shop", i, "pet"))
        pic = T.pet_picture(s["name"], y1 - y0, x1 - x0)
        if s.get("frozen"):
            pic = pic * 0.4 + ICE * 0.6
        frame[y0:y1, x0:x1] = pic
        if "attack" in s:
            T.draw_number(
                frame, shopread.box(LAYOUT, "shop", i, "shop_attack"), s["attack"]
            )
            T.draw_number(
                frame, shopread.box(LAYOUT, "shop", i, "shop_health"), s["health"]
            )
    for key, value in counters.items():
        T.draw_number(frame, tuple(LAYOUT[key]), value)
    return frame.astype(np.uint8)


def pet(name, a, h, level=1, perk="none"):
    return {"name": name, "attack": a, "health": h, "level": level, "perk": perk}


def shop_pet(name, a, h, frozen=False):
    return {"name": name, "attack": a, "health": h, "frozen": frozen}


def food(name, frozen=False):
    return {"name": name, "frozen": frozen}


def labels(team, shop, counters):
    return {
        "team": [p if p else {"empty": True} for p in team],
        "shop": [s if s else {"empty": True} for s in shop],
        "counters": counters,
    }


FIRST = (
    [pet("Ant", 2, 1), pet("Fish", 3, 4, 2, "Garlic"), pet("Pig", 6, 9), None, None],
    [
        shop_pet("Beaver", 3, 2),
        shop_pet("Otter", 1, 3, frozen=True),
        shop_pet("Duck", 7, 8),
        food("Apple"),
        food("Honey", frozen=True),
        None,
        None,
    ],
    {"gold": 10, "lives": 5, "turn": 3, "wins": 1},
)
### The same kinds of things in new places, with new stats and the frozen
###   ones swapped: what the reader must cope with after one lesson
SECOND = (
    [pet("Pig", 5, 6, perk="Garlic"), pet("Ant", 7, 8, 2), None, None, None],
    [
        shop_pet("Otter", 2, 4),
        shop_pet("Beaver", 6, 9, frozen=True),
        food("Honey"),
        food("Apple", frozen=True),
        shop_pet("Duck", 1, 3),
        None,
        None,
    ],
    {"gold": 7, "lives": 4, "turn": 6, "wins": 2},
)


class ShopCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        patcher = mock.patch.object(lab, "DATA_DIR", self.tmp.name)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.tmp.cleanup)
        shopread.save_layout(LAYOUT)
        shopread.teach_shop(make_screen(*FIRST), labels(*FIRST))


class TestReadingTheShop(ShopCase):
    def test_reads_a_new_screen_after_one_lesson(self):
        seen = shopread.read_shop(make_screen(*SECOND))
        team, shop, counters = SECOND
        self.assertEqual(seen["counters"], counters)
        for i, want in enumerate(team):
            got = seen["team"][i]
            with self.subTest(team=i):
                if want is None:
                    self.assertTrue(got["empty"])
                    continue
                for k in ("name", "attack", "health", "perk"):
                    self.assertEqual(got[k], want[k])
                self.assertEqual(got["level"], str(want["level"]))
        for i, want in enumerate(shop):
            got = seen["shop"][i]
            with self.subTest(shop=i):
                if want is None:
                    self.assertEqual(got["kind"], "empty")
                    continue
                self.assertEqual(got["name"], want["name"])
                self.assertEqual(got["frozen"], want["frozen"])
                if "attack" in want:
                    self.assertEqual(got["kind"], "pet")
                    self.assertEqual(
                        (got["attack"], got["health"]), (want["attack"], want["health"])
                    )
                else:
                    self.assertEqual(got["kind"], "food")
        self.assertEqual(seen["unsure"], [])

    def test_layout_is_checked(self):
        with self.assertRaises(ValueError):
            shopread.save_layout({"team": [[0.1, 0.1]]})

    def test_capture_round_trip_is_shrunk(self):
        big = np.zeros((1800, 2880, 3), dtype=np.uint8)
        shopread.save_capture(big)
        self.assertEqual(shopread.load_capture().shape, (1000, 1600, 3))


class TestThePlayer(ShopCase):
    def test_screen_becomes_the_engines_player(self):
        state = labels(
            [pet("Otter", 5, 6, perk="Melon"), pet("Ant", 7, 8, 3), None, None, None],
            [
                shop_pet("Pig", 2, 4, frozen=True),
                food("Honey"),
                shop_pet("Fish", 6, 9),
                food("Apple", frozen=True),
            ],
            {"gold": 7, "lives": 4, "turn": 6, "wins": 2},
        )
        player, screen_slots = shopread.to_player(state)
        self.assertEqual(
            (player.gold, player.lives, player.turn, player.wins), (7, 4, 6, 2)
        )
        team = [s.pet for s in player.team if not s.empty]
        self.assertEqual([p.name for p in team], ["pet-otter", "pet-ant"])
        self.assertEqual((team[1].attack, team[1].health, team[1].level), (7, 8, 3))
        self.assertEqual(team[0].status, "status-melon-armor")
        kinds = [(s.slot_type, s.obj.name, s.frozen) for s in player.shop.slots]
        ### Pets first, as the engine keeps them, each with its screen number
        self.assertEqual(
            kinds,
            [
                ("pet", "pet-pig", True),
                ("pet", "pet-fish", False),
                ("food", "food-honey", False),
                ("food", "food-apple", True),
            ],
        )
        self.assertEqual(screen_slots, [1, 3, 2, 4])

    def test_moves_use_screen_slot_numbers(self):
        from sapai.rl import env as E

        ### Food on the left of the screen still keeps its screen number
        state = labels(
            [pet("Ant", 2, 1), None, None, None, None],
            [food("Apple"), None, shop_pet("Pig", 2, 4)],
            {"gold": 10, "lives": 5, "turn": 1, "wins": 0},
        )
        player, slots = shopread.to_player(state)
        buy = E.ACTION_START["buy_pet"] + 0
        feed = E.ACTION_START["buy_food"] + 1 * E.MAX_TEAM + 0
        self.assertEqual(
            shopread.move_text(player, buy, slots), "Buy Pig (shop slot 3)"
        )
        self.assertEqual(
            shopread.move_text(player, feed, slots),
            "Feed Apple (shop slot 1) to Ant (front pet)",
        )


@unittest.skipUnless(
    os.path.exists(os.path.join(shopread.MODELS_DIR, "token_22.5M.npz")),
    "exported models not present",
)
class TestSuggestions(ShopCase):
    def test_top_three_moves_with_probabilities(self):
        out = shopread.advise(make_screen(*SECOND), "token_22.5M")
        moves = out["moves"]
        self.assertEqual(len(moves), 3)
        probs = [m["prob"] for m in moves]
        self.assertEqual(probs, sorted(probs, reverse=True))
        self.assertTrue(0 < sum(probs) <= 1.0001)
        for m in moves:
            self.assertIsInstance(m["text"], str)
        self.assertGreater(out["legal"], 3)

    def test_models_are_listed_best_first(self):
        names = [n for n, _ in shopread.list_models()]
        self.assertEqual(names[0], "token_22.5M")


class TestOverlayParts(ShopCase):
    """The overlay's window needs a screen; what it shows is worked out here"""

    def test_percentages(self):
        from gui import overlay

        self.assertEqual(overlay.pct(0.62), "62.0%")
        self.assertEqual(overlay.pct(0.00001), "<0.1%")
        self.assertEqual(overlay.pct(1.0), ">99.9%")

    def test_status_line(self):
        from gui import overlay

        line = overlay.status_line({"turn": 6, "gold": 7, "lives": 4, "wins": None})
        self.assertEqual(line, "Turn 6 · 7 gold · 4 lives · ? trophies")

    @unittest.skipUnless(
        os.path.exists(os.path.join(shopread.MODELS_DIR, "token_22.5M.npz")),
        "exported models not present",
    )
    def test_advisor_reads_saves_the_capture_and_learns_new_lessons(self):
        from gui import overlay

        advisor = overlay.Advisor("token_22.5M")
        screen = make_screen(*SECOND)
        shown = overlay.summary(advisor.advise(screen))
        self.assertEqual(shown["status"], "Turn 6 · 7 gold · 4 lives · 2 trophies")
        self.assertEqual(len(shown["moves"]), 3)
        self.assertEqual(shown["warning"], "")
        np.testing.assert_array_equal(shopread.load_capture(), screen)
        ### A pet it has never seen is flagged until it is taught
        team = [pet("Beaver", 5, 6), None, None, None, None]
        new = make_screen(team, SECOND[1], SECOND[2])
        self.assertIn("team 1 name", overlay.summary(advisor.advise(new))["warning"])
        os.utime(lab.memory_path(), (1, 1))
        shopread.teach_shop(new, labels(team, SECOND[1], SECOND[2]))
        self.assertEqual(overlay.summary(advisor.advise(new))["warning"], "")

    def test_not_set_up_yet(self):
        from gui import overlay

        os.remove(shopread.layout_path())
        with self.assertRaisesRegex(ValueError, "Not set up yet"):
            overlay.Advisor("token_22.5M").advise(make_screen(*SECOND))


if __name__ == "__main__":
    unittest.main()
