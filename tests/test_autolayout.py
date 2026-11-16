"""
Automatic layout on synthetic battle frames drawn like the game: a red heart
(split by a white number) with a dark grey stone to its left under every
pet, pets 89 px apart, the front pets either side of the middle, and red
hats above some pets that must not be taken for hearts.
"""

import unittest

import numpy as np

from gui import autolayout as A
from gui import lab

W, H = 952, 561
S = 89
HY = 406
MID = W / 2
GAP = 0.70 * S


def pair_x(side, k):
    return MID + (-1 if side == "player" else 1) * (GAP + k * S)


def draw(slots, hats=()):
    """slots: (side, k) pairs that hold a pet; hats: (side, k) with a red hat"""
    img = np.full((H, W, 3), (40, 120, 60), dtype=np.uint8)
    for side, k in slots:
        hx = int(pair_x(side, k) - A.PAIR_DX * S)
        sx = int(pair_x(side, k) + A.PAIR_DX * S)
        img[HY - 14 : HY + 14, hx - 14 : hx + 14] = (225, 20, 25)  # heart
        img[HY - 10 : HY + 10, hx - 2 : hx + 3] = 255  # its number splits it
        img[HY - 14 : HY + 14, sx - 14 : sx + 14] = (80, 80, 80)  # stone
        img[HY - 10 : HY + 10, sx - 2 : sx + 3] = 255
    for side, k in hats:
        x, y = int(pair_x(side, k)), HY - 105
        img[y - 14 : y + 14, x - 14 : x + 14] = (230, 25, 30)
    return img


class TestAutoLayout(unittest.TestCase):
    def check(self, layout):
        for side in lab.SIDES:
            for k in range(5):
                x, _ = layout[side][k]
                self.assertAlmostEqual(x * W, pair_x(side, k), delta=2)
        ### The health box sits on the heart
        cx, cy, _, _ = lab.part_box(layout, "opponent", 0, "health")
        self.assertAlmostEqual(cx * W, pair_x("opponent", 0) - A.PAIR_DX * S, delta=2)
        self.assertAlmostEqual(cy * H, HY, delta=2)

    def test_full_teams(self):
        slots = [(s, k) for s in lab.SIDES for k in range(5)]
        layout, report = A.battle_layout(draw(slots))
        self.check(layout)
        self.assertEqual(report["hearts"], 10)

    def test_missing_pets_are_filled_in(self):
        ### Two pets a side, not at the front: the grid still places all ten
        layout, report = A.battle_layout(
            draw([("player", 2), ("player", 4), ("opponent", 1), ("opponent", 3)])
        )
        self.check(layout)
        self.assertEqual(report["found"], {"player": [2, 4], "opponent": [1, 3]})

    def test_red_hats_are_not_hearts(self):
        slots = [(s, k) for s in lab.SIDES for k in range(3)]
        hats = [("opponent", k) for k in range(5)]
        layout, report = A.battle_layout(draw(slots, hats))
        self.check(layout)
        self.assertEqual(report["hearts"], 6)

    def test_level_box_per_side_and_flipped_opponent(self):
        layout, _ = A.battle_layout(draw([(s, k) for s in lab.SIDES for k in range(5)]))
        mine = lab.part_box(layout, "player", 0, "level")[0] * W - pair_x("player", 0)
        theirs = lab.part_box(layout, "opponent", 0, "level")[0] * W - pair_x(
            "opponent", 0
        )
        self.assertLess(mine, 0)
        self.assertGreater(theirs, 0)
        self.assertTrue(layout["flip_opponent"])
        self.assertEqual(lab.check_layout(layout), [])

    def test_no_pets(self):
        with self.assertRaises(ValueError):
            A.battle_layout(draw([]))


if __name__ == "__main__":
    unittest.main()
