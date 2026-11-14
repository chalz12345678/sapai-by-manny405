import unittest

import numpy as np

from sapai.pets import Pet
from sapai.player import Player


def make_player(names):
    np.random.seed(1)
    p = Player()
    for name in names:
        pet = Pet(name)
        pet.player = p
        pet.shop = p.shop
        p.team.append(pet)
    for slot in p.team:
        if not slot.empty:
            slot.pet.team = p.team
    return p


def shop_health(p):
    return [s.obj.health for s in p.shop.slots if s.slot_type == "pet"]


class TestSellTriggers(unittest.TestCase):
    def test_duck_alone(self):
        p = make_player(["duck"])
        before = shop_health(p)
        p.sell(0)
        self.assertEqual(shop_health(p), [h + 1 for h in before])

    def test_duck_behind_other_sell_pets(self):
        """A Sell pet in front of the Duck must not swallow the Duck's ability"""
        for front in ["pig", "beaver", "pigeon"]:
            p = make_player([front, "duck"])
            before = shop_health(p)
            p.sell(1)
            self.assertEqual(shop_health(p), [h + 1 for h in before], front)
            self.assertEqual(len(p.team), 1)

    def test_pig_behind_duck(self):
        p = make_player(["duck", "pig"])
        gold = p.gold
        p.sell(1)
        ### Pig sells for 1 and its ability gives 1 more
        self.assertEqual(p.gold, gold + 2)
        self.assertEqual(len(p.team), 1)

    def test_selling_non_sell_pet_triggers_nothing(self):
        p = make_player(["duck", "pig", "ant"])
        before, gold = shop_health(p), p.gold
        p.sell(2)
        self.assertEqual(shop_health(p), before)
        self.assertEqual(p.gold, gold + 1)
        self.assertEqual(len(p.team), 2)


if __name__ == "__main__":
    unittest.main()
