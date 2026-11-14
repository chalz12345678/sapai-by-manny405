import json
import unittest

from sapai.data import data
from sapai.foods import Food
from sapai.pets import Pet

from gui import game


def setup(team_names, seed=1):
    m = game.Match("human", "greedy", seed=seed)
    p = m.seats[0].player
    for n in team_names:
        pet = Pet(n)
        pet.player, pet.shop = p, p.shop
        p.team.append(pet)
        pet.team = p.team
    for s in p.shop.slots:
        if s.slot_type == "food":
            s.obj = Food("canned-food")
            s.obj.shop, s.obj.player, s.cost = p.shop, p, 3
            return m, p, p.shop.slots.index(s)


def bonus(m):
    out = []
    for x in m.state()["seats"][0]["shop"]:
        if x["kind"] == "food" or x["frozen"]:
            continue
        b = data["pets"]["pet-" + x["name"].replace(" ", "-")]
        out.append((x["attack"] - b["baseAttack"], x["health"] - b["baseHealth"]))
    return out


class TestCannedFood(unittest.TestCase):
    def test_current_and_future_shop_pets_get_one(self):
        m, p, ci = setup(["ant"])
        self.assertTrue(json.loads(m.do("buy_food_untargeted", ci))["ok"])
        self.assertTrue(all(b == (1, 1) for b in bonus(m)), bonus(m))
        p.gold = 10
        m.do("roll")
        self.assertTrue(all(b == (1, 1) for b in bonus(m)), bonus(m))
        m.do("end_turn")
        self.assertTrue(all(b == (1, 1) for b in bonus(m)), bonus(m))

    def test_team_unchanged(self):
        m, p, ci = setup(["ant"])
        m.do("buy_food_untargeted", ci)
        ant = m.state()["seats"][0]["team"][0]
        self.assertEqual((ant["attack"], ant["health"]), (2, 2))

    def test_rabbit_ignores_shop_pets(self):
        m, p, ci = setup(["rabbit", "ant"])
        m.do("buy_food_untargeted", ci)
        self.assertTrue(all(b == (1, 1) for b in bonus(m)), bonus(m))


if __name__ == "__main__":
    unittest.main()
