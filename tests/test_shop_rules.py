"""Shop rules confirmed in the game (docs/RULES.md, the Rule Checklist)"""

import unittest

import numpy as np

from sapai.actions import ACTION_START, action_mask
from sapai.battle import Battle
from sapai.data import RULES
from sapai.pets import Pet
from sapai.player import Player
from sapai.shop import Shop, ShopSlot, get_shop_rules
from sapai.teams import Team


def names(shop):
    return [s.obj.name.split("-", 1)[1] for s in shop]


class TestShopSlots(unittest.TestCase):
    def test_slots_by_turn(self):
        ### 3 pets and 1 food until turn 5, then 4 and 2, then 5 and 2 from turn 9
        expected = {1: (3, 1), 4: (3, 1), 5: (4, 2), 8: (4, 2), 9: (5, 2), 11: (5, 2)}
        for turn, (pets, foods) in expected.items():
            rules = get_shop_rules(turn)
            self.assertEqual((rules[0], rules[1]), (pets, foods), turn)

    def test_nine_slots_in_all(self):
        self.assertEqual(RULES["max_shop_slots"], 9)


class TestFullShop(unittest.TestCase):
    def full_shop(self):
        ### 7 pets and 2 foods
        return Shop(
            [
                "ant",
                "beaver",
                "cricket",
                "duck",
                "fish",
                "horse",
                "pig",
                "apple",
                "honey",
            ]
        )

    def test_extra_food_replaces_the_rightmost_pet(self):
        shop = self.full_shop()
        shop.append(ShopSlot("food-bread-crumbs"))
        self.assertEqual(len(shop), 9)
        self.assertNotIn("pig", names(shop))
        self.assertEqual(names(shop)[-1], "bread-crumbs")
        self.assertIn("apple", names(shop))
        self.assertIn("honey", names(shop))

    def test_frozen_pets_are_protected(self):
        shop = self.full_shop()
        shop.freeze(6)  # Pig
        shop.append(ShopSlot("food-bread-crumbs"))
        self.assertIn("pig", names(shop))
        self.assertNotIn("horse", names(shop))

    def test_with_every_pet_frozen_a_food_goes(self):
        shop = self.full_shop()
        for i in range(7):
            shop.freeze(i)
        shop.append(ShopSlot("food-bread-crumbs"))
        self.assertEqual(len(shop), 9)
        self.assertEqual(
            names(shop)[:7],
            ["ant", "beaver", "cricket", "duck", "fish", "horse", "pig"],
        )
        self.assertIn("bread-crumbs", names(shop))
        self.assertEqual(sum(n in ("apple", "honey") for n in names(shop)), 1)

    def test_with_everything_frozen_the_new_item_is_lost(self):
        shop = self.full_shop()
        for i in range(9):
            shop.freeze(i)
        shop.append(ShopSlot("food-bread-crumbs"))
        self.assertNotIn("bread-crumbs", names(shop))
        self.assertEqual(len(shop), 9)


class TestLevelUpChoice(unittest.TestCase):
    def levelled(self):
        player = Player(shop=Shop(["fish", "fish"]), team=Team([Pet("fish")]))
        player.gold = 30
        player.buy_combine(1, 0)
        player.buy_combine(0, 0)
        return player

    def test_two_pets_from_the_next_tier(self):
        player = self.levelled()
        levelup = [s for s in player.shop if s.slot_type == "levelup"]
        self.assertEqual(len(levelup), 2)
        self.assertEqual(len({s.group for s in levelup}), 1)
        self.assertEqual({s.obj.tier for s in levelup}, {2})

    def test_buying_one_removes_the_other(self):
        player = self.levelled()
        player.buy_pet(0)
        self.assertEqual([s for s in player.shop if s.slot_type == "levelup"], [])
        self.assertEqual(len(player.team), 2)

    def test_only_one_can_be_frozen(self):
        player = self.levelled()
        player.freeze(0)
        with self.assertRaises(Exception):
            player.freeze(1)
        mask = action_mask(player)
        self.assertTrue(mask[ACTION_START["freeze"] + 0])  # unfreeze the first
        self.assertFalse(mask[ACTION_START["freeze"] + 1])

    def test_frozen_one_stays_and_the_other_goes(self):
        player = Player()
        player.shop.levelup()
        first = [i for i, s in enumerate(player.shop) if s.slot_type == "levelup"][0]
        kept = player.shop[first].obj.name
        player.freeze(first)
        for step in ("roll", "next turn"):
            with self.subTest(step=step):
                if step == "roll":
                    player.gold = 10
                    player.roll()
                else:
                    player.end_turn()
                    player.start_turn()
                levelup = [s for s in player.shop if s.slot_type == "levelup"]
                self.assertEqual([s.obj.name for s in levelup], [kept])
                self.assertTrue(levelup[0].frozen)
                self.assertEqual([s for s in player.shop if s.slot_type == "none"], [])

    def test_pair_survives_saving(self):
        player = self.levelled()
        restored = Player.from_state(player.state)
        levelup = [s for s in restored.shop if s.slot_type == "levelup"]
        self.assertEqual(len(levelup), 2)
        self.assertEqual(len({s.group for s in levelup}), 1)


class TestPerkDamageAboveTheCap(unittest.TestCase):
    def test_steak_on_a_50_attack_pet_hits_for_70(self):
        p = Pet("pig")
        p._attack = 50
        p.status = "status-steak-attack"
        self.assertEqual(p.attack, 70)
        self.assertEqual(p.attack_stat, 50)

    def test_it_breaks_through_melon(self):
        a = Pet("pig")
        a._attack, a._health = 50, 50
        a.status = "status-steak-attack"
        b = Pet("pig")
        b._attack, b._health = 1, 50
        b.status = "status-melon-armor"
        np.random.seed(0)
        battle = Battle(Team([a]), Team([b]))
        self.assertEqual(battle.battle(), 0)


if __name__ == "__main__":
    unittest.main()


class TestCowMilk(unittest.TestCase):
    def milk(self, shop):
        return [s for s in shop if s.slot_type == "food"]

    def test_two_free_milks_replace_all_food_even_frozen(self):
        player = Player(
            shop=Shop(["cow", "ant", "apple", "honey"]), team=Team([Pet("pig")])
        )
        player.freeze(2)
        player.freeze(3)
        player.buy_pet(0)
        milk = self.milk(player.shop)
        self.assertEqual([s.obj.name for s in milk], ["food-milk", "food-milk"])
        self.assertEqual([s.cost for s in milk], [0, 0])

    def test_milk_level_follows_the_cow(self):
        for level, name in ((2, "food-better-milk"), (3, "food-best-milk")):
            cow = Pet("cow")
            cow.level = level
            shop = Shop([cow, "apple"])
            player = Player(shop=shop, team=Team([Pet("pig")]))
            player.buy_pet(0)
            self.assertEqual([s.obj.name for s in self.milk(player.shop)], [name, name])

    def test_milks_show_on_a_full_shop_of_frozen_pets(self):
        shop = Shop(
            [
                "cow",
                "ant",
                "fish",
                "pig",
                "duck",
                "beaver",
                "horse",
                "otter",
                "mosquito",
            ]
        )
        player = Player(shop=shop, team=Team([Pet("pig")]))
        for i in range(1, 9):
            player.freeze(i)
        player.buy_pet(0)
        self.assertEqual(len(self.milk(player.shop)), 2)
        self.assertEqual(len(player.shop), 9)
        self.assertNotIn("none", [s.slot_type for s in player.shop])
