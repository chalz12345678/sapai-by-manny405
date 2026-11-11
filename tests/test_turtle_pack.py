# %%
import os
import re
import unittest

import numpy as np

from sapai import *
from sapai.data import data
from sapai.shop import get_shop_rules

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

YAML_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data", "turtle_pack_pets.yaml"
)


def filled(team):
    return [x for x in team if not x.empty]


def slug(name):
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


@unittest.skipIf(yaml is None, "pyyaml is required to compare against the YAML")
class TestTurtlePackData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(YAML_PATH) as f:
            cls.y = yaml.safe_load(f)

    def test_exactly_turtle_pack_pets(self):
        shop_pets = {k for k, v in data["pets"].items() if type(v["tier"]) == int}
        yaml_pets = {"pet-" + slug(p["name"]) for p in self.y["pets"]}
        self.assertEqual(len(yaml_pets), 60)
        self.assertEqual(shop_pets, yaml_pets)

    def test_pet_stats(self):
        for p in self.y["pets"]:
            fd = data["pets"]["pet-" + slug(p["name"])]
            self.assertEqual(fd["tier"], p["tier"], p["name"])
            self.assertEqual(fd["baseAttack"], p["attack"], p["name"])
            self.assertEqual(fd["baseHealth"], p["health"], p["name"])
            self.assertEqual(fd["packs"], ["StandardPack"])
            for level in [1, 2, 3]:
                self.assertIn(
                    p["ability"][f"L{level}"],
                    fd[f"level{level}Ability"]["description"],
                )

    def test_foods(self):
        for f in self.y["foods"]:
            fd = data["foods"]["food-" + slug(f["name"])]
            self.assertEqual(fd["tier"], f["tier"], f["name"])
            self.assertEqual(fd["cost"], f["cost"], f["name"])
            self.assertEqual(Food(fd["id"]).cost, f["cost"])
        for f in self.y["pet_created_foods"]:
            fd = data["foods"]["food-" + slug(f["name"])]
            ### Never stocked in the shop on their own
            self.assertEqual(fd["tier"], "Summoned")
            self.assertEqual(fd["cost"], f["cost"])
        shop_foods = {k for k, v in data["foods"].items() if type(v["tier"]) == int}
        self.assertEqual(
            shop_foods, {"food-" + slug(f["name"]) for f in self.y["foods"]}
        )

    def test_shop_only_rolls_turtle_pack(self):
        for turn in range(1, 12):
            rules = get_shop_rules(turn)
            tier_avail = rules[2]
            avail_pets, avail_foods, pp, fp = rules[4], rules[5], rules[6], rules[7]
            expected_pets = [
                "pet-" + slug(p["name"])
                for p in self.y["pets"]
                if p["tier"] <= tier_avail
            ]
            expected_foods = [
                "food-" + slug(f["name"])
                for f in self.y["foods"]
                if f["tier"] <= tier_avail
            ]
            self.assertEqual(sorted(avail_pets), sorted(expected_pets))
            self.assertEqual(sorted(avail_foods), sorted(expected_foods))
            ### Every pet and food available on a turn is equally likely
            self.assertTrue(np.allclose(pp, 1 / len(expected_pets)))
            self.assertTrue(np.allclose(fp, 1 / len(expected_foods)))

    def test_no_expansion_pack(self):
        with self.assertRaises(Exception):
            Shop(pack="ExpansionPack1")


class TestTurtlePackAbilities(unittest.TestCase):
    def test_pigeon_stocks_bread_crumbs(self):
        for level in [1, 3]:
            pigeon = Pet("pigeon")
            pigeon.level = level
            player = Player(team=[pigeon])
            nslots = len(player.shop)
            player.sell(0)
            crumbs = [x for x in player.shop if x.obj.name == "food-bread-crumbs"]
            self.assertEqual(len(crumbs), level)
            self.assertEqual(len(player.shop), nslots + level)
            self.assertEqual(crumbs[0].cost, 0)

        ### Bread crumbs give +1 attack
        player = Player(shop=Shop(["bread-crumbs"]), team=["ant"])
        player.buy_food(0, 0)
        self.assertEqual(player.team[0].attack, 3)
        self.assertEqual(player.team[0].health, 2)
        self.assertEqual(player.gold, 10)

    def test_worm_stocks_apple(self):
        for level, food in [(1, "food-apple"), (2, "food-better-apple")]:
            worm = Pet("worm")
            worm.level = level
            player = Player(team=[worm])
            player.start_turn()
            self.assertEqual(player.shop[-1].obj.name, food)
            self.assertEqual(player.shop[-1].cost, 2)
            player.buy_food(len(player.shop) - 1, 0)
            self.assertEqual(player.gold, 10 - 2)
            self.assertEqual(player.team[0].attack, 1 + level)
            self.assertEqual(player.team[0].health, 4 + level)

    def test_cow_milk(self):
        for level, food, attack, health in [
            (1, "food-milk", 1, 2),
            (3, "food-best-milk", 3, 6),
        ]:
            cow = Pet("cow")
            cow.level = level
            player = Player(shop=Shop([cow, "apple", "pear"]))
            player.buy_pet(0)
            self.assertEqual([x.obj.name for x in player.shop], [food, food])
            self.assertEqual([x.cost for x in player.shop], [0, 0])
            player.buy_food(0, 0)
            self.assertEqual(player.team[0].attack, 4 + attack)
            self.assertEqual(player.team[0].health, 6 + health)

    def test_cake_sell_value(self):
        player = Player(shop=Shop(["cake"]), team=["ant"])
        player.buy_food(0, 0)
        self.assertEqual(player.team[0].pet.status, "status-cake")
        player.end_turn()
        player.start_turn()
        player.end_turn()
        self.assertEqual(player.team[0].pet.sell_value_bonus, 2)
        gold = player.gold
        player.sell(0)
        self.assertEqual(player.gold, gold + 1 + 2)

    def test_sell_value_by_level(self):
        for level in [1, 2, 3]:
            ant = Pet("ant")
            ant.level = level
            player = Player(team=[ant])
            player.sell(0)
            self.assertEqual(player.gold, 10 + level)

        ### Pig's ability adds to the level sell value
        pig = Pet("pig")
        pig.level = 2
        player = Player(team=[pig])
        player.sell(0)
        self.assertEqual(player.gold, 10 + 2 + 2)

        ### Cake adds to the level sell value
        ant = Pet("ant")
        ant.level = 3
        ant.sell_value_bonus = 1
        player = Player(team=[ant])
        player.sell(0)
        self.assertEqual(player.gold, 10 + 3 + 1)

    def test_bread_health(self):
        player = Player(shop=Shop(["bread"]), team=["ant"])
        player.buy_food(0, 0)
        player.end_turn()
        self.assertEqual(player.team[0].health, 2 + 7)
        player.start_turn()
        self.assertEqual(player.team[0].health, 2)

    def test_meat_bone_and_steak(self):
        fish = Pet("fish")
        fish.status = "status-bone-attack"
        self.assertEqual(fish.attack, 2 + 3)
        fish.status = "status-steak-attack"
        self.assertEqual(fish.attack, 2 + 20)

    def test_scorpion_peanut(self):
        self.assertEqual(Pet("scorpion").status, "status-poison-attack")
        ### Peanut knocks out a pet with very high health
        enemy = Pet("mammoth")
        enemy._health = 50
        b = Battle(Team(["scorpion"]), Team([enemy]))
        self.assertEqual(b.battle(), 2)

    def test_rhino_double_damage_tier_1(self):
        t = Team(["rhino"], battle=True)
        enemy = Team(["ant", "dragon"], battle=True)
        t[0].pet.knockout_trigger(enemy)
        self.assertEqual(enemy[0].health, 2 - 8)

        t = Team(["rhino"], battle=True)
        enemy = Team(["dragon"], battle=True)
        t[0].pet.knockout_trigger(enemy)
        self.assertEqual(enemy[0].health, 8 - 4)

    def test_dolphin_triggers_per_level(self):
        t = Team(["dolphin"], battle=True)
        t[0].pet.level = 2
        enemy = Team(["ant", "fish", "mammoth"], battle=True)
        t[0].pet.sob_trigger(enemy)
        ### 4 damage to the lowest health enemy, two times
        self.assertEqual(enemy[0].health, 2 - 4)
        self.assertEqual(enemy[1].health, 3 - 4)
        self.assertEqual(enemy[2].health, 12)

    def test_crocodile_triggers_per_level(self):
        t = Team(["crocodile"], battle=True)
        t[0].pet.level = 3
        enemy = Team(["mammoth", "mammoth", "mammoth"], battle=True)
        t[0].pet.sob_trigger(enemy)
        ### 8 damage to the last enemy, three times. The last mammoth faints
        ###   after two hits, so the third hit goes to the next last enemy.
        self.assertEqual([x.health for x in filled(enemy)], [12, 4, -4])

    def test_tiger_repeats_at_tiger_level(self):
        mosquito = Pet("mosquito")
        tiger = Pet("tiger")
        tiger.level = 2
        t = Team([mosquito, tiger], battle=True)
        enemy = Team(["mammoth", "mammoth", "mammoth"], battle=True)
        mosquito.sob_trigger(enemy)
        ### Level 1 mosquito hits 1 enemy, then repeats as level 2 hitting 2
        total_health = sum([x.health for x in filled(enemy)])
        self.assertEqual(total_health, 3 * 12 - 3)

    def test_armadillo(self):
        t = Team(["armadillo", "fish"], battle=True)
        enemy = Team(["ant"], battle=True)
        t[0].pet.sob_trigger(enemy)
        self.assertEqual(t[0].health, 8 + 8)
        self.assertEqual(t[1].health, 3 + 8)
        self.assertEqual(enemy[0].health, 2 + 8)

    def test_penguin_level_2_friends(self):
        fish = Pet("fish")
        fish.level = 2
        penguin = Pet("penguin")
        penguin.level = 2
        t = Team([fish, "ant", penguin])
        penguin.sot_trigger()
        self.assertEqual((t[0].attack, t[0].health), (4, 5))
        self.assertEqual((t[1].attack, t[1].health), (2, 2))
        self.assertEqual((t[2].attack, t[2].health), (2, 3))

    def test_parrot_copies_at_parrot_level(self):
        parrot = Pet("parrot")
        parrot.level = 2
        t = Team(["mosquito", parrot])
        parrot.eot_trigger()
        self.assertEqual(parrot.ability["trigger"], "StartOfBattle")
        self.assertEqual(parrot.ability["effect"]["target"]["n"], 2)

    def test_rabbit_three_times_per_turn(self):
        player = Player(shop=Shop(["apple"] * 4), team=["rabbit", "fish"])
        player.gold = 20
        for _ in range(4):
            player.buy_food(0, 1)
        ### 4 apples, only 3 rabbit triggers
        self.assertEqual(player.team[1].attack, 2 + 4)
        self.assertEqual(player.team[1].health, 3 + 4 + 3)

        ### Resets next turn
        player.end_turn()
        player.start_turn()
        rabbit = player.team[0].pet
        activated, _, _ = rabbit.eats_shop_food_trigger(player.team[1].pet)
        self.assertTrue(activated)

    def test_cat_two_times_per_turn(self):
        player = Player(shop=Shop(["apple"] * 3), team=["cat"])
        for _ in range(3):
            player.buy_food(0, 0)
        ### Two apples doubled, third apple normal
        self.assertEqual(player.team[0].attack, 4 + 2 + 2 + 1)
        self.assertEqual(player.team[0].health, 5 + 2 + 2 + 1)

    def test_snail_after_loss(self):
        player = Player(team=["ant", "fish", "beaver", "snail"])
        player.end_turn()
        self.assertEqual([x.attack for x in filled(player.team)], [2, 2, 3, 2])

        player = Player(team=["ant", "fish", "beaver", "snail"])
        player.lf_winner = False
        player.end_turn()
        self.assertEqual([x.attack for x in filled(player.team)], [3, 3, 4, 2])

    def test_fish_level_up(self):
        ### Level 1 to 2: two friends +1/+1
        fish = Pet("fish")
        fish.experience = 1
        player = Player(shop=Shop(["fish"]), team=[fish, "beaver", "beaver"])
        player.buy_combine(0, 0)
        self.assertEqual(player.team[0].pet.level, 2)
        self.assertEqual((player.team[1].attack, player.team[1].health), (4, 3))
        self.assertEqual((player.team[2].attack, player.team[2].health), (4, 3))

        ### Level 2 to 3: two friends +2/+2
        fish = Pet("fish")
        fish.level = 2
        fish.experience = 2
        player = Player(shop=Shop(["fish"]), team=[fish, "beaver", "beaver"])
        player.buy_combine(0, 0)
        self.assertEqual(player.team[0].pet.level, 3)
        self.assertEqual((player.team[1].attack, player.team[1].health), (5, 4))
        self.assertEqual((player.team[2].attack, player.team[2].health), (5, 4))

    def test_ox_times_per_turn(self):
        for level, attack in [(1, 2), (2, 3)]:
            ox = Pet("ox")
            ox.level = level
            t = Team(["fish", "fish", ox], battle=True)
            ox.faint_trigger(t[1].pet, [0, 1])
            ox.faint_trigger(t[0].pet, [0, 0])
            self.assertEqual(ox.status, "status-melon-armor")
            self.assertEqual(ox.attack, attack)

    def test_wolverine_every_4_friend_hurt(self):
        t = Team(["ant", "ant", "wolverine"], battle=True)
        enemy = Team(["mammoth", "mammoth"], battle=True)
        wolverine = t[2].pet
        for _ in range(3):
            activated, _, _ = wolverine.friend_hurt_trigger(t[0].pet, enemy)
            self.assertFalse(activated)
        activated, _, _ = wolverine.friend_hurt_trigger(t[0].pet, enemy)
        self.assertTrue(activated)
        self.assertEqual([x.health for x in filled(enemy)], [9, 9])
        ### Wolverine getting hurt itself does not count
        activated, _, _ = wolverine.friend_hurt_trigger(wolverine, enemy)
        self.assertFalse(activated)

    def test_wolverine_in_battle(self):
        ### Both hedgehogs faint in the first attack, so each mammoth is hurt
        ###   twice. Wolverine triggers once after 4 hurt friends.
        team = [Pet("hedgehog"), "mammoth", "mammoth", "mammoth", "wolverine"]
        team[0]._attack = 1
        team[0]._health = 1
        enemy = Pet("hedgehog")
        enemy._attack = 1
        enemy._health = 1
        b = Battle(Team(team), Team([enemy]))
        b.battle()
        found = [
            x
            for x in b.battle_history["attack 0"]["phase_hurt_and_faint_aa"]
            if x[0] == "RemoveHealth"
        ]
        self.assertEqual(len(found), 1)

    def test_deer_bus(self):
        t = Team(["deer"], battle=True)
        t[0].pet.faint_trigger(t[0].pet, [0, 0])
        self.assertEqual(t[0].pet.name, "pet-bus")
        self.assertEqual((t[0].attack, t[0].health), (5, 3))
        self.assertEqual(t[0].pet.status, "status-splash-attack")

    def test_rat_level_2(self):
        rat = Pet("rat")
        rat.level = 2
        t = Team([rat], battle=True)
        enemy = Team(["fish"], battle=True)
        rat.faint_trigger(rat, [0, 0], enemy)
        names = [x.pet.name for x in filled(enemy)]
        self.assertEqual(names, ["pet-dirty-rat", "pet-dirty-rat", "pet-fish"])

    def test_rooster_level_2(self):
        rooster = Pet("rooster")
        rooster.level = 2
        t = Team([rooster], battle=True)
        rooster.faint_trigger(rooster, [0, 0])
        chicks = [x for x in filled(t) if x.pet.name == "pet-chick"]
        self.assertEqual(len(chicks), 2)
        self.assertEqual([(x.attack, x.health) for x in chicks], [(3, 1), (3, 1)])

    def test_monkey_front_most_friend(self):
        player = Player(team=["ant", "fish", "monkey"])
        player.end_turn()
        self.assertEqual((player.team[0].attack, player.team[0].health), (4, 4))

    def test_hedgehog_level_2(self):
        hedgehog = Pet("hedgehog")
        hedgehog.level = 2
        t = Team([hedgehog, "mammoth"], battle=True)
        hedgehog.faint_trigger(hedgehog, [0, 0], Team())
        self.assertEqual(t[1].health, 12 - 4)

    def test_skunk_level_3(self):
        skunk = Pet("skunk")
        skunk.level = 3
        t = Team([skunk], battle=True)
        enemy = Team(["mammoth"], battle=True)
        skunk.sob_trigger(enemy)
        ### 99% of health removed, leaving 1
        self.assertEqual(enemy[0].health, 1)

    def test_spider_summons_tier_3(self):
        state = np.random.RandomState(seed=5).get_state()
        spider = Pet("spider", seed_state=state)
        spider.level = 2
        t = Team([spider], battle=True)
        spider.faint_trigger(spider, [0, 0])
        summoned = t[0].pet
        self.assertEqual(summoned.tier, 3)
        self.assertEqual(summoned.level, 2)
        self.assertEqual((summoned.attack, summoned.health), (4, 4))

    ### Regression tests for crashes found by random battles

    def test_crab_without_living_friends(self):
        crab = Pet("crab")
        crab._health = 0
        t = Team([crab], battle=True)
        crab.sob_trigger(Team())
        self.assertEqual(crab.health, 0)

    def test_spider_summon_on_full_team(self):
        from sapai.effects import SummonRandomPet

        spider = Pet("spider")
        t = Team([spider, "ant", "ant", "ant", "ant"], battle=True)
        targets, possible = SummonRandomPet(spider, [0, 0], [t], te_idx=[0, 0])
        self.assertEqual(targets, [])

    def test_fainted_shark_does_not_gain(self):
        ant = Pet("ant")
        shark = Pet("shark")
        ant._health = 0
        shark._health = 0
        t = Team([ant, shark], battle=True)
        activated, _, _ = shark.faint_trigger(ant, [0, 0])
        self.assertFalse(activated)
        self.assertEqual(shark.health, 0)

    def test_summoned_pets_connected_next_turn(self):
        ### Bus summoned during the shop phase can use the shop next turn
        player = Player(shop=Shop(["sleeping-pill"]), team=["deer"])
        player.buy_food(0, 0)
        player.start_turn()
        self.assertIs(player.team[0].pet.shop, player.shop)
        self.assertIs(player.team[0].pet.player, player)


# %%
