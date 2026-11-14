"""
Audit of the tier 1 and tier 2 pets' shop abilities (buy, sell, level-up,
start of turn, end of turn, and faint in the shop) against their ability
text in data/turtle_pack.yaml. Expected values come from the text, not from
running the engine. Battle abilities are in tests/scenarios/tier1_2.yaml.
"""

import unittest

import numpy as np

from sapai.data import data
from sapai.pets import Pet
from sapai.player import Player
from sapai.shop import Shop
from sapai.teams import Team


def pet(name, level=1, attack=None, health=None):
    p = Pet(name)
    p.level = level
    if attack is not None:
        p._attack = attack
    if health is not None:
        p._health = health
    return p


def team_pets(player):
    return [s.pet for s in player.team if not s.empty]


def stats(p):
    return (p.attack, p.health)


class AuditCase(unittest.TestCase):
    def setUp(self):
        np.random.seed(1234)


class TestBeaver(AuditCase):
    """Sell: Give two random friends +1/+2/+3 attack."""

    def test_two_friends_both_get_attack(self):
        for level in [1, 2, 3]:
            fish, ant = pet("fish"), pet("ant")
            player = Player(team=Team([pet("beaver", level), fish, ant]))
            player.sell(0)
            self.assertEqual(stats(fish), (2 + level, 3), level)
            self.assertEqual(stats(ant), (2 + level, 2), level)
            self.assertEqual(len(team_pets(player)), 2)

    def test_one_friend(self):
        fish = pet("fish")
        player = Player(team=Team([pet("beaver", 2), fish]))
        player.sell(0)
        self.assertEqual(stats(fish), (4, 3))

    def test_exactly_two_of_three_friends(self):
        for seed in range(10):
            np.random.seed(seed)
            friends = [pet("fish"), pet("fish"), pet("fish")]
            player = Player(team=Team([pet("beaver")] + friends))
            player.sell(0)
            gained = sorted(f.attack - 2 for f in friends)
            self.assertEqual(gained, [0, 1, 1])
            self.assertTrue(all(f.health == 3 for f in friends))

    def test_alone(self):
        player = Player(team=Team([pet("beaver")]))
        player.sell(0)
        self.assertEqual(player.gold, 11)
        self.assertEqual(team_pets(player), [])


class TestDuck(AuditCase):
    """Sell: Give shop pets +1/+2/+3 health."""

    def test_shop_pets_gain_health(self):
        for level in [1, 2, 3]:
            fish = pet("fish")
            player = Player(
                shop=Shop(["ant", "otter", "apple"]),
                team=Team([pet("duck", level), fish]),
            )
            player.sell(0)
            shop_pets = player.shop.pets
            self.assertEqual(
                [stats(p) for p in shop_pets], [(2, 2 + level), (1, 4 + level)]
            )
            ### Team pets are not shop pets
            self.assertEqual(stats(fish), (2, 3))
            self.assertEqual(player.gold, 10 + level)


class TestOtter(AuditCase):
    """Buy: Give one/two/three random friends +1 health."""

    def buy_otter(self, level, friends):
        shop = Shop(["otter"])
        shop[0].obj.level = level
        player = Player(shop=shop, team=Team(friends))
        otter = shop[0].obj
        player.buy_pet(0)
        return player, otter

    def test_level_1_one_friend_of_two(self):
        for seed in range(10):
            np.random.seed(seed)
            friends = [pet("fish"), pet("ant")]
            _, otter = self.buy_otter(1, friends)
            gained = sorted(f.health - b for f, b in zip(friends, [3, 2]))
            self.assertEqual(gained, [0, 1])
            self.assertEqual([f.attack for f in friends], [2, 2])
            self.assertEqual(stats(otter), (1, 4))

    def test_level_2_two_friends(self):
        friends = [pet("fish"), pet("ant")]
        _, otter = self.buy_otter(2, friends)
        self.assertEqual([stats(f) for f in friends], [(2, 4), (2, 3)])
        self.assertEqual(stats(otter), (1, 4))

    def test_level_3_three_friends(self):
        friends = [pet("fish"), pet("ant"), pet("pig")]
        _, otter = self.buy_otter(3, friends)
        self.assertEqual([stats(f) for f in friends], [(2, 4), (2, 3), (4, 2)])
        self.assertEqual(stats(otter), (1, 4))

    def test_level_3_with_fewer_friends(self):
        friends = [pet("fish")]
        _, otter = self.buy_otter(3, friends)
        self.assertEqual(stats(friends[0]), (2, 4))
        self.assertEqual(stats(otter), (1, 4))

    def test_no_friends(self):
        player, otter = self.buy_otter(1, [])
        self.assertEqual(stats(otter), (1, 4))
        self.assertEqual(player.gold, 7)


class TestPig(AuditCase):
    """Sell: Gain +1/+2/+3 gold (on top of the sell value, which is the level)."""

    def test_extra_gold(self):
        for level in [1, 2, 3]:
            player = Player(team=Team([pet("pig", level)]))
            player.sell(0)
            self.assertEqual(player.gold, 10 + level + level, level)


class TestPigeon(AuditCase):
    """Sell: Stock one/two/three free Bread Crumbs."""

    def test_level_2(self):
        player = Player(shop=Shop(["ant"]), team=Team([pet("pigeon", 2)]))
        player.sell(0)
        crumbs = [s for s in player.shop if s.obj.name == "food-bread-crumbs"]
        self.assertEqual(len(crumbs), 2)
        self.assertTrue(all(s.cost == 0 for s in crumbs))

    def test_full_shop_stocks_only_what_fits(self):
        ### 7 is the most shop slots: one free slot means one Bread Crumbs
        shop = Shop(["ant", "ant", "ant", "ant", "ant", "apple"])
        player = Player(shop=shop, team=Team([pet("pigeon", 3)]))
        player.sell(0)
        crumbs = [s for s in player.shop if s.obj.name == "food-bread-crumbs"]
        self.assertEqual(len(crumbs), 1)
        self.assertEqual(len(player.shop), 7)

    def test_crumbs_are_free_and_give_attack(self):
        player = Player(shop=Shop(["ant"]), team=Team([pet("pigeon"), pet("fish")]))
        player.sell(0)
        idx = [
            i for i, s in enumerate(player.shop) if s.obj.name == "food-bread-crumbs"
        ]
        fish = team_pets(player)[0]
        player.buy_food(idx[0], fish)
        self.assertEqual(player.gold, 11)
        self.assertEqual(stats(fish), (3, 3))


class TestSwan(AuditCase):
    """Start of turn: Gain +1/+2/+3 gold."""

    def test_gold(self):
        for level in [1, 2, 3]:
            player = Player(team=Team([pet("swan", level)]))
            player.gold = 0
            player.start_turn()
            self.assertEqual(player.gold, 10 + level, level)

    def test_two_swans(self):
        player = Player(team=Team([pet("swan"), pet("swan", 2)]))
        player.start_turn()
        self.assertEqual(player.gold, 13)


class TestWorm(AuditCase):
    """Start of turn: Stock one Apple / Better Apple / Best Apple that costs 2."""

    def test_level_3_best_apple(self):
        worm = pet("worm", 3)
        player = Player(team=Team([worm]))
        player.start_turn()
        apples = [s for s in player.shop if s.obj.name == "food-best-apple"]
        self.assertEqual(len(apples), 1)
        self.assertEqual(apples[0].cost, 2)
        player.buy_food(apples[0].obj, 0)
        self.assertEqual(player.gold, 8)
        self.assertEqual(stats(worm), (4, 7))

    def test_level_1_apple_only_one(self):
        player = Player(team=Team([pet("worm")]))
        player.start_turn()
        foods = [s for s in player.shop if s.slot_type == "food"]
        cheap = [s for s in foods if s.obj.name == "food-apple" and s.cost == 2]
        self.assertEqual(len(cheap), 1)


class TestLateGameStock(AuditCase):
    """From turn 9 the shop has 5 pet and 2 food slots, which is the engine's
    7-slot shop limit (docs/RULES.md 2.8, marked "assumed"). The Worm and
    Pigeon texts have no condition, so they should still stock their food."""

    # OPEN QUESTION (docs/RULES.md Q8): sapai/effects.py StockFood stops when the shop already
    # has RULES["max_shop_slots"] (7) slots, so from turn 9 the Worm stocks
    # nothing. This rests on the assumed 7-slot limit (RULES.md 2.8).
    @unittest.expectedFailure
    def test_worm_on_turn_9(self):
        player = Player(team=Team([pet("worm")]), shop=Shop(turn=8), turn=8)
        player.start_turn()
        self.assertEqual(player.turn, 9)
        cheap = [s for s in player.shop if s.obj.name == "food-apple" and s.cost == 2]
        self.assertEqual(len(cheap), 1)

    # OPEN QUESTION (docs/RULES.md Q8): same cause as the Worm test above.
    @unittest.expectedFailure
    def test_pigeon_on_turn_9(self):
        player = Player(team=Team([pet("pigeon")]), shop=Shop(turn=9), turn=9)
        player.sell(0)
        crumbs = [s for s in player.shop if s.obj.name == "food-bread-crumbs"]
        self.assertEqual(len(crumbs), 1)


class TestSnail(AuditCase):
    """End turn: If you lost the last battle, give the three nearest friends
    ahead +1/+2/+3 attack."""

    def make(self, level, lf_winner):
        friends = [pet("pig"), pet("fish"), pet("ant"), pet("beaver")]
        snail = pet("snail", level)
        player = Player(team=Team(friends + [snail]), lf_winner=lf_winner)
        return player, friends, snail

    def test_after_loss(self):
        for level in [1, 2, 3]:
            player, friends, snail = self.make(level, False)
            player.end_turn()
            ### Pig is the fourth friend ahead and gets nothing
            self.assertEqual(
                [f.attack for f in friends], [4, 2 + level, 2 + level, 3 + level]
            )
            self.assertEqual([f.health for f in friends], [1, 3, 2, 2])
            self.assertEqual(stats(snail), (2, 3))

    def test_after_win_or_draw_or_first_turn(self):
        ### Match passes winner=True for a win and for a draw
        for lf_winner in [True, None]:
            player, friends, _ = self.make(1, lf_winner)
            player.end_turn()
            self.assertEqual([f.attack for f in friends], [4, 2, 2, 3])

    def test_snail_in_front_has_no_friends_ahead(self):
        snail, fish = pet("snail"), pet("fish")
        player = Player(team=Team([snail, fish]), lf_winner=False)
        player.end_turn()
        self.assertEqual(stats(fish), (2, 3))
        self.assertEqual(stats(snail), (2, 3))


class TestFish(AuditCase):
    """Level-up: Give two friends +1/+1 (level 1 to 2) or +2/+2 (2 to 3)."""

    def test_only_one_friend(self):
        fish = Pet("fish")
        fish.experience = 1
        ant = pet("ant")
        player = Player(shop=Shop(["fish"]), team=Team([fish, ant]))
        player.buy_combine(0, 0)
        self.assertEqual(fish.level, 2)
        self.assertEqual(stats(ant), (3, 3))

    def test_no_friends(self):
        fish = Pet("fish")
        fish.experience = 1
        player = Player(shop=Shop(["fish"]), team=Team([fish]))
        player.buy_combine(0, 0)
        ### Merging gives +1/+1; the ability has no one to buff
        self.assertEqual((fish.level, stats(fish)), (2, (3, 4)))

    def test_exactly_two_of_three_friends(self):
        for seed in range(10):
            np.random.seed(seed)
            fish = Pet("fish")
            fish.experience = 1
            friends = [pet("pig"), pet("pig"), pet("pig")]
            player = Player(shop=Shop(["fish"]), team=Team([fish] + friends))
            player.buy_combine(0, 0)
            gained = sorted((f.attack - 4, f.health - 1) for f in friends)
            self.assertEqual(gained, [(0, 0), (1, 1), (1, 1)])

    def test_combine_two_team_fish(self):
        a = Pet("fish")
        b = Pet("fish")
        b.experience = 1
        ant, beaver = pet("ant"), pet("beaver")
        player = Player(team=Team([a, ant, b, beaver]))
        player.combine(0, 2)
        self.assertEqual(a.level, 2)
        self.assertEqual(stats(ant), (3, 3))
        self.assertEqual(stats(beaver), (4, 3))

    def test_level_up_by_chocolate(self):
        fish = Pet("fish")
        fish.experience = 1
        ant, beaver = pet("ant"), pet("beaver")
        player = Player(shop=Shop(["chocolate"]), team=Team([fish, ant, beaver]))
        player.buy_food(0, 0)
        self.assertEqual(fish.level, 2)
        self.assertEqual(stats(ant), (3, 3))
        self.assertEqual(stats(beaver), (4, 3))
        ### Chocolate gives no stats; the Fish does not buff itself
        self.assertEqual(stats(fish), (2, 3))


class TestHorse(AuditCase):
    """Friend summoned: Give it +1/+2/+3 attack until end of battle."""

    def test_bought_pet_gets_attack_until_next_turn(self):
        for level in [1, 2, 3]:
            horse = pet("horse", level)
            player = Player(shop=Shop(["fish"]), team=Team([horse]))
            fish = player.shop[0].obj
            player.buy_pet(0)
            self.assertEqual(stats(fish), (2 + level, 3), level)
            self.assertEqual(stats(horse), (2, 1))
            ### "until end of battle": gone at the start of the next turn
            player.start_turn()
            self.assertEqual(stats(fish), (2, 3))

    def test_cricket_fainting_in_the_shop(self):
        horse = pet("horse")
        cricket = pet("cricket")
        player = Player(shop=Shop(["sleeping-pill"]), team=Team([cricket, horse]))
        player.buy_food(0, 0)
        zombies = [p for p in team_pets(player) if p.name == "pet-zombie-cricket"]
        self.assertEqual(len(zombies), 1)
        self.assertEqual(stats(zombies[0]), (2, 1))


class TestShopFaints(AuditCase):
    """Faint abilities also work when a Sleeping Pill knocks a pet out."""

    def test_ant(self):
        fish = pet("fish")
        player = Player(shop=Shop(["sleeping-pill"]), team=Team([pet("ant", 2), fish]))
        player.buy_food(0, 0)
        self.assertEqual(stats(fish), (4, 5))

    def test_cricket_summons_in_its_place(self):
        fish = pet("fish")
        player = Player(
            shop=Shop(["sleeping-pill"]), team=Team([pet("cricket", 3), fish])
        )
        player.buy_food(0, 0)
        pets = team_pets(player)
        self.assertEqual([p.name for p in pets], ["pet-zombie-cricket", "pet-fish"])
        self.assertEqual(stats(pets[0]), (3, 3))

    def test_hedgehog_hurts_friends(self):
        fish, ant = pet("fish"), pet("ant")
        player = Player(
            shop=Shop(["sleeping-pill"]), team=Team([pet("hedgehog"), fish, ant])
        )
        player.buy_food(0, 0)
        ### Fish 2/1; Ant faints from the damage and gives the Fish +1/+1
        self.assertEqual(stats(fish), (3, 2))
        self.assertEqual([p.name for p in team_pets(player)], ["pet-fish"])

    def test_flamingo(self):
        fish, ant, pig = pet("fish"), pet("ant"), pet("pig")
        player = Player(
            shop=Shop(["sleeping-pill"]),
            team=Team([pet("flamingo"), fish, ant, pig]),
        )
        player.buy_food(0, 0)
        self.assertEqual(
            [stats(fish), stats(ant), stats(pig)], [(3, 4), (3, 3), (4, 1)]
        )

    def test_peacock_hurt_in_shop(self):
        peacock = pet("peacock")
        player = Player(
            shop=Shop(["sleeping-pill"]), team=Team([pet("hedgehog"), peacock])
        )
        player.buy_food(0, 0)
        self.assertEqual(stats(peacock), (5, 3))


class TestSpider(AuditCase):
    """Faint: Summon one random level 1/2/3 tier 3 pet as a 2/2 / 4/4 / 6/6."""

    def test_summon(self):
        for level in [1, 2, 3]:
            for seed in range(5):
                np.random.seed(seed)
                spider = pet("spider", level)
                fish = pet("fish")
                t = Team([spider, fish], battle=True)
                spider._health = 0
                spider.faint_trigger(spider, [0, 0])
                pets = [s.pet for s in t if not s.empty]
                self.assertEqual(len(pets), 2)
                summoned = pets[0]
                self.assertIs(pets[1], fish)
                self.assertEqual(data["pets"][summoned.name]["tier"], 3)
                self.assertEqual(summoned.level, level)
                self.assertEqual(stats(summoned), (2 * level, 2 * level))

    def test_summon_in_shop(self):
        player = Player(
            shop=Shop(["sleeping-pill"]), team=Team([pet("spider", 2), pet("fish")])
        )
        player.buy_food(0, 0)
        pets = team_pets(player)
        self.assertEqual(len(pets), 2)
        self.assertEqual(data["pets"][pets[0].name]["tier"], 3)
        self.assertEqual((pets[0].level, stats(pets[0])), (2, (4, 4)))


if __name__ == "__main__":
    unittest.main()
