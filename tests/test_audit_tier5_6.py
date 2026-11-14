"""
Audit of tier 5 and tier 6 shop abilities (buy, sell, food, end of turn)
against their ability text in data/turtle_pack.yaml. Expected values are
worked out by hand from the text; tests marked expectedFailure show a place
where the engine does not do what the text says.
"""

import unittest

from sapai import Pet, Player, Shop, Team


def stats(pet):
    return (pet.attack, pet.health)


def pet(name, level=1, attack=None, health=None, status=None):
    p = Pet(name)
    p.level = level
    if attack is not None:
        p._attack = attack
    if health is not None:
        p._health = health
    if status is not None:
        p.status = status
    return p


################################################################################
#### Scorpion
################################################################################


class TestScorpion(unittest.TestCase):
    def test_bought_scorpion_has_peanut(self):
        ### Buying counts as being summoned
        player = Player(shop=Shop(["scorpion"]))
        player.buy_pet(0)
        self.assertEqual(player.team[0].pet.status, "status-poison-attack")

    ### FIXED (was an engine mismatch): Pet.friend_summoned_trigger (sapai/pets.py) does not
    ###   check triggeredBy, so the Scorpion's own "Summoned" (Self) ability
    ###   fires whenever any friend is bought/summoned and replaces its perk
    ###   with Peanut.
    def test_buying_a_friend_does_not_give_scorpion_peanut(self):
        scorpion = pet("scorpion", status="status-garlic-armor")
        player = Player(shop=Shop(["fish"]), team=Team([scorpion]))
        player.buy_pet(0)
        self.assertEqual(scorpion.status, "status-garlic-armor")


################################################################################
#### Monkey
################################################################################


class TestMonkey(unittest.TestCase):
    def test_levels(self):
        for level, amount in [(1, 2), (2, 4), (3, 6)]:
            fish = pet("fish")
            player = Player(team=Team([fish, pet("pig"), pet("monkey", level)]))
            player.end_turn()
            self.assertEqual(stats(fish), (2 + amount, 3 + amount))
            self.assertEqual(stats(player.team[1].pet), (4, 1))

    def test_front_most_friend_behind_gaps(self):
        monkey = pet("monkey")
        fish = pet("fish")
        player = Player(team=Team([fish, monkey]))
        player.team.move_backward()
        player.end_turn()
        self.assertEqual(stats(fish), (4, 5))
        self.assertEqual(stats(monkey), (1, 2))

    ### FIXED (was an engine mismatch): target RightMostFriend (sapai/effects.py get_target)
    ###   is fidx[0] and does not skip the Monkey itself, so a Monkey in front
    ###   buffs itself instead of the front-most friend.
    def test_monkey_in_front_buffs_a_friend_not_itself(self):
        monkey = pet("monkey")
        fish = pet("fish")
        player = Player(team=Team([monkey, fish]))
        player.end_turn()
        self.assertEqual(stats(monkey), (1, 2))
        self.assertEqual(stats(fish), (4, 5))

    ### FIXED (was an engine mismatch): same as above; with no friends the Monkey buffs itself
    def test_monkey_alone_does_nothing(self):
        monkey = pet("monkey")
        player = Player(team=Team([monkey]))
        player.end_turn()
        self.assertEqual(stats(monkey), (1, 2))


################################################################################
#### Cow
################################################################################


class TestCow(unittest.TestCase):
    def test_better_milk_replaces_all_food(self):
        cow = pet("cow", 2)
        player = Player(shop=Shop([cow, "fish", "apple", "pear"]))
        player.buy_pet(0)
        foods = [x for x in player.shop if x.slot_type == "food"]
        self.assertEqual([x.obj.name for x in foods], ["food-better-milk"] * 2)
        self.assertEqual([x.cost for x in foods], [0, 0])
        ### Other shop pets are left alone
        self.assertEqual(
            [x.obj.name for x in player.shop if x.slot_type == "pet"], ["pet-fish"]
        )
        gold = player.gold
        player.buy_food(foods[0], 0)
        self.assertEqual(player.gold, gold)
        self.assertEqual(stats(player.team[0].pet), (4 + 2, 6 + 4))

    def test_milk_added_when_shop_had_no_food(self):
        player = Player(shop=Shop(["cow"]))
        player.buy_pet(0)
        self.assertEqual([x.obj.name for x in player.shop], ["food-milk"] * 2)

    def test_cow_bought_to_merge_still_triggers(self):
        cow = pet("cow")
        player = Player(shop=Shop(["cow", "apple"]), team=Team([cow]))
        player.buy_combine(0, 0)
        foods = [x.obj.name for x in player.shop if x.slot_type == "food"]
        self.assertEqual(len(foods), 2)
        self.assertTrue(all("milk" in x for x in foods))


################################################################################
#### Seal
################################################################################


class TestSeal(unittest.TestCase):
    def test_three_friends_each_gain(self):
        for level in [1, 2, 3]:
            seal = pet("seal", level)
            friends = [pet("fish"), pet("pig"), pet("duck")]
            player = Player(shop=Shop(["apple"]), team=Team([seal] + friends))
            player.buy_food(0, 0)
            self.assertEqual(stats(seal), (3 + 1, 8 + 1))
            self.assertEqual(
                [stats(x) for x in friends],
                [(2 + level, 3), (4 + level, 1), (2 + level, 2)],
            )

    def test_four_friends_three_gain(self):
        seal = pet("seal")
        friends = [pet("fish"), pet("pig"), pet("duck"), pet("beaver")]
        player = Player(shop=Shop(["apple"]), team=Team([seal] + friends))
        player.buy_food(0, 0)
        gains = [x.attack - Pet(x.name).attack for x in friends]
        self.assertEqual(sorted(gains), [0, 1, 1, 1])
        self.assertEqual(stats(seal), (4, 9))

    def test_fewer_friends(self):
        seal = pet("seal")
        fish = pet("fish")
        player = Player(shop=Shop(["apple"]), team=Team([seal, fish]))
        player.buy_food(0, 0)
        self.assertEqual(stats(fish), (3, 3))

        seal = pet("seal")
        player = Player(shop=Shop(["apple"]), team=Team([seal]))
        player.buy_food(0, 0)
        self.assertEqual(stats(seal), (4, 9))

    def test_food_for_a_friend_does_not_trigger(self):
        seal = pet("seal")
        fish = pet("fish")
        pig = pet("pig")
        player = Player(shop=Shop(["apple"]), team=Team([seal, fish, pig]))
        player.buy_food(0, 1)
        self.assertEqual(stats(fish), (3, 4))
        self.assertEqual(stats(pig), (4, 1))
        self.assertEqual(stats(seal), (3, 8))


################################################################################
#### Turkey (shop side)
################################################################################


class TestTurkeyShop(unittest.TestCase):
    def test_bought_friend_is_summoned(self):
        for level, a, h in [(1, 3, 1), (2, 6, 2), (3, 9, 3)]:
            player = Player(shop=Shop(["fish"]), team=Team([pet("turkey", level)]))
            player.buy_pet(0)
            fish = [x.pet for x in player.team if x.pet.name == "pet-fish"][0]
            self.assertEqual(stats(fish), (2 + a, 3 + h))
            ### Permanent, not until end of battle
            self.assertEqual(fish._until_end_of_battle_attack_buff, 0)

    def test_turkey_does_not_buff_itself_when_bought(self):
        player = Player(shop=Shop(["turkey"]), team=Team([pet("fish")]))
        player.buy_pet(0)
        turkey = [x.pet for x in player.team if x.pet.name == "pet-turkey"][0]
        self.assertEqual(stats(turkey), (3, 4))


################################################################################
#### Dragon
################################################################################


class TestDragon(unittest.TestCase):
    def test_buy_tier_1_gives_all_friends(self):
        for level in [1, 2, 3]:
            dragon = pet("dragon", level)
            fish = pet("fish")
            player = Player(shop=Shop(["ant"]), team=Team([dragon, fish]))
            player.buy_pet(0)
            ant = [x.pet for x in player.team if x.pet.name == "pet-ant"][0]
            self.assertEqual(stats(fish), (2 + level, 3 + level))
            ### The bought pet is a friend too
            self.assertEqual(stats(ant), (2 + level, 2 + level))
            self.assertEqual(stats(dragon), (3, 8))

    def test_buy_tier_2_does_nothing(self):
        dragon = pet("dragon")
        fish = pet("fish")
        player = Player(shop=Shop(["hedgehog"]), team=Team([dragon, fish]))
        player.buy_pet(0)
        self.assertEqual(stats(fish), (2, 3))

    def test_four_times_per_turn(self):
        dragon = pet("dragon")
        fish = pet("fish")
        player = Player(shop=Shop(["ant"] * 5), team=Team([dragon, fish]))
        player.gold = 30
        for _ in range(5):
            player.buy_pet(0)
            ant = [x.pet for x in player.team if x.pet.name == "pet-ant"][0]
            player.sell(ant)
        self.assertEqual(stats(fish), (2 + 4, 3 + 4))

        ### Works again next turn
        player.end_turn()
        player.start_turn()
        player.shop = Shop(["ant"])
        player.gold = 10
        for slot in player.team:
            slot.pet.shop = player.shop
        player.buy_pet(0)
        self.assertEqual(stats(fish), (2 + 5, 3 + 5))

    def test_buying_tier_1_to_merge_triggers(self):
        dragon = pet("dragon")
        ant = pet("ant")
        player = Player(shop=Shop(["ant"]), team=Team([dragon, ant]))
        player.buy_combine(0, 1)
        ### Merge: 3/3, then +1/+1 from Dragon
        self.assertEqual(stats(ant), (4, 4))


################################################################################
#### Cat
################################################################################


class TestCat(unittest.TestCase):
    def test_multiplier_per_level(self):
        for level, mult in [(1, 2), (2, 3), (3, 4)]:
            cat = pet("cat", level)
            player = Player(shop=Shop(["pear"]), team=Team([cat]))
            player.buy_food(0, 0)
            self.assertEqual(stats(cat), (4 + 2 * mult, 5 + 2 * mult))

    def test_cupcake_doubled_until_end_of_battle(self):
        cat = pet("cat")
        player = Player(shop=Shop(["cupcake"]), team=Team([cat]))
        player.buy_food(0, 0)
        self.assertEqual(stats(cat), (4 + 6, 5 + 6))
        self.assertEqual((cat._attack, cat._health), (4, 5))

    def test_milk_from_cow_is_doubled(self):
        cat = pet("cat")
        player = Player(shop=Shop(["cow"]), team=Team([cat]))
        player.buy_pet(0)
        player.buy_food(0, 0)
        self.assertEqual(stats(cat), (4 + 2, 5 + 4))

    def test_salad_doubled_for_each_pet(self):
        cat = pet("cat")
        fish = pet("fish")
        player = Player(shop=Shop(["salad-bowl"]), team=Team([cat, fish]))
        player.buy_food(0)
        self.assertEqual(stats(cat), (6, 7))
        self.assertEqual(stats(fish), (4, 5))

    def test_resets_next_turn(self):
        cat = pet("cat")
        player = Player(shop=Shop(["apple"] * 2), team=Team([cat]))
        player.buy_food(0, 0)
        player.buy_food(0, 0)
        player.end_turn()
        player.start_turn()
        player.shop = Shop(["apple"])
        player.buy_food(0, 0)
        self.assertEqual(stats(cat), (4 + 6, 5 + 6))

    ### FIXED (was an engine mismatch): Pet.cat_trigger (sapai/pets.py) spends one of the two
    ###   uses on every food bought, including perk foods like Honey that give
    ###   no attack or health, so the second Apple is not doubled.
    def test_perk_food_does_not_use_a_trigger(self):
        cat = pet("cat")
        player = Player(shop=Shop(["honey", "apple", "apple"]), team=Team([cat]))
        player.buy_food(0, 0)
        player.buy_food(0, 0)
        player.buy_food(0, 0)
        self.assertEqual(stats(cat), (4 + 2 + 2, 5 + 2 + 2))


################################################################################
#### Faint pets in the shop (Sleeping Pill)
################################################################################


class TestFaintInShop(unittest.TestCase):
    def test_mammoth_pill(self):
        fish = pet("fish")
        player = Player(
            shop=Shop(["sleeping-pill"]), team=Team([pet("mammoth", 2), fish])
        )
        player.buy_food(0, 0)
        self.assertEqual(stats(fish), (6, 7))

    def test_rooster_pill(self):
        player = Player(shop=Shop(["sleeping-pill"]), team=Team([pet("rooster", 2)]))
        player.buy_food(0, 0)
        chicks = [x.pet for x in player.team if not x.empty]
        self.assertEqual([x.name for x in chicks], ["pet-chick"] * 2)
        self.assertEqual([stats(x) for x in chicks], [(3, 1), (3, 1)])

    def test_shark_and_fly_pill(self):
        shark = pet("shark")
        fly = pet("fly")
        player = Player(
            shop=Shop(["sleeping-pill"]), team=Team([pet("fish"), shark, fly])
        )
        player.buy_food(0, 0)
        self.assertEqual(stats(shark), (4, 4))
        names = [x.pet.name for x in player.team if not x.empty]
        self.assertEqual(names, ["pet-zombie-fly", "pet-shark", "pet-fly"])


if __name__ == "__main__":
    unittest.main()
