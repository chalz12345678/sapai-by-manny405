"""
Audit of tier 3 and tier 4 pets whose abilities work outside battle (shop,
start of turn, end of turn), plus a few battle details the scenario format
cannot express. Every expectation comes from the ability text in
data/turtle_pack.yaml. Battle abilities are in tests/scenarios/tier3_4.yaml.

Tests marked expectedFailure show a place where the engine does not do what
the text says; the comment above each one explains the mismatch.
"""

import unittest

from sapai.battle import Battle
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


def stats(p):
    if not isinstance(p, Pet):
        p = p.pet
    return (p.attack, p.health)


def filled(team):
    return [x.pet for x in team if not x.empty]


################################################################################
# Giraffe: "Start of turn: Give the nearest / two nearest / three nearest
# friends ahead +1 attack and +1 health."
################################################################################


class TestGiraffe(unittest.TestCase):
    def run_giraffe(self, level):
        team = [pet("fish"), pet("ant"), pet("beaver"), pet("giraffe", level)]
        player = Player(team=Team(team))
        player.start_turn()
        return [stats(x) for x in filled(player.team)]

    def test_level_1_buffs_nearest_friend_ahead(self):
        self.assertEqual(self.run_giraffe(1), [(2, 3), (2, 2), (4, 3), (1, 2)])

    def test_level_2_buffs_two_nearest_friends_ahead(self):
        self.assertEqual(self.run_giraffe(2), [(2, 3), (3, 3), (4, 3), (1, 2)])

    def test_level_3_buffs_three_nearest_friends_ahead(self):
        self.assertEqual(self.run_giraffe(3), [(3, 4), (3, 3), (4, 3), (1, 2)])

    def test_giraffe_in_front_buffs_nobody(self):
        player = Player(team=Team([pet("giraffe", 3), pet("fish")]))
        player.start_turn()
        self.assertEqual([stats(x) for x in filled(player.team)], [(1, 2), (2, 3)])

    def test_buff_stops_at_50(self):
        player = Player(team=Team([pet("fish", 1, 50, 49), pet("giraffe")]))
        player.start_turn()
        self.assertEqual(stats(player.team[0]), (50, 50))


################################################################################
# Rabbit: "Friendly ate food: Give them +1/+2/+3 health. Works 3 times per
# turn."
################################################################################


class TestRabbit(unittest.TestCase):
    def test_amount_per_level(self):
        for level in [1, 2, 3]:
            player = Player(
                shop=Shop(["apple"]), team=Team([pet("rabbit", level), pet("fish")])
            )
            player.buy_food(0, 1)
            ### Apple +1/+1, Rabbit +level health
            self.assertEqual(stats(player.team[1]), (3, 4 + level), level)

    def test_level_3_works_three_times_per_turn(self):
        player = Player(
            shop=Shop(["apple"] * 4), team=Team([pet("rabbit", 3), pet("fish")])
        )
        player.gold = 20
        for _ in range(4):
            player.buy_food(0, 1)
        self.assertEqual(stats(player.team[1]), (2 + 4, 3 + 4 + 3 * 3))


################################################################################
# Dog: "Friend summoned: Gain +2/+1, +4/+2, +6/+3 until next turn."
# Buying a pet summons it to the team.
################################################################################


class TestDogInShop(unittest.TestCase):
    def test_buying_a_pet_buffs_dog_until_next_turn(self):
        for level, (a, h) in [(1, (2, 1)), (2, (4, 2)), (3, (6, 3))]:
            player = Player(shop=Shop(["fish"]), team=Team([pet("dog", level)]))
            player.buy_pet(0)
            dog = player.team[0].pet
            self.assertEqual(stats(dog), (3 + a, 2 + h), level)
            ### The buff lasts through the battle ...
            player.end_turn()
            self.assertEqual(stats(dog), (3 + a, 2 + h), level)
            ### ... and is gone next turn
            player.start_turn()
            self.assertEqual(stats(dog), (3, 2), level)

    def test_buying_the_dog_itself_does_not_trigger(self):
        player = Player(shop=Shop(["dog"]), team=Team([pet("fish")]))
        player.buy_pet(0)
        dog = [x for x in filled(player.team) if x.name == "pet-dog"][0]
        self.assertEqual(stats(dog), (3, 2))


################################################################################
# Bison: "End turn: If there is at least one level 3 friend, gain +2/+2,
# +4/+4, +6/+6."
################################################################################


class TestBison(unittest.TestCase):
    def test_gains_with_level_3_friend(self):
        for level, gain in [(1, 2), (2, 4), (3, 6)]:
            player = Player(team=Team([pet("fish", 3), pet("bison", level)]))
            player.end_turn()
            self.assertEqual(stats(player.team[1]), (4 + gain, 4 + gain), level)

    def test_no_gain_without_level_3_friend(self):
        player = Player(team=Team([pet("fish", 2), pet("bison")]))
        player.end_turn()
        self.assertEqual(stats(player.team[1]), (4, 4))

    # FIXED (was an engine mismatch): the text needs a level 3 *friend*, but Team.check_lvl3
    # (sapai/teams.py) also counts the Bison itself, so a level 3 Bison with
    # no level 3 friend gains +6/+6 every turn.
    def test_level_3_bison_does_not_count_itself(self):
        player = Player(team=Team([pet("fish", 1), pet("bison", 3)]))
        player.end_turn()
        self.assertEqual(stats(player.team[1]), (4, 4))

    def test_level_3_bison_alone_does_not_gain(self):
        player = Player(team=Team([pet("bison", 3)]))
        player.end_turn()
        self.assertEqual(stats(player.team[0]), (4, 4))


################################################################################
# Squirrel: "Start of turn: Discount shop food by 1/2/3 gold."
################################################################################


class TestSquirrel(unittest.TestCase):
    def test_discount_per_level(self):
        for level in [1, 2, 3]:
            player = Player(team=Team([pet("squirrel", level)]))
            player.start_turn()
            foods = [x for x in player.shop if x.slot_type == "food"]
            self.assertTrue(len(foods) > 0)
            for slot in foods:
                self.assertEqual(slot.cost, max(slot.obj.cost - level, 0), level)
            ### The discount is spent when buying
            slot = foods[0]
            gold = player.gold
            player.buy_food(slot.obj, 0)
            self.assertEqual(player.gold, gold - max(slot.obj.cost - level, 0))


################################################################################
# Penguin: "Start of turn: Give two level 2+ friends +1/+1, +2/+2, +3/+3."
################################################################################


class TestPenguin(unittest.TestCase):
    def test_buffs_the_two_level_2_friends(self):
        for level in [1, 2, 3]:
            team = [pet("fish", 2), pet("ant"), pet("beaver", 3), pet("penguin", level)]
            player = Player(team=Team(team))
            player.start_turn()
            got = [stats(x) for x in filled(player.team)]
            self.assertEqual(
                got,
                [(2 + level, 3 + level), (2, 2), (3 + level, 2 + level), (2, 3)],
                level,
            )

    def test_only_two_of_three_level_2_friends(self):
        for seed in range(10):
            team = [pet("fish", 2), pet("ant", 2), pet("beaver", 3), pet("penguin")]
            player = Player(team=Team(team))
            player.start_turn()
            base = [(2, 3), (2, 2), (3, 2)]
            got = [stats(x) for x in filled(player.team)][:3]
            buffed = [g != b for g, b in zip(got, base)]
            self.assertEqual(sum(buffed), 2)
            for g, b, was in zip(got, base, buffed):
                if was:
                    self.assertEqual(g, (b[0] + 1, b[1] + 1))

    def test_level_2_penguin_does_not_buff_itself(self):
        player = Player(team=Team([pet("fish", 2), pet("penguin", 2)]))
        player.start_turn()
        self.assertEqual([stats(x) for x in filled(player.team)], [(4, 5), (2, 3)])


################################################################################
# Parrot: "End turn: Copy the ability of the nearest friend ahead as level
# 1/2/3 until end of battle."
################################################################################


class TestParrot(unittest.TestCase):
    def sob_damage(self, dolphin_level, parrot_level):
        team = [pet("dolphin", dolphin_level), pet("parrot", parrot_level)]
        player = Player(team=Team(team))
        player.end_turn()
        b = Battle(player.team, Team([pet("pig", 1, 1, 50)]))
        b.start()
        return 50 - b.t1[0].pet.health

    def test_copies_at_parrot_level(self):
        ### Dolphin L2 shoots twice; the Parrot shoots 1/2/3 times
        self.assertEqual(self.sob_damage(2, 1), 4 * (2 + 1))
        self.assertEqual(self.sob_damage(2, 3), 4 * (2 + 3))
        self.assertEqual(self.sob_damage(3, 1), 4 * (3 + 1))

    def test_no_friend_ahead_copies_nothing(self):
        player = Player(team=Team([pet("parrot")]))
        player.end_turn()
        self.assertEqual(player.team[0].pet.ability["trigger"], "EndOfTurn")

    # FIXED (was an engine mismatch): the copy should last "until end of battle", but
    # TransferAbility (sapai/effects.py) sets override_ability on the team's
    # Parrot and nothing ever clears it (battles run on copies; start_turn
    # does not reset it). The Parrot keeps the copied ability for the rest of
    # the game and loses its own End turn ability.
    def test_parrot_has_its_own_ability_again_next_turn(self):
        player = Player(team=Team([pet("fish"), pet("parrot")]))
        player.end_turn()
        player.start_turn()
        self.assertEqual(player.team[1].pet.ability["trigger"], "EndOfTurn")

    def test_parrot_copies_the_new_friend_ahead_next_turn(self):
        player = Player(team=Team([pet("fish"), pet("dolphin"), pet("parrot")]))
        ### Turn 1: copies Dolphin
        player.end_turn()
        player.start_turn()
        ### Turn 2: Dolphin is sold, Fish is now ahead of the Parrot
        player.sell(1)
        player.team.move_forward()
        player.end_turn()
        parrot = [x for x in filled(player.team) if x.name == "pet-parrot"][0]
        self.assertEqual(parrot.ability["trigger"], "LevelUp")


################################################################################
# Skunk: "Start of battle: Remove 33% of health from the highest health enemy."
################################################################################


class TestSkunkTemporaryHealth(unittest.TestCase):
    # FIXED (was an engine mismatch): ReduceHealth (sapai/effects.py) computes the new health
    # from pet.health (which includes "until next turn" buffs such as Bread
    # or Dog) but writes it to pet._health, so the temporary buff is counted
    # twice. A Pig with 3 health + 7 from Bread (10 shown) should drop to 7,
    # the engine gives it 14.
    def test_skunk_on_pet_with_bread(self):
        enemy = Player(shop=Shop(["bread"]), team=Team([pet("pig", 1, 1, 3)]))
        enemy.buy_food(0, 0)
        enemy.end_turn()
        self.assertEqual(enemy.team[0].pet.health, 10)
        b = Battle(Team([pet("skunk")]), enemy.team)
        b.start()
        self.assertEqual(b.t1[0].pet.health, 7)


################################################################################
# Whale: "Start of battle: Swallow the nearest friend ahead. When this faints,
# release it as a level 1/2/3 pet." (Only the level is checked: the text does
# not say what stats the released pet has.)
################################################################################


class TestWhaleReleaseLevel(unittest.TestCase):
    def test_release_level_matches_whale_level(self):
        for level in [1, 2, 3]:
            team = Team([pet("fish"), pet("whale", level, 3, 1)])
            b = Battle(team, Team([pet("pig", 1, 1, 2)]))
            ### R1: Whale and Pig knock each other out, Fish is released
            self.assertEqual(b.battle(), 0)
            released = filled(b.t0)
            self.assertEqual([x.name for x in released], ["pet-fish"], level)
            self.assertEqual(released[0].level, level)


if __name__ == "__main__":
    unittest.main()
