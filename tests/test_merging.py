import unittest

from sapai.pets import Pet
from sapai.player import Player
from sapai.teams import Team


def pet(name, level=1, experience=0):
    p = Pet(name)
    p.level = level
    p.experience = experience
    return p


def merge(a, b):
    """Merge team pet 1 into team pet 0. Returns the kept pet and the number
    of level-up reward slots added to the shop"""
    player = Player(team=Team([a, b]))
    before = len(player.shop.slots)
    player.combine(0, 1)
    return player.team[0].pet, len(player.shop.slots) - before


class TestMergingAddsExperience(unittest.TestCase):
    def check(self, a, b, level, experience, rewards):
        kept, added = merge(pet("pig", *a), pet("pig", *b))
        self.assertEqual((kept.level, kept.experience), (level, experience))
        self.assertEqual(added, rewards)

    def test_two_single_pets(self):
        self.check((1, 0), (1, 0), 1, 1, 0)

    def test_a_pair_onto_a_single_pet_levels_up(self):
        self.check((1, 0), (1, 1), 2, 0, 1)

    def test_two_pairs(self):
        self.check((1, 1), (1, 1), 2, 1, 1)

    def test_level_2_onto_a_single_pet(self):
        self.check((1, 0), (2, 0), 2, 1, 1)

    def test_two_level_2_pets_reach_level_3(self):
        self.check((2, 1), (2, 1), 3, 0, 1)

    def test_from_level_1_to_level_3_gives_two_rewards(self):
        self.check((1, 1), (2, 2), 3, 0, 2)

    def test_copies(self):
        self.assertEqual(pet("pig").copies, 1)
        self.assertEqual(pet("pig", 1, 1).copies, 2)
        self.assertEqual(pet("pig", 2).copies, 3)
        self.assertEqual(pet("pig", 2, 2).copies, 5)
        self.assertEqual(pet("pig", 3).copies, 6)

    def test_level_3_keeps_no_experience(self):
        p = pet("pig", 3)
        p.gain_experience(2)
        self.assertEqual((p.level, p.experience), (3, 0))

    def test_merging_keeps_the_higher_stats_plus_one(self):
        a, b = pet("pig"), pet("pig")
        a._attack, a._health = 5, 2
        b._attack, b._health = 3, 7
        kept, _ = merge(a, b)
        self.assertEqual((kept._attack, kept._health), (6, 8))

    def test_merging_adds_one_per_copy_merged_in(self):
        ### Two level 2 pets: the merged-in pet is worth 3 copies, so +3/+3
        a, b = pet("pig", 2), pet("pig", 2)
        a._attack, a._health = 6, 4
        b._attack, b._health = 5, 7
        kept, _ = merge(a, b)
        self.assertEqual((kept._attack, kept._health), (9, 10))
        ### A pair (2 copies) onto a single pet: +2/+2
        a, b = pet("pig"), pet("pig", 1, 1)
        a._attack, a._health = 3, 3
        b._attack, b._health = 4, 4
        kept, _ = merge(a, b)
        self.assertEqual((kept._attack, kept._health), (6, 6))
        ### Never above 50
        a, b = pet("pig", 2), pet("pig", 2)
        a._attack, a._health = 49, 49
        kept, _ = merge(a, b)
        self.assertEqual((kept._attack, kept._health), (50, 50))

    def test_fish_levelup_ability_fires_when_merging_levels_it_up(self):
        fish, ant = pet("fish", 1, 1), pet("ant")
        player = Player(team=Team([fish, pet("fish"), ant]))
        player.combine(0, 1)
        self.assertEqual(player.team[0].pet.level, 2)
        ### Fish level 1: give two friends +1/+1 (only the Ant is left)
        ant_after = [s.pet for s in player.team if s.pet.name == "pet-ant"][0]
        self.assertEqual(
            (ant_after._attack, ant_after._health),
            (Pet("ant")._attack + 1, Pet("ant")._health + 1),
        )


if __name__ == "__main__":
    unittest.main()
