import unittest

from sapai.trigger_check import EXPECTED, PETS, run

### A representative set of neighbours: every Sell pet (the Duck behind a Pig
###   bug), summoners, buffers, Whale and Tiger
OTHERS = [
    "pet-pig",
    "pet-duck",
    "pet-beaver",
    "pet-pigeon",
    "pet-cricket",
    "pet-sheep",
    "pet-horse",
    "pet-ant",
    "pet-hedgehog",
    "pet-whale",
    "pet-tiger",
    "pet-turtle",
]


class TestNoPetBlocksAnother(unittest.TestCase):
    def test_every_ability_still_fires_next_to_other_pets(self):
        problems, checked = run(PETS, OTHERS)
        self.assertGreater(checked, 1000)
        unexpected = [x for x in problems if x[2] not in EXPECTED]
        self.assertEqual(unexpected, [])


if __name__ == "__main__":
    unittest.main()


class TestCheckerSituations(unittest.TestCase):
    """The checker's own situations really set off the abilities they test"""

    def test_tiger_repeats_the_ability_ahead(self):
        from sapai.trigger_check import TIGER, chain

        team_for, enemy = TIGER[("StartOfBattle", "Self")]
        run = chain(team_for, enemy, "tiger_repeat")
        self.assertEqual(run("pet-dolphin", "pet-fish", False), (True, True))

    def test_wolverine_fires_on_the_fourth_hurt(self):
        from sapai.trigger_check import battle_wolverine

        self.assertEqual(
            battle_wolverine("pet-wolverine", "pet-fish", True), (True, True)
        )

    def test_chains_reach_their_trigger(self):
        from sapai.trigger_check import SITUATIONS

        for key, pet in [
            (("Hurt", "Self"), "pet-peacock"),
            (("Faint", "Self"), "pet-ant"),
            (("Summoned", "EachFriend"), "pet-horse"),
        ]:
            for label, situation in SITUATIONS[key]:
                with self.subTest(pet=pet, situation=label):
                    result = situation(pet, "pet-fish", False, seed=0)
                    self.assertEqual(result, (True, True))

    def test_a_blocked_ability_is_reported(self):
        from unittest import mock

        from sapai.pets import Pet
        from sapai.trigger_check import run

        original = Pet.hurt_trigger

        def blocked_next_to_ox(self, *args, **kwargs):
            result = original(self, *args, **kwargs)
            if any(not s.empty and s.pet.name == "pet-ox" for s in self.team):
                return False, [], []
            return result

        with mock.patch.object(Pet, "hurt_trigger", blocked_next_to_ox):
            problems, _ = run(["pet-peacock"], ["pet-ox", "pet-ant"])
        self.assertTrue(problems)
        self.assertEqual({x[2] for x in problems}, {"pet-ox"})
