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
