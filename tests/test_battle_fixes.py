import unittest

from sapai.battle import Battle
from sapai.pets import Pet
from sapai.teams import Team

from gui import game


def team(specs):
    t = Team([Pet(n) for n, _, _ in specs])
    for slot, (_, a, h) in zip(t, specs):
        slot.pet._attack = a
        slot.pet._health = h
    return t


class TestRhino(unittest.TestCase):
    def test_rhino_chains_knockouts(self):
        a = team([("rhino", 5, 8)])
        b = team([("ant", 2, 1), ("fish", 2, 3), ("beaver", 3, 2)])
        self.assertEqual(Battle(a, b).battle(), 0)

    def test_replay_shows_each_hit_before_the_faint_it_causes(self):
        frames, _ = game.replay_battle(team([("rhino", 5, 8)]),
                                       team([("ant", 2, 1), ("fish", 2, 3), ("beaver", 3, 2)]))
        order = [l for f in frames for l in f["lines"]]
        hit_fish = next(i for i, l in enumerate(order) if "Rhino deals damage to Fish" in l)
        fish_faints = next(i for i, l in enumerate(order) if "Fish faints" in l)
        hit_beaver = next(i for i, l in enumerate(order) if "Rhino deals damage to Beaver" in l)
        beaver_faints = next(i for i, l in enumerate(order) if "Beaver faints" in l)
        self.assertLess(hit_fish, fish_faints)
        self.assertLess(fish_faints, hit_beaver)
        self.assertLess(hit_beaver, beaver_faints)


class TestCrab(unittest.TestCase):
    def test_crab_keeps_at_least_one_health(self):
        """25% of a 1-3 health friend rounds to 0; the Crab must not faint from it."""
        frames, _ = game.replay_battle(team([("rhino", 1, 50)]), team([("pig", 1, 3), ("crab", 3, 5)]))
        start = next(f for f in frames if f["label"].startswith("Start of battle"))
        crab = next(p for p in start["teams"][1] if p["name"] == "crab")
        self.assertEqual(crab["health"], 1)


if __name__ == "__main__":
    unittest.main()
