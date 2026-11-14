import os
import tempfile
import unittest

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


@unittest.skipIf(yaml is None, "pyyaml is required to read scenarios")
class TestScenarios(unittest.TestCase):
    def test_every_scenario_matches_the_engine(self):
        from sapai.scenarios import check, load

        scenarios = load()
        self.assertGreater(len(scenarios), 0)
        for s in scenarios:
            with self.subTest(f"{s['file']}: {s['name']}"):
                if s.get("discrepancy"):
                    ### A battle from the real game the engine can't reproduce
                    ###   yet: once it can, drop the discrepancy mark
                    self.assertNotEqual(
                        check(s),
                        [],
                        "the engine now reproduces this battle: remove"
                        " 'discrepancy: true' and move it to from_game.yaml",
                    )
                else:
                    self.assertEqual(check(s), [])

    def test_a_wrong_expectation_fails(self):
        from sapai.scenarios import check

        s = {"name": "x", "team": ["Fish 2/3"], "enemy": ["Fish 2/3"]}
        self.assertNotEqual(check(dict(s, result="win")), [])
        self.assertNotEqual(
            check(dict(s, result="draw", survivors={"team": ["Fish 2/1"]})), []
        )
        self.assertEqual(check(dict(s, result="draw")), [])

    def test_hidden_randomness_is_reported(self):
        from sapai.scenarios import check

        s = {
            "name": "x",
            "team": ["Mosquito 2/3"],
            "enemy": ["Ant 2/1", "Fish 2/1"],
            "runs": 30,
        }
        self.assertTrue(any("random" in p for p in check(s)))

    def test_seen_battles_pass_when_the_engine_can_end_that_way(self):
        from sapai.scenarios import check

        s = {
            "name": "x",
            "seen": True,
            "team": ["Mosquito 2/3"],
            "enemy": ["Ant 2/1", "Fish 2/1"],
            "runs": 60,
        }
        ### Either ending can happen (Mosquito picks its target at random)
        self.assertEqual(check(dict(s, result="win")), [])
        self.assertEqual(check(dict(s, result="draw")), [])
        self.assertNotEqual(check(dict(s, result="loss")), [])

    def test_pet_spec(self):
        from sapai.scenarios import parse_pet

        p = parse_pet("Hedgehog 3/2 L2 Meat Bone")
        self.assertEqual(
            p,
            {
                "id": "pet-hedgehog",
                "attack": 3,
                "health": 2,
                "level": 2,
                "perk": "status-bone-attack",
            },
        )
        with self.assertRaises(ValueError):
            parse_pet("Unicorn 1/1")
        with self.assertRaises(ValueError):
            parse_pet("Fish 2/3 Ketchup")

    def test_load_one_file(self):
        from sapai.scenarios import load

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "x.yaml")
            with open(path, "w") as f:
                f.write("- team: [Fish 2/3]\n  enemy: [Fish 2/3]\n  result: draw\n")
            [s] = load([path])
            self.assertEqual(s["file"], "x.yaml")
            self.assertEqual(s["name"], "x.yaml #1")


if __name__ == "__main__":
    unittest.main()
