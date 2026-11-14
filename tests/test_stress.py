import unittest


class TestStress(unittest.TestCase):
    def test_random_battles_and_matches_break_no_rules(self):
        from sapai.stress import run

        problems, examples, checks = run(battles=150, games=4, seed=3)
        self.assertEqual(dict(problems), {}, examples)
        self.assertEqual(checks["battles"], 150)
        self.assertGreater(checks["battles without chance (side check)"], 50)
        self.assertGreater(checks["shop actions"], 20)


if __name__ == "__main__":
    unittest.main()
